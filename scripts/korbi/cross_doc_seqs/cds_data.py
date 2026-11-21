"""Offline replay of the main run's training data stream (docs/cross_doc_seqs/README.md, step 1).

Reproduces, without a GPU and without building anything, which document pieces Megatron feeds at
every global step of a phase:

    step -> 1024 blended sample indices      (MegatronPretrainingReferenceLayoutSampler, replayed)
    blended index -> (dataset, sample)       (BlendedDataset dataset_index / dataset_sample_index)
    sample -> document pieces                (GPTDataset shuffle_index / sample_index / document_index)

All index arrays are read (memory-mapped) from the run's --data-cache-path; the blend entry is
found by its description (paths, sequence length, size), never rebuilt. Document coordinates are
Megatron's VIRTUAL ones: with --dataloader-add-bos offset 0 is the BOS Megatron prepends at read
time and token k of the .bin document is offset k + 1 (gpt_dataset.py _get_document).

"Owned" tokens: a sample is sequence_length + 1 tokens and consecutive samples of a dataset share
one token (the label of the last position). A piece list here covers the sample's first
sequence_length tokens only -- every token is owned by exactly one sample, so the pieces of a set
of samples partition their tokens and a repacking can conserve them exactly.

numpy only, plus torch for the sampler's torch.randperm (bit-exact with the training container's
torch; check with --check-megatron or MEGATRON_SAMPLE_DUMP_DIR).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

import numpy as np

_IDX_MAGIC = b"MMIDIDX\x00\x00"
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.float32,
           8: np.uint16}


# --------------------------------------------------------------------------------------------- data
class IndexedTokens:
    """Read-only view of a Megatron MMap indexed dataset (<prefix>.idx / .bin)."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        with open(prefix + ".idx", "rb") as f:
            assert f.read(9) == _IDX_MAGIC, f"{prefix}.idx: not a Megatron MMap index"
            (version,) = struct.unpack("<Q", f.read(8))
            assert version == 1, f"{prefix}.idx: version {version}"
            (code,) = struct.unpack("<B", f.read(1))
            self.dtype = np.dtype(_DTYPES[code])
            self.n_seq, self.n_doc = struct.unpack("<QQ", f.read(16))
            self._off = f.tell()

    @cached_property
    def _idx(self) -> np.memmap:
        return np.memmap(self.prefix + ".idx", mode="r", dtype=np.uint8)

    @cached_property
    def lengths(self) -> np.ndarray:
        return np.frombuffer(self._idx, np.int32, self.n_seq, self._off)

    def load_lengths(self) -> None:
        """Read the lengths into memory with one sequential read (random page faults into GPFS
        memory maps cost milliseconds each; the replay looks up ~55 lengths per sequence)."""
        self.__dict__["lengths"] = np.fromfile(self.prefix + ".idx", np.int32, self.n_seq, offset=self._off)

    @cached_property
    def pointers(self) -> np.ndarray:
        return np.frombuffer(self._idx, np.int64, self.n_seq, self._off + 4 * self.n_seq)

    @cached_property
    def _bin(self) -> np.memmap:
        return np.memmap(self.prefix + ".bin", mode="r", dtype=self.dtype)

    def get(self, seq: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Real tokens [offset, offset + length) of sequence ``seq`` (no virtual BOS)."""
        n = int(self.lengths[seq])
        length = n - offset if length is None else min(length, n - offset)
        start = int(self.pointers[seq]) // self.dtype.itemsize + offset
        return np.asarray(self._bin[start:start + max(length, 0)])


def read_blend(path: str) -> tuple[list[float], list[str]]:
    """--data-args-path file: whitespace separated "weight prefix" pairs (get_blend_from_list)."""
    toks = open(path).read().split()
    assert len(toks) % 2 == 0, f"{path}: odd number of tokens"
    return [float(w) for w in toks[0::2]], [p.strip() for p in toks[1::2]]


# ------------------------------------------------------------------------------------------- cache
def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    cache_dir: str
    blend_md5: str
    desc: dict  # the BlendedDataset description (OrderedDict, as Megatron wrote it)

    @property
    def size(self) -> int:
        return int(self.desc["size"])

    @property
    def prefixes(self) -> list[str]:
        return [d["dataset_path"] for d in self.desc["datasets"]]

    @property
    def sequence_length(self) -> int:
        return int(self.desc["datasets"][0]["sequence_length"])

    @property
    def add_bos(self) -> bool:
        return bool(self.desc["datasets"][0].get("add_bos", False))

    def gpt_md5(self, i: int) -> str:
        return _md5(json.dumps(self.desc["datasets"][i], indent=4))

    def path(self, kind: str, md5: str, name: str) -> str:
        return os.path.join(self.cache_dir, f"{md5}-{kind}-train-{name}.npy")


def find_cache_entry(cache_dir: str, prefixes: list[str], sequence_length: int,
                     size: int | None = None, blend_md5: str | None = None) -> CacheEntry:
    """The BlendedDataset-train cache entry of this blend: same dataset paths (in order) and sequence
    length (and size, if given). Several matches (e.g. other train_iters) are an error unless ``size``
    or ``blend_md5`` picks one. Verifies that the file name is the md5 of the description."""
    if blend_md5:
        cands = [os.path.join(cache_dir, f"{blend_md5}-BlendedDataset-train-description.txt")]
    else:
        cands = sorted(glob.glob(os.path.join(cache_dir, "*-BlendedDataset-train-description.txt")))
    hits = []
    for c in cands:
        text = open(c).read()
        desc = json.loads(text, object_pairs_hook=OrderedDict)
        ds = desc.get("datasets") or []
        if [d.get("dataset_path") for d in ds] != prefixes:
            continue
        if any(int(d.get("sequence_length", -1)) != sequence_length for d in ds):
            continue
        if size is not None and int(desc["size"]) != size:
            continue
        md5 = os.path.basename(c).split("-")[0]
        assert _md5(text) == md5, f"{c}: description does not hash to its file name"
        hits.append(CacheEntry(cache_dir, md5, desc))
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} BlendedDataset cache entries match in {cache_dir} "
                         f"(size={size}): {[h.blend_md5 for h in hits]}")
    return hits[0]


# ----------------------------------------------------------------------------------------- sampler
@dataclass(frozen=True)
class SamplerGeometry:
    """The run's sampler: cyclic + data_sharding + reference layout (data_samplers.py
    MegatronPretrainingReferenceLayoutSampler); a phase with its own dataset starts at
    ``phase_start`` (the consumed position inside the phase is (iteration - phase_start) x gbs)."""

    total_samples: int  # len(BlendedDataset)
    global_batch_size: int
    phase_start: int
    ref_dp: int
    ref_mbs: int
    ref_gbs: int | None = None

    @property
    def gbs_ref(self) -> int:
        return self.ref_gbs or self.global_batch_size

    @property
    def mb_per_epoch(self) -> int:
        return self.total_samples // (self.ref_dp * self.ref_mbs)


class StepSampler:
    """Blended sample indices of every global step, positions in data-parallel rank order (the
    actual layout dp x mbs = gbs x 1 gives rank d position d; any other actual layout splits the
    same list in order)."""

    def __init__(self, geo: SamplerGeometry):
        self.geo = geo
        assert geo.gbs_ref % (geo.ref_dp * geo.ref_mbs) == 0
        self._perm_epoch, self._perm = None, None

    def _perm_for(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            import torch

            g = torch.Generator()
            g.manual_seed(int(epoch))
            self._perm = torch.randperm(self.geo.mb_per_epoch * self.geo.ref_mbs, generator=g).numpy()
            self._perm_epoch = epoch
        return self._perm

    def step_after(self, ckpt: int) -> np.ndarray:
        """The samples of the step taken from a checkpoint at iteration ``ckpt`` (the step Megatron
        logs as "iteration ckpt + 1"): consumed position in the phase (ckpt - phase_start) x gbs."""
        g = self.geo
        q = (ckpt - g.phase_start) * g.global_batch_size + np.arange(g.global_batch_size, dtype=np.int64)
        assert q[0] >= 0, f"checkpoint {ckpt} is before the phase start {g.phase_start}"
        per_ref_rank = (g.gbs_ref // (g.ref_dp * g.ref_mbs)) * g.ref_mbs
        ref_step, pos = np.divmod(q, g.gbs_ref)
        ref_rank, rem = np.divmod(pos, per_ref_rank)
        k, i = np.divmod(rem, g.ref_mbs)
        ref_mb = ref_step * (g.gbs_ref // (g.ref_dp * g.ref_mbs)) + k
        epoch, j = np.divmod(ref_mb, g.mb_per_epoch)
        out = np.empty_like(q)
        for e in np.unique(epoch):
            sel = epoch == e
            perm = self._perm_for(int(e))
            out[sel] = ref_rank[sel] * (g.mb_per_epoch * g.ref_mbs) + perm[j[sel] * g.ref_mbs + i[sel]]
        return out

    def check_against_megatron(self, ckpt: int) -> None:
        """Compare with Megatron's own sampler (needs megatron on the path)."""
        from megatron.training.datasets.data_samplers import MegatronPretrainingReferenceLayoutSampler

        g = self.geo
        s = MegatronPretrainingReferenceLayoutSampler(
            None, total_samples=g.total_samples,
            consumed_samples=(ckpt - g.phase_start) * g.global_batch_size,
            micro_batch_size=g.global_batch_size, global_batch_size=g.global_batch_size,
            data_parallel_rank=0, data_parallel_size=1, data_sharding=True,
            reference_dp_size=g.ref_dp, reference_micro_batch_size=g.ref_mbs,
            reference_global_batch_size=g.ref_gbs)
        ref = np.asarray(next(iter(s)), dtype=np.int64)
        mine = self.step_after(ckpt)
        assert np.array_equal(ref, mine), f"sampler replay differs from Megatron after checkpoint {ckpt}"


# ------------------------------------------------------------------------------------------ stream
class BlendStream:
    """Sample -> document pieces for one cached blend."""

    def __init__(self, entry: CacheEntry):
        self.entry = entry
        e = entry
        self.dataset_index = np.load(e.path("BlendedDataset", e.blend_md5, "dataset_index"), mmap_mode="r")
        self.dataset_sample_index = np.load(e.path("BlendedDataset", e.blend_md5, "dataset_sample_index"),
                                            mmap_mode="r")
        self.seq = e.sequence_length
        self.bos = 1 if e.add_bos else 0
        self._gpt: dict[int, tuple] = {}
        self.tokens = [IndexedTokens(p) for p in e.prefixes]

    def _arrays(self, ds: int, mmap: bool = True):
        if ds not in self._gpt:
            e, md5 = self.entry, self.entry.gpt_md5(ds)
            self._gpt[ds] = tuple(np.load(e.path("GPTDataset", md5, n), mmap_mode="r" if mmap else None)
                                  for n in ("document_index", "sample_index", "shuffle_index"))
        return self._gpt[ds]

    def preload(self, datasets=None, threads: int = 32) -> float:
        """Read the blend's and the given datasets' index arrays and document lengths into memory
        (parallel sequential reads; all datasets by default). Returns GB read."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.time()
        e = self.entry
        self.dataset_index = np.load(e.path("BlendedDataset", e.blend_md5, "dataset_index"))
        self.dataset_sample_index = np.load(e.path("BlendedDataset", e.blend_md5, "dataset_sample_index"))
        todo = sorted(set(range(len(self.tokens))) if datasets is None else set(int(d) for d in datasets))

        def one(ds):
            self._gpt.pop(ds, None)
            arrs = self._arrays(ds, mmap=False)
            self.tokens[ds].load_lengths()
            return sum(a.nbytes for a in arrs) + self.tokens[ds].lengths.nbytes

        with ThreadPoolExecutor(threads) as pool:
            gb = sum(pool.map(one, todo)) / 1e9
        gb += (self.dataset_index.nbytes + self.dataset_sample_index.nbytes) / 1e9
        print(f"preloaded {len(todo)} datasets' indices: {gb:.1f} GB in {time.time() - t0:.0f} s", flush=True)
        return gb

    def pieces(self, bidx: int) -> tuple[int, list[tuple[int, int, int]]]:
        """(dataset, [(doc, virtual offset, length), ...]) of blended sample ``bidx``: its first
        ``seq`` (owned) tokens; the extra label token is left out."""
        ds = int(self.dataset_index[bidx])
        dsi = int(self.dataset_sample_index[bidx])
        doc_index, sample_index, shuffle_index = self._arrays(ds)
        lens = self.tokens[ds].lengths
        i = int(shuffle_index[dsi])
        d0, o0 = (int(x) for x in sample_index[i])
        d1, o1 = (int(x) for x in sample_index[i + 1])
        out = []
        if d0 == d1:
            out.append((int(doc_index[d0]), o0, o1 - o0 + 1))
        else:
            doc = int(doc_index[d0])
            out.append((doc, o0, int(lens[doc]) + self.bos - o0))
            for d in range(d0 + 1, d1):
                doc = int(doc_index[d])
                out.append((doc, 0, int(lens[doc]) + self.bos))
            out.append((int(doc_index[d1]), 0, o1 + 1))
        # drop the label-only token (Megatron's add_extra_token_to_sequence)
        doc, off, n = out[-1]
        out[-1] = (doc, off, n - 1)
        out = [p for p in out if p[2] > 0]
        assert sum(p[2] for p in out) == self.seq, (bidx, out)
        return ds, out

    def read(self, ds: int, doc: int, off: int, n: int, bos_id: int | None) -> np.ndarray:
        """Tokens of a piece in virtual coordinates (BOS = ``bos_id`` at offset 0 with add_bos)."""
        if not self.bos:
            return self.tokens[ds].get(doc, off, n)
        if off > 0:
            return self.tokens[ds].get(doc, off - 1, n)
        inner = self.tokens[ds].get(doc, 0, n - 1)
        return np.concatenate([np.array([bos_id], dtype=inner.dtype), inner])


# ------------------------------------------------------------------------------------- stream files
STREAM_KEYS = ("slot_iter", "slot_pos", "slot_bidx", "seg_ptr", "seg_ds", "seg_doc", "seg_off", "seg_len")


def save_stream(path: str, arrays: dict, meta: dict) -> None:
    """One extracted step range: <path>/{<key>.npy, meta.json}, written atomically (tmp dir + rename)."""
    tmp = path + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for k in STREAM_KEYS:
        np.save(os.path.join(tmp, k + ".npy"), arrays[k])
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, path)


def load_stream(path: str) -> tuple[dict, dict]:
    arrays = {k: np.load(os.path.join(path, k + ".npy"), mmap_mode="r") for k in STREAM_KEYS}
    return arrays, json.load(open(os.path.join(path, "meta.json")))


def stream_dir(out: str, a: int, b: int) -> str:
    """The steps taken from checkpoints a .. b-1 (logged iterations a+1 .. b)."""
    return os.path.join(out, "stream", f"ck{a:07d}_{b:07d}")
