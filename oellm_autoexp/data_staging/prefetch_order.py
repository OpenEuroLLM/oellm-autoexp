"""Reconstruct the deterministic ``.bin`` access order of a Megatron
GPTDataset.

Mirrors, without importing Megatron:
  * the ``.idx`` binary format               (indexed_dataset.py _IndexReader)
  * the sample->document->byte resolution    (gpt_dataset.py _query_document_sample_shuffle_indices)
  * the default pretraining sampler          (data_samplers.py MegatronPretrainingSampler)

Given the dataset prefix and the cached index ``.npy`` files, this yields the
exact ordered list of ``.bin`` byte ranges a run touches, so a prefetcher can
warm/stage them just ahead of the training cursor.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

# .idx header: see megatron/core/datasets/indexed_dataset.py
_INDEX_HEADER = b"MMIDIDX\x00\x00"
# DType enum code -> itemsize in bytes (DType in indexed_dataset.py)
_DTYPE_ITEMSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}


class IdxFile:
    """Parsed ``.idx``: per-document byte pointer + token length, and item
    size."""

    def __init__(self, idx_path: str | Path) -> None:
        idx_path = str(idx_path)
        with open(idx_path, "rb") as f:
            assert f.read(9) == _INDEX_HEADER, f"bad .idx header: {idx_path}"
            (version,) = struct.unpack("<Q", f.read(8))
            assert version == 1, f"unsupported .idx version {version}: {idx_path}"
            (code,) = struct.unpack("<B", f.read(1))
            self.itemsize = _DTYPE_ITEMSIZE[code]
            (self.sequence_count,) = struct.unpack("<Q", f.read(8))
            (self.document_count,) = struct.unpack("<Q", f.read(8))
            header_end = f.tell()
        buf = np.memmap(idx_path, mode="r", order="C")
        # Layout after the header: int32 lengths, then int64 byte pointers.
        self.sequence_lengths = np.frombuffer(
            buf, dtype=np.int32, count=self.sequence_count, offset=header_end
        )
        self.sequence_pointers = np.frombuffer(
            buf,
            dtype=np.int64,
            count=self.sequence_count,
            offset=header_end + self.sequence_lengths.nbytes,
        )

    def doc_byte_range(self, doc_id: int) -> tuple[int, int]:
        """(byte_offset, byte_length) of document ``doc_id`` in the
        ``.bin``."""
        off = int(self.sequence_pointers[doc_id])
        nbytes = int(self.sequence_lengths[doc_id]) * self.itemsize
        return off, nbytes


def load_cached_indices(cache_dir: str | Path, desc_hash: str, split: str = "train"):
    """Load the three cached index arrays (memory-mapped, read-only).

    ``desc_hash`` is the md5 prefix in the cache filenames, e.g.
    ``09a1affae47eef300894d3d78388acf6``.
    """
    base = f"{desc_hash}-GPTDataset-{split}"
    d = Path(cache_dir)
    document_index = np.load(d / f"{base}-document_index.npy", mmap_mode="r")
    sample_index = np.load(d / f"{base}-sample_index.npy", mmap_mode="r")
    shuffle_index = np.load(d / f"{base}-shuffle_index.npy", mmap_mode="r")
    return document_index, sample_index, shuffle_index


def discover_desc_hash(cache_dir: str | Path, split: str = "train") -> str:
    """Find the (single) cached description hash for ``split`` in
    ``cache_dir``.

    Raises if zero or more than one is present (caller must
    disambiguate).
    """
    d = Path(cache_dir)
    suffix = f"-GPTDataset-{split}-shuffle_index.npy"
    hits = sorted(p.name[: -len(suffix)] for p in d.glob(f"*{suffix}"))
    if len(hits) != 1:
        raise ValueError(
            f"expected exactly one {split} index in {cache_dir}, found {len(hits)}: {hits}"
        )
    return hits[0]


def discover_hashes_by_prefix(cache_dir, prefixes, split: str = "train") -> dict:
    """Map each dataset prefix to its GPTDataset cache hash, by matching the
    prefix path inside the cached description.txt files.

    For a blend, the cache holds one GPTDataset per sub-dataset.
    """
    import glob

    descs = {}
    for p in glob.glob(os.path.join(str(cache_dir), f"*-GPTDataset-{split}-description.txt")):
        h = os.path.basename(p).split("-GPTDataset")[0]
        try:
            descs[h] = open(p).read()
        except OSError:
            pass
    out = {}
    for prefix in prefixes:
        name = os.path.basename(prefix)
        hits = [h for h, t in descs.items() if prefix in t] or [
            h for h, t in descs.items() if name in t
        ]
        if not hits:
            raise ValueError(
                f"no GPTDataset {split} description matches prefix {prefix} in {cache_dir}"
            )
        out[prefix] = hits[0]
    return out


def load_blend_weights(cache_dir, prefixes, split: str = "train") -> dict:
    """Per-file fraction of the blend from the BlendedDataset dataset_index
    (count of samples drawn from each sub-dataset).

    Falls back to equal weights when there is no blend cache (single
    dataset).
    """
    import glob

    di_files = glob.glob(
        os.path.join(str(cache_dir), f"*-BlendedDataset-{split}-dataset_index.npy")
    )
    if not di_files:
        return {p: 1.0 / len(prefixes) for p in prefixes}
    di = np.load(di_files[0], mmap_mode="r")
    counts = np.bincount(np.asarray(di), minlength=len(prefixes)).astype(np.float64)
    total = max(1.0, counts.sum())
    return {prefixes[i]: float(counts[i] / total) for i in range(len(prefixes))}


def compute_lane_bounds(document_index, sample_index, shuffle_len, num_lanes, lane_block_docs):
    """Per-file lane geometry over ONE epoch.

    Returns (epoch_samples Me, K, bounds[K+1], docs_per_sample). Detects
    the epoch size from the sample_index doc-column wrap (file-order
    document_index per epoch).
    """
    S = int(shuffle_len)
    doc_col = np.asarray(sample_index[:S, 0])
    drops = np.nonzero(np.diff(doc_col) < 0)[0]
    Me = int(drops[0]) + 1 if len(drops) else S
    K = max(1, min(num_lanes, Me))
    Ms = Me // K
    bounds = [int(document_index[int(sample_index[j * Ms][0])]) for j in range(K)]
    bounds.append(int(document_index[int(sample_index[min(K * Ms, Me)][0])]))
    r = max(1e-9, (bounds[-1] - bounds[0]) / max(1, K * Ms))
    return Me, K, bounds, r


def node_sample_order(
    num_samples_total: int,
    dp_size: int,
    node_dp_ranks: list[int],
    micro_batch_size: int,
    start_sample: int = 0,
):
    """Yield global dataset sample indices in the order the given data-
    parallel.

    ranks consume them, matching MegatronPretrainingSampler (dataloader_type
    'single'): within each contiguous block of ``mbs * dp_size`` samples, dp rank
    ``r`` consumes ``[r*mbs : (r+1)*mbs]``. Blocks advance in order.
    """
    block = micro_batch_size * dp_size
    idx = start_sample
    while idx + block <= num_samples_total:
        for r in node_dp_ranks:
            base = idx + r * micro_batch_size
            yield from range(base, base + micro_batch_size)
        idx += block


def sample_byte_ranges(
    global_idx: int, document_index, sample_index, shuffle_index, idx_file: IdxFile
):
    """List of (byte_offset, byte_length) for the documents sample
    ``global_idx`` reads."""
    sidx = int(shuffle_index[global_idx])
    doc_beg = int(sample_index[sidx][0])
    doc_end = int(sample_index[sidx + 1][0])
    ranges = []
    for i in range(doc_beg, doc_end + 1):
        ranges.append(idx_file.doc_byte_range(int(document_index[i])))
    return ranges


def sample_doc_ids(global_idx: int, document_index, sample_index, shuffle_index):
    """List of document ids that sample ``global_idx`` reads (in order)."""
    sidx = int(shuffle_index[global_idx])
    doc_beg = int(sample_index[sidx][0])
    doc_end = int(sample_index[sidx + 1][0])
    return [int(document_index[i]) for i in range(doc_beg, doc_end + 1)]


def iter_access_docs(
    document_index,
    sample_index,
    shuffle_index,
    dp_size: int,
    node_dp_ranks: list[int],
    micro_batch_size: int,
    start_sample: int = 0,
    num_samples_total: int | None = None,
):
    """Yield the document ids the node reads, in consumption order.

    This is the stream the document-granular mirror prefetcher walks (it
    resolves each id to a byte range via the .idx).
    """
    if num_samples_total is None:
        num_samples_total = len(shuffle_index)
    for gidx in node_sample_order(
        num_samples_total, dp_size, node_dp_ranks, micro_batch_size, start_sample
    ):
        yield from sample_doc_ids(gidx, document_index, sample_index, shuffle_index)


def iter_access_ranges(
    idx_file: IdxFile,
    document_index,
    sample_index,
    shuffle_index,
    dp_size: int,
    node_dp_ranks: list[int],
    micro_batch_size: int,
    start_sample: int = 0,
    num_samples_total: int | None = None,
):
    """Yield (byte_offset, byte_length) for every ``.bin`` read the node
    performs, in consumption order.

    This is the stream a prefetcher walks.
    """
    if num_samples_total is None:
        num_samples_total = len(shuffle_index)
    for gidx in node_sample_order(
        num_samples_total, dp_size, node_dp_ranks, micro_batch_size, start_sample
    ):
        yield from sample_byte_ranges(gidx, document_index, sample_index, shuffle_index, idx_file)
