#!/usr/bin/env python3
"""Step 2 of the cross-document packing pipeline (docs/cross_doc_seqs/README.md): one embedding per
document that the extracted steps read (the documents of non-solo slots; a solo slot is one long
document filling a whole sequence and is never repacked).

    srun -n <gpus> embed_docs.py --out <pipeline dir> --name q3e256 \
        --model /gpfs/projects/ehpc533/users/korbip/models/Qwen3-Embedding-0.6B --pooling last \
        --tokenizer /gpfs/projects/ehpc533/users/korbip/data/tokenizers/tokenizer-256k-agent \
        --dim 256 --max-src-tokens 640 --max-len 512

The text of a document is its first --max-src-tokens training tokens (the .bin, no virtual BOS),
decoded with the training tokenizer and re-tokenized for the embedding model (truncated at
--max-len). One process per GPU (RANK / WORLD_SIZE / LOCAL_RANK, else SLURM_PROCID / ...); process r embeds keys
r, r + world, ... of every stream chunk and writes <out>/emb/<name>/ck<a>_<b>/part_<r>_of_<world>.npz
{keys: int64 dataset << 40 | document, vec: float16 [n, dim]} atomically. Existing parts are skipped,
so the job can be restarted / extended (new stream chunks) at any time.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cds_data import IndexedTokens, load_stream  # noqa: E402

KEY_SHIFT = 40


def chunk_keys(stream_path: str) -> np.ndarray:
    """Sorted unique document keys of the chunk's non-solo slots."""
    arr, _ = load_stream(stream_path)
    nseg = np.diff(np.asarray(arr["seg_ptr"]))
    seg_slot = np.repeat(np.arange(len(nseg)), nseg)
    keep = nseg[seg_slot] > 1
    ds = np.asarray(arr["seg_ds"])[keep].astype(np.int64)
    doc = np.asarray(arr["seg_doc"])[keep].astype(np.int64)
    return np.unique((ds << KEY_SHIFT) | doc)


class DocTexts:
    """Batches of (keys, model inputs), built in DataLoader workers. A batch arrives with every
    document's (dataset, byte offset, token count) resolved in the parent, so a worker does one read
    per document, all of a batch's reads in flight at once (GPFS random reads are latency-bound)."""

    def __init__(self, prefixes, dtypes, a):
        self.prefixes, self.dtypes, self.a = prefixes, dtypes, a
        self._tok = None
        self._fds: dict = {}
        self._pool = None

    def _init(self):
        # (re)initialize per process: a DataLoader worker forked from a process that already used this
        # object inherits a thread pool without threads (its reads would wait forever) and shared fds
        if getattr(self, "_pid", None) != os.getpid():
            self._tok, self._fds, self._pool, self._pid = None, {}, None, os.getpid()
        if self._tok is None:
            from concurrent.futures import ThreadPoolExecutor

            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.a.tokenizer)
            self._model_tok = AutoTokenizer.from_pretrained(self.a.model)
            # does the tokenizer append its own end token (Qwen3-Embedding: <|endoftext|>, the state
            # last-token pooling reads)? Then nothing is appended here.
            mt = self._model_tok
            self._adds_end = len(mt("a")["input_ids"]) > len(mt("a", add_special_tokens=False)["input_ids"])
            self._pool = ThreadPoolExecutor(self.a.read_threads)

    def _read(self, ds: int, off: int, n: int) -> list:
        fd = self._fds.get(ds)
        if fd is None:
            fd = self._fds[ds] = os.open(self.prefixes[ds] + ".bin", os.O_RDONLY)
        dt = self.dtypes[ds]
        return np.frombuffer(os.pread(fd, n * dt.itemsize, off), dtype=dt).tolist()

    def batch(self, item):
        self._init()
        keys, ds, off, ntok = item
        a = self.a
        toks = list(self._pool.map(self._read, ds.tolist(), off.tolist(), ntok.tolist()))
        texts = self._tok.batch_decode(toks, skip_special_tokens=True)
        texts = [a.prefix + t for t in texts]
        mt = self._model_tok
        enc = mt(texts, add_special_tokens=True, truncation=True, max_length=a.max_len - 1)["input_ids"]
        eos = mt.eos_token_id
        if a.pooling == "last" and not self._adds_end and eos is not None:  # pool on an end token
            enc = [e + [eos] for e in enc]
        if a.pack:  # padding-free: documents concatenated, varlen flash attention, positions restart
            lens = np.array([len(e) for e in enc], np.int64)
            ids = np.fromiter((t for e in enc for t in e), dtype=np.int64, count=int(lens.sum()))
            pos = np.concatenate([np.arange(n) for n in lens])
            cu = np.r_[0, np.cumsum(lens)].astype(np.int32)
            return {"keys": keys, "ids": ids, "pos": pos, "cu": cu, "maxlen": int(lens.max())}
        L = max(1, max(len(e) for e in enc))
        pad = mt.pad_token_id if mt.pad_token_id is not None else (eos or 0)
        ids = np.full((len(enc), L), pad, dtype=np.int64)
        mask = np.zeros((len(enc), L), dtype=np.int64)
        for i, e in enumerate(enc):
            if a.pooling == "last":  # left padding: the last position is every row's last token
                ids[i, L - len(e):] = e
                mask[i, L - len(e):] = 1
            else:
                ids[i, :len(e)] = e
                mask[i, :len(e)] = 1
        return {"keys": keys, "ids": ids, "mask": mask}


def embed_batch(model, b, a, dev):
    """(vectors [n, dim] float32 normalized, model tokens) of one batch from DocTexts.batch."""
    import torch

    if "cu" in b:
        ids = torch.as_tensor(b["ids"], device=dev)[None]
        pos = torch.as_tensor(b["pos"], device=dev)[None]
        cu = torch.as_tensor(b["cu"], device=dev, dtype=torch.int32)
        m = int(b["maxlen"])
        h = model(input_ids=ids, position_ids=pos, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                  max_length_q=m, max_length_k=m).last_hidden_state[0]
        if a.pooling == "last":
            v = h[cu[1:].long() - 1]
        else:
            lens = (cu[1:] - cu[:-1]).long()
            seg = torch.repeat_interleave(torch.arange(len(lens), device=dev), lens)
            v = torch.zeros(len(lens), h.shape[-1], device=dev, dtype=torch.float32).index_add_(0, seg, h.float())
            v = v / lens.clamp(min=1)[:, None]
        n_tok = int(b["cu"][-1])
    else:
        ids = torch.as_tensor(b["ids"], device=dev)
        mask = torch.as_tensor(b["mask"], device=dev)
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        if a.pooling == "last":
            v = h[:, -1]
        else:
            mk = mask.unsqueeze(-1).to(h.dtype)
            v = (h * mk).sum(1) / mk.sum(1).clamp(min=1)
        n_tok = int(b["mask"].sum())
    return torch.nn.functional.normalize(v.float()[:, : a.dim], dim=-1), n_tok


class _Batches:
    """torch Dataset over precomputed batches (sorted by length, so padding stays small)."""

    def __init__(self, texts: DocTexts, batches):
        self.texts, self.batches = texts, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.texts.batch(self.batches[i])


def resolve(prefixes, keys: np.ndarray, max_src: int):
    """Per key: dataset, byte offset of the document in its .bin, tokens to read (sorted vectorized
    lookups of the .idx lengths / pointers: near-sequential page access)."""
    ds = (keys >> KEY_SHIFT).astype(np.int64)
    doc = keys & ((1 << KEY_SHIFT) - 1)
    off = np.empty(len(keys), np.int64)
    ntok = np.empty(len(keys), np.int64)
    dtypes = {}
    for d in np.unique(ds):
        m = np.nonzero(ds == d)[0]
        it = IndexedTokens(prefixes[int(d)])
        dtypes[int(d)] = it.dtype
        o = np.argsort(doc[m], kind="stable")
        idx = doc[m][o]
        ntok[m[o]] = np.minimum(np.asarray(it.lengths[idx]), max_src)
        off[m[o]] = np.asarray(it.pointers[idx])
    return ds, off, ntok, dtypes


def make_batches(keys, ds, off, ntok, a) -> list:
    """Batches of about --batch-tokens model tokens, by estimated length (source tokens)."""
    est = np.minimum(ntok * a.len_ratio + 8, a.max_len).astype(np.int64)
    order = np.argsort(est, kind="stable")
    batches, cur, cur_max, sum_est = [], [], 0, 0
    for i in order:
        m = max(cur_max, int(est[i]))
        # padded: the batch costs its longest document x its size; packed: the sum of the lengths
        cost = (sum_est + int(est[i])) if a.pack else m * (len(cur) + 1)
        if cur and cost > a.batch_tokens:
            c = np.asarray(cur)
            batches.append((keys[c], ds[c], off[c], ntok[c]))
            cur, m, sum_est = [], int(est[i]), 0
        cur.append(i)
        cur_max = m
        sum_est += int(est[i])
    if cur:
        c = np.asarray(cur)
        batches.append((keys[c], ds[c], off[c], ntok[c]))
    rng = np.random.default_rng(0)
    rng.shuffle(batches)  # mix lengths over time (steady throughput, honest progress estimates)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="embedding run name (<out>/emb/<name>)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pooling", choices=["last", "mean"], required=True,
                    help="last: Qwen3-Embedding (EOS state, left padding); mean: e5 / BERT-style")
    ap.add_argument("--prefix", default="", help='text prefix, e.g. "passage: " for e5')
    ap.add_argument("--tokenizer", required=True, help="the training tokenizer (to decode the .bin)")
    ap.add_argument("--dim", type=int, default=256, help="Matryoshka truncation (then re-normalized)")
    ap.add_argument("--max-src-tokens", type=int, default=640)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--len-ratio", type=float, default=1.0, help="model tokens per source token (batching estimate)")
    ap.add_argument("--batch-tokens", type=int, default=None,
                    help="model tokens per batch (default: 131072 packed, 65536 padded)")
    ap.add_argument("--pack", choices=["auto", "on", "off"], default="auto",
                    help="padding-free batches (varlen flash attention); auto = on with flash_attention_2")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (dynamic shapes)")
    ap.add_argument("--check", type=int, default=0,
                    help="first embed this many docs both packed and padded and print their cosine")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--read-threads", type=int, default=32, help="concurrent .bin reads per loader worker")
    ap.add_argument("--attn", default="flash_attention_2",
                    help="attn_implementation (flash_attention_2: unpadded, no O(L^2) mask; falls back to sdpa)")
    ap.add_argument("--chunks", default=None, help="comma list of stream chunk names (default: all)")
    ap.add_argument("--bench", type=int, default=0, help="embed only this many docs of the first chunk, print speed")
    a = ap.parse_args()

    import torch
    from transformers import AutoModel

    # RANK / WORLD_SIZE / LOCAL_RANK (run_pipeline.sh: one process per GPU), else one task per GPU
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    dev = torch.device("cuda", local % max(torch.cuda.device_count(), 1)) if torch.cuda.is_available() else torch.device("cpu")
    blend = json.load(open(os.path.join(a.out, "blend.json")))
    prefixes = blend["prefixes"]
    root = os.path.join(a.out, "emb", a.name)
    os.makedirs(root, exist_ok=True)
    if rank == 0:
        json.dump(vars(a), open(os.path.join(root, "config.json"), "w"), indent=1)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    try:
        model = AutoModel.from_pretrained(a.model, dtype=dtype, attn_implementation=a.attn)
    except (ImportError, ValueError) as e:  # e.g. flash_attention_2 unavailable: masked SDPA
        print(f"rank {rank}: attn_implementation={a.attn} unavailable ({e}); using sdpa", flush=True)
        model = AutoModel.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa")
    model = model.to(dev).eval()
    flash = model.config._attn_implementation == "flash_attention_2"
    a.pack = (a.pack == "on") or (a.pack == "auto" and flash)
    assert not a.pack or flash, "--pack on needs flash_attention_2 (varlen kernels)"
    if a.batch_tokens is None:
        a.batch_tokens = 131072 if a.pack else 65536
    if a.compile:
        model = torch.compile(model, dynamic=True)
    if rank == 0:
        print(f"model {a.model}: attention {model.config._attn_implementation}, packed {a.pack}, "
              f"batch {a.batch_tokens} tokens, compile {a.compile}", flush=True)

    chunks = sorted(p for p in glob.glob(os.path.join(a.out, "stream", "ck*_*")) if not p.endswith(".tmp"))
    if a.chunks:
        want = set(a.chunks.split(","))
        chunks = [c for c in chunks if os.path.basename(c) in want]
    for cpath in chunks:
        name = os.path.basename(cpath)
        dst = os.path.join(root, name, f"part_{rank:04d}_of_{world:04d}.npz")
        if os.path.exists(dst) and not a.bench:
            continue
        keys = chunk_keys(cpath)[rank::world]
        if a.bench:
            keys = keys[: a.bench]
        t0 = time.time()
        t_res = time.time()
        ds_, off_, ntok_, dtypes = resolve(prefixes, keys, a.max_src_tokens)
        t_res = time.time() - t_res
        texts = DocTexts(prefixes, dtypes, a)
        batches = make_batches(keys, ds_, off_, ntok_, a)
        if a.check and a.pack:
            n = min(a.check, len(keys))
            item = (keys[:n], ds_[:n], off_[:n], ntok_[:n])
            with torch.inference_mode():
                vp, _ = embed_batch(model, texts.batch(item), a, dev)
                a.pack = False
                pad_texts = DocTexts(prefixes, dtypes, a)
                vq = torch.cat([embed_batch(model, pad_texts.batch(tuple(x[j:j + 64] for x in item)), a, dev)[0]
                                for j in range(0, n, 64)])
                a.pack = True
            cos = (vp * vq).sum(-1)
            print(f"CHECK rank {rank}: packed vs padded over {n} docs: cosine min {float(cos.min()):.5f}, "
                  f"mean {float(cos.mean()):.5f}", flush=True)
        loader = torch.utils.data.DataLoader(_Batches(texts, batches), batch_size=None, shuffle=False,
                                             num_workers=a.workers, prefetch_factor=4 if a.workers else None,
                                             persistent_workers=False)
        out_k, out_v, ntok, done = [], [], 0, 0
        t_wait = t_gpu = 0.0
        tw = time.time()
        with torch.inference_mode():
            for i, b in enumerate(loader):
                t_wait += time.time() - tw
                tg = time.time()
                v, n_tok = embed_batch(model, b, a, dev)
                out_k.append(np.asarray(b["keys"]))
                out_v.append(v.to(torch.float16).cpu().numpy())
                ntok += n_tok
                done += len(b["keys"])
                t_gpu += time.time() - tg
                tw = time.time()
                if rank == 0 and (i % 200 == 0 or i == len(batches) - 1):
                    dt = time.time() - t0
                    print(f"[{name}] rank 0: {done}/{len(keys)} docs, {done / dt:.0f} docs/s, "
                          f"{ntok / dt / 1e3:.0f}k tok/s", flush=True)
        k = np.concatenate(out_k) if out_k else np.zeros(0, np.int64)
        v = np.concatenate(out_v) if out_v else np.zeros((0, a.dim), np.float16)
        dt = time.time() - t0
        if a.bench:
            print(f"BENCH rank {rank}: {len(k)} docs in {dt:.1f} s = {len(k) / dt:.0f} docs/s, "
                  f"{ntok / dt / 1e3:.0f}k model tok/s, mean {ntok / max(len(k), 1):.0f} tok/doc; "
                  f"resolve {t_res:.1f} s, loader wait {t_wait:.1f} s, GPU {t_gpu:.1f} s", flush=True)
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst[:-4] + ".tmp.npz"
        np.savez(tmp, keys=k, vec=v)
        os.replace(tmp, dst)
        print(f"[{name}] rank {rank}: {len(k)} docs, {dt:.0f} s ({len(k) / max(dt, 1e-9):.0f} docs/s)", flush=True)


if __name__ == "__main__":
    main()
