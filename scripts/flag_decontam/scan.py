"""Scan Megatron .bin shards for reference n-grams on GPUs (docs/flag_post_decontamination.md).

    python scripts/flag_decontam/scan.py --mode {count,scan} --work $WORK --shards shards.txt \
        [--part K --nparts N] [--sample-tokens T]

Every node runs this once (part = SLURM_NODEID, nparts = SLURM_NNODES by default); it splits its
share of the work into units of whole sequences (~--chunk-tokens tokens each) and runs one worker
process per visible GPU. Per unit a worker reads the token range from the .bin (--readers threads
keep the next units loading), maps it to the canonical stream, hashes the 8- and 13-gram windows
that stay inside one sequence and looks them up in the sorted reference table.

  count: per reference n-gram, how often it occurs (sampling the first --sample-tokens of every
         shard). build_table.py --prune drops the n-grams that are common in ordinary data.
  scan:  per unit, every (sequence, reference segment) pair with >= --min-hits distinct matched
         n-grams -> $WORK/hits/<shard key>/<first seq>.npz (seq, seg, matched). Thresholds are
         applied later (finalize.py), so they can change without rescanning. Units already done
         are skipped: a rerun resumes.
Token ids outside the tokenizer's vocabulary are dropped from the canonical stream and reported
($WORK/anomalies/*.json and a WARNING line).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NGRAM_SIZES, Index, ngram_hashes  # noqa: E402


def shard_key(prefix: str, root: str) -> str:
    return os.path.relpath(prefix, root).replace("/", "__")


def plan_units(prefixes, chunk_tokens, sample_tokens):
    """[(prefix, seq_start, seq_end, tokens)] in whole sequences of about chunk_tokens."""
    units = []
    for p in prefixes:
        idx = Index(p + ".idx")
        csum = np.cumsum(idx.lengths, dtype=np.int64)
        total = int(csum[-1]) if len(csum) else 0
        limit = min(total, sample_tokens) if sample_tokens else total
        start = 0
        while start < idx.seq_count:
            before = int(csum[start - 1]) if start else 0
            if before >= limit:
                break
            end = int(np.searchsorted(csum, min(before + chunk_tokens, limit), "right"))
            end = min(max(end, start + 1), idx.seq_count)
            units.append((p, start, end, int(csum[end - 1]) - before))
            start = end
    return units


def read_unit(prefix, s0, s1, idx_cache):
    idx = idx_cache.get(prefix)
    if idx is None:
        idx = idx_cache[prefix] = Index(prefix + ".idx")
    lengths = np.asarray(idx.lengths[s0:s1], dtype=np.int64)
    ptr = idx.pointers[s0:s1]
    item = np.dtype(idx.dtype).itemsize
    contiguous = np.array_equal(ptr[1:] - ptr[:-1], lengths[:-1] * item)
    if contiguous:
        toks = np.fromfile(prefix + ".bin", dtype=idx.dtype, count=int(lengths.sum()), offset=int(ptr[0]))
    else:  # never seen in the flag shards; correct but slow
        with open(prefix + ".bin", "rb") as f:
            parts = []
            for p, n in zip(ptr, lengths):
                f.seek(int(p))
                parts.append(np.frombuffer(f.read(int(n) * item), dtype=idx.dtype))
        toks = np.concatenate(parts)
    return toks, lengths


def process(toks, lengths, s0, lut, table, csr_off, csr_seg, dev, mode, min_hits, counts, bad_info):
    t = torch.from_numpy(toks.astype(np.int64, copy=False)).to(dev, non_blocking=True)
    bad = (t < 0) | (t >= lut.shape[0])
    n_bad = int(bad.sum())
    if n_bad:   # ids outside the tokenizer's vocabulary: no reference contains them; drop and report
        bad_info.append((n_bad, int(t[bad].min()), int(t[bad].max())))
        t = torch.where(bad, 0, t)
    c = lut[t]
    if n_bad:
        c[bad] = -1
    del t, bad
    seq = torch.repeat_interleave(torch.arange(len(lengths), device=dev),
                                  torch.from_numpy(lengths).to(dev))
    keep = c >= 0
    c, seq = c[keep], seq[keep]
    del keep
    hit_seq, hit_gram = [], []
    for n in NGRAM_SIZES:
        h = ngram_hashes(c, n)
        if h.numel() == 0:
            continue
        L = h.shape[0]
        valid = seq[:L] == seq[n - 1:n - 1 + L]
        j = torch.searchsorted(table, h)
        j.clamp_(max=table.shape[0] - 1)
        found = (table[j] == h) & valid
        del h, valid
        hit_seq.append(seq[:L][found])
        hit_gram.append(j[found])
    if not hit_seq:
        return None
    hs, hg = torch.cat(hit_seq), torch.cat(hit_gram)
    if mode == "count":
        counts += torch.bincount(hg, minlength=counts.shape[0])
        return None
    if hs.numel() == 0:
        return np.zeros((0, 3), dtype=np.int64)
    pairs = torch.unique(torch.stack([hs, hg], 1), dim=0).cpu().numpy()   # distinct (seq, gram)
    g = pairs[:, 1]
    k = csr_off[g + 1] - csr_off[g]                                       # segments per gram
    rep_seq = np.repeat(pairs[:, 0], k)
    starts = np.repeat(csr_off[g] - np.concatenate([[0], np.cumsum(k)[:-1]]), k)
    segs = csr_seg[starts + np.arange(k.sum())]
    sp, matched = np.unique(np.stack([rep_seq, segs], 1), axis=0, return_counts=True)
    sel = matched >= min_hits
    out = np.empty((int(sel.sum()), 3), dtype=np.int64)
    out[:, 0] = sp[sel, 0] + s0
    out[:, 1] = sp[sel, 1]
    out[:, 2] = matched[sel]
    return out


def worker(rank, ngpu, units, args):
    torch.cuda.set_device(rank % ngpu) if torch.cuda.is_available() else None
    dev = torch.device(f"cuda:{rank % ngpu}") if torch.cuda.is_available() else torch.device("cpu")
    tab = np.load(os.path.join(args.work, "table", "table.npz"))
    table = torch.from_numpy(tab["hash"]).to(dev)
    csr_off, csr_seg = tab["csr_off"], tab["csr_seg"]
    lut = torch.from_numpy(np.load(os.path.join(args.work, "table", "lut.npy"))).to(dev)
    counts = torch.zeros(table.shape[0], dtype=torch.int64, device=dev) if args.mode == "count" else None
    mine = units[rank::ngpu]
    todo = []
    for u in mine:
        out = os.path.join(args.work, "hits", shard_key(u[0], args.root), f"{u[1]:012d}.npz")
        if args.mode == "scan" and os.path.exists(out):
            continue
        todo.append((u, out))
    q: queue.Queue = queue.Queue(maxsize=2 * args.readers)

    def reader(k):
        cache = {}
        try:
            for u, out in todo[k::args.readers]:
                q.put((u, out, *read_unit(u[0], u[1], u[2], cache)))
        except BaseException as e:   # a read error must fail the job, not leave the worker waiting
            q.put(e)
        q.put(None)

    for k in range(args.readers):    # np.fromfile releases the GIL: parallel GPFS streams
        threading.Thread(target=reader, args=(k,), daemon=True).start()
    t0, done_tok, finished = time.time(), 0, 0
    anomalies = []
    while finished < args.readers:
        item = q.get()
        if item is None:
            finished += 1
            continue
        if isinstance(item, BaseException):
            raise item
        (prefix, s0, s1, ntok), out, toks, lengths = item
        bad_info = []
        res = process(toks, lengths, s0, lut, table, csr_off, csr_seg, dev, args.mode, args.min_hits, counts,
                      bad_info)
        if bad_info:
            n, lo, hi = bad_info[0]
            anomalies.append({"shard": os.path.relpath(prefix, args.root), "seqs": [s0, s1], "ids_out_of_range": n,
                              "min": lo, "max": hi})
            print(f"[node{args.part} gpu{rank}] WARNING {n} token ids outside the vocabulary (min {lo}, max {hi}) "
                  f"in {shard_key(prefix, args.root)} seqs {s0}-{s1}", flush=True)
        if args.mode == "scan":
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out + ".tmp.npz", hits=res if res is not None else np.zeros((0, 3), np.int64),
                     seq_range=np.array([s0, s1]))
            os.replace(out + ".tmp.npz", out)
        done_tok += ntok
        dt = time.time() - t0
        print(f"[node{args.part} gpu{rank}] {shard_key(prefix, args.root)} seqs {s0}-{s1} "
              f"hits={0 if res is None else len(res)} | {done_tok / 1e9:.1f}B tok, "
              f"{done_tok / dt / 1e6:.0f} Mtok/s", flush=True)
    if anomalies:
        path = os.path.join(args.work, "anomalies", f"{args.mode}_part{args.part:03d}_gpu{rank}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(anomalies, f, indent=1)
    if args.mode == "count":
        path = os.path.join(args.work, "counts", f"part{args.part:03d}_gpu{rank}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, counts.cpu().numpy())
    print(f"[node{args.part} gpu{rank}] done: {done_tok / 1e9:.1f}B tokens in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["count", "scan"], required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--shards", required=True, help="file: one Megatron prefix per line (weight optional)")
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_NODEID", 0)))
    ap.add_argument("--nparts", type=int, default=int(os.environ.get("SLURM_NNODES", 1)))
    ap.add_argument("--chunk-tokens", type=int, default=1 << 28)
    ap.add_argument("--sample-tokens", type=int, default=0, help="count: tokens read per shard (0 = all)")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--gpus", type=int, default=0, help="worker processes (default: visible GPUs)")
    ap.add_argument("--readers", type=int, default=2, help="reader threads per worker")
    args = ap.parse_args()

    prefixes = [line.split()[-1] for line in open(args.shards) if line.strip()]
    units = plan_units(prefixes, args.chunk_tokens, args.sample_tokens)
    # deal units out by size so every node gets about the same number of tokens
    units.sort(key=lambda u: -u[3])
    loads = [0] * args.nparts
    mine = []
    for u in units:
        k = min(range(args.nparts), key=loads.__getitem__)
        loads[k] += u[3]
        if k == args.part:
            mine.append(u)
    ngpu = args.gpus or max(torch.cuda.device_count(), 1)
    print(f"part {args.part}/{args.nparts}: {len(mine)} units, {sum(u[3] for u in mine) / 1e9:.1f}B tokens "
          f"of {sum(u[3] for u in units) / 1e9:.1f}B, {ngpu} workers", flush=True)
    mp.spawn(worker, args=(ngpu, mine, args), nprocs=ngpu, join=True)


if __name__ == "__main__":
    main()
