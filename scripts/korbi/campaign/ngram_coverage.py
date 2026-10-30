"""How much of the token stream the frequent n-grams cover, per order, for sizing Engram tables
(user 2026-10-30: scaling the Engram share to ~50 %: are 2- / 3-grams saturated there, and are
4- / 5-grams frequent enough to be worth a table?).

On a nested sample of the production data mix (0.5 / 1 / 2 / 4B tokens by default; the smaller
samples are prefixes of the larger ones, so growth with tokens can be fitted), after Engram's
token map, for every n-gram order and sample size:
  - positions (tokens with n - 1 predecessors in their document) and distinct n-grams;
  - coverage of the top K n-grams (the share of positions whose n-gram is among the K most
    frequent), for K = table rows per head (Engram stores each n-gram once per head);
  - the count of the K-th n-gram, and the share of positions whose n-gram occurs >= 2 / 10 / 100
    times.

n-grams never cross documents. Keys are 64-bit hashes of the canonical ids (expected key
collisions over 4B positions < 1). The key space is split into buckets over the GPUs: each
process holds the whole sample in host RAM and counts its buckets with torch.unique on its GPU;
only count histograms are merged, which is enough for exact top-K coverage.

    torchrun --standalone --nproc-per-node 4 scripts/korbi/campaign/ngram_coverage.py \\
        --datamix $HYBRID_ENV/data/flag_decon_v2_agenttok_datamix_mn5.txt \\
        --token-map $HYBRID_ENV/data/engram_token_map_256k.npy --out <dir>

Needs Megatron on PYTHONPATH (megatron.core.datasets.indexed_dataset).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from megatron.core.datasets.indexed_dataset import IndexedDataset

MIX = 0x9E3779B97F4A7C15 - (1 << 64)  # golden-ratio multiplier as a signed int64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} rank {os.environ.get('RANK', 0)}] {msg}", flush=True)


def read_datamix(path):
    entries = []
    for line in Path(path).read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            entries.append((float(parts[0]), parts[1]))
    total = sum(w for w, _ in entries)
    return [(w / total, p) for w, p in entries]


def load_sample(datamix, total_tokens, seed):
    """Concatenated token slices (one contiguous slice per shard, starting at a document
    boundary, length = weight x total) plus every token's position within its document and its
    shard-relative offset (for the nested prefixes). Returns tokens [T] int32, docpos [T] int32,
    frac [T] float16 (offset / slice length: a token belongs to the prefix sample p if frac < p),
    slices [(start, end)] in the concatenation."""
    rng = np.random.default_rng(seed)
    cap = sum(int(round(w * total_tokens)) for w, _ in datamix)  # preallocated: no concat copy
    toks = np.empty(cap, dtype=np.int32)
    poss = np.empty(cap, dtype=np.int32)
    fracs = np.empty(cap, dtype=np.float16)
    slices, at = [], 0
    for w, prefix in datamix:
        want = int(round(w * total_tokens))
        if want == 0:
            continue
        index = IndexedDataset(prefix, mmap=True).index  # text_document: one sequence per document
        lengths = index.sequence_lengths.astype(np.int64)
        pointers = index.sequence_pointers.astype(np.int64)
        dtype = np.dtype(index.dtype)
        cum = np.concatenate([[0], np.cumsum(lengths)])
        if cum[-1] <= want:
            d0 = 0
        else:
            last_start = int(np.searchsorted(cum, cum[-1] - want, side="right")) - 1
            d0 = int(rng.integers(0, max(last_start, 0) + 1))
        d1 = int(np.searchsorted(cum, cum[d0] + want, side="left"))
        d1 = min(max(d1, d0 + 1), len(lengths))
        start = pointers[d0] // dtype.itemsize
        end = pointers[d1 - 1] // dtype.itemsize + lengths[d1 - 1]
        bin_ = np.memmap(prefix + ".bin", dtype=dtype, mode="r")
        t = np.asarray(bin_[start:end], dtype=np.int32)[:want]
        n = t.shape[0]
        doc_lens = lengths[d0:d1]
        doc_starts = np.repeat(np.concatenate([[0], np.cumsum(doc_lens)[:-1]]), doc_lens)[:n]
        pos = (np.arange(n, dtype=np.int64) - doc_starts).astype(np.int32)
        toks[at : at + n] = t
        poss[at : at + n] = pos
        fracs[at : at + n] = np.arange(n, dtype=np.float32) / max(n, 1)
        slices.append((at, at + n))
        at += n
    return toks[:at], poss[:at], fracs[:at], slices


def ngram_keys(ids, docpos, n):
    """[T] canonical ids (int64, on GPU) -> [T] int64 keys of the n-gram ending at each token and
    a validity mask (the n-gram lies inside one document)."""
    key = ids.clone()
    for i in range(1, n):
        prev = torch.nn.functional.pad(ids, (i, 0))[: ids.shape[0]]
        key = key * MIX + (prev + 1)  # wraps mod 2^64
    key = key ^ (key >> 29)
    return key, docpos >= (n - 1)


def count_histogram(keys):
    """Distinct-count histogram: (count values, number of n-grams with that count), numpy."""
    _, counts = torch.unique(keys, return_counts=True)
    values, mult = torch.unique(counts, return_counts=True)
    return values.cpu().numpy(), mult.cpu().numpy()


def merge_hists(hists):
    acc = {}
    for values, mult in hists:
        for v, m in zip(values.tolist(), mult.tolist()):
            acc[v] = acc.get(v, 0) + m
    values = np.array(sorted(acc, reverse=True), dtype=np.int64)  # most frequent first
    return values, np.array([acc[v] for v in values], dtype=np.int64)


def summarize(values, mult, ks, thresholds):
    """Coverage of the top K from a (descending) count histogram."""
    positions = int((values * mult).sum())
    distinct = int(mult.sum())
    cum_n = np.cumsum(mult)
    cum_pos = np.cumsum(values * mult)
    out = {"positions": positions, "distinct": distinct, "topk": {}, "count_at_least": {}}
    for k in ks:
        if k >= distinct:
            out["topk"][str(k)] = {"coverage": 1.0, "kth_count": int(values[-1])}
            continue
        j = int(np.searchsorted(cum_n, k, side="left"))  # bin holding the k-th n-gram
        before_n = int(cum_n[j - 1]) if j else 0
        before_pos = int(cum_pos[j - 1]) if j else 0
        covered = before_pos + (k - before_n) * int(values[j])
        out["topk"][str(k)] = {"coverage": covered / positions, "kth_count": int(values[j])}
    for t in thresholds:
        sel = values >= t
        out["count_at_least"][str(t)] = float((values[sel] * mult[sel]).sum() / positions)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datamix", required=True)
    ap.add_argument("--token-map", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--samples", type=float, nargs="+", default=[0.5e9, 1e9, 2e9, 4e9])
    ap.add_argument("--rows", type=int, nargs="+",
                    default=[5_000_000, 10_000_000, 19_480_384, 40_000_000, 80_000_000])
    ap.add_argument("--thresholds", type=int, nargs="+", default=[2, 10, 100])
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200_000_000, help="tokens per GPU pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = int(max(args.samples))
    fracs_levels = sorted(s / total for s in args.samples)

    t0 = time.time()
    tokens, docpos, frac, slices = load_sample(read_datamix(args.datamix), total, args.seed)
    log(f"sample: {tokens.shape[0] / 1e9:.3f}B tokens from {len(slices)} shards in {time.time() - t0:.0f} s")
    tmap = None
    if args.token_map:
        tmap = torch.from_numpy(np.load(args.token_map).astype(np.int64)).cuda()
        top = int(tokens.max())
        if top >= tmap.shape[0]:
            raise SystemExit(f"token id {top} >= token map size {tmap.shape[0]}: wrong map for this data")
        if rank == 0:
            log(f"token map: {tmap.shape[0]} ids -> {int(tmap.max()) + 1} canonical; max token id {top}")

    # chunks of whole shard slices (an n-gram never crosses a slice: each starts at a document)
    chunks, cur = [], []
    for s, e in slices:
        cur.append((s, e))
        if sum(b - a for a, b in cur) >= args.chunk:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    mine = [b for b in range(args.buckets) if b % world == rank]
    results = {}
    for n in args.orders:
        for b in mine:
            keys_all, frac_all = [], []
            for chunk in chunks:
                s, e = chunk[0][0], chunk[-1][1]
                ids = torch.from_numpy(tokens[s:e]).cuda(non_blocking=True).to(torch.int64)
                if tmap is not None:
                    ids = tmap[ids]
                pos = torch.from_numpy(docpos[s:e]).cuda(non_blocking=True)
                key, valid = ngram_keys(ids, pos, n)
                valid &= torch.remainder(key, args.buckets) == b
                keys_all.append(key[valid])
                frac_all.append(torch.from_numpy(frac[s:e]).cuda(non_blocking=True)[valid])
                del ids, pos, key, valid
            keys = torch.cat(keys_all)
            fr = torch.cat(frac_all)
            del keys_all, frac_all
            for p in fracs_levels:
                values, mult = count_histogram(keys[fr < p] if p < 1.0 else keys)
                results.setdefault(str(n), {}).setdefault(f"{p:.6f}", []).append(
                    (values.tolist(), mult.tolist()))
            log(f"order {n} bucket {b}: {keys.numel() / 1e6:.0f}M positions")
            del keys, fr
            torch.cuda.empty_cache()
    (out / f"hist_rank{rank}.json").write_text(json.dumps(results))
    (out / f"done_rank{rank}").write_text("ok")
    if rank != 0:
        return

    while not all((out / f"done_rank{r}").exists() for r in range(world)):
        time.sleep(5)
    merged = {}
    for r in range(world):
        part = json.loads((out / f"hist_rank{r}.json").read_text())
        for n, levels in part.items():
            for p, hists in levels.items():
                merged.setdefault(n, {}).setdefault(p, []).extend(hists)
    summary = {"datamix": args.datamix, "token_map": args.token_map, "sample_tokens": total,
               "rows": args.rows, "orders": {}}
    for n in sorted(merged, key=int):
        for p in sorted(merged[n], key=float):
            values, mult = merge_hists([(np.array(v), np.array(m)) for v, m in merged[n][p]])
            s = summarize(values, mult, args.rows, args.thresholds)
            s["sample_tokens"] = float(p) * total
            summary["orders"].setdefault(n, []).append(s)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\n{'n':>2} {'sample':>7} {'positions':>11} {'distinct':>11} "
          + " ".join(f"cov@{k / 1e6:g}M" for k in args.rows)
          + " " + " ".join(f">={t}" for t in args.thresholds))
    for n, rows in summary["orders"].items():
        for s in rows:
            print(f"{n:>2} {s['sample_tokens'] / 1e9:6.2f}B {s['positions']:>11,} {s['distinct']:>11,} "
                  + " ".join(f"{s['topk'][str(k)]['coverage']:8.3f}" for k in args.rows)
                  + " " + " ".join(f"{s['count_at_least'][str(t)]:5.3f}" for t in args.thresholds))
    log(f"done in {time.time() - t0:.0f} s; {out / 'summary.json'}")


if __name__ == "__main__":
    main()
