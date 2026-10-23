"""Recompute the candidate pairs of a scan against corrected segment n-gram sets (no rescan).

    python scripts/flag_decontam/recheck.py --work $WORK [--frac 0.7] [--procs 64]

Used when the table built for the scan contained n-grams that should not count (v2: the
`ensure_ascii` JSON variant of non-ASCII text, whose `\\uXXXX` runs match unrelated escaped text).
Removing n-grams can only shorten runs, so only the pairs that pass the loosest rule finalize
will be asked for (longest run >= max(--min-run, min(ceil(--frac * n_tokens), --abs-run))) can
change a decision. For those whose segment build_table.variants() now hashes differently, the
document is re-read, its windows hashed and matched against the segment's corrected n-grams
(also intersected with the scanned table, i.e. after pruning), and the run recomputed.

Writes $WORK/hits_fixed/<shard>/<unit>.npz for every unit (unchanged rows copied; corrected rows
updated, dropped below --min-run) and hits_fixed/recheck_report.json. finalize.py --hits-dir
hits_fixed then decides on the corrected runs.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_table import TOKENIZER, variants  # noqa: E402
from common import Index, ngram_hashes, runs  # noqa: E402

_G = {}


def _init():
    _G["idx"] = {}                 # the lookup table and n-gram sets come with the fork (module global _G)
    torch.set_num_threads(1)


def longest_run(h, n, allowed):
    m = np.flatnonzero(np.isin(h, allowed))
    if not len(m):
        return 0, 0, 0
    m = torch.from_numpy(m)
    z = torch.zeros(len(m), dtype=torch.int64)
    _, _, a, b, w = runs(m, z, z, torch.tensor([n]))
    return int((b - a).max()), int(w.sum()), int(len(a))


def _recheck(job):
    """job: (shard key, seq, [segments]) -> [(seq, seg, run, windows, runs)]"""
    key, seq, seglist = job
    prefix = _G["prefixes"][key]
    idx = _G["idx"].get(prefix) or _G["idx"].setdefault(prefix, Index(prefix + ".idx"))
    toks = np.fromfile(prefix + ".bin", dtype=idx.dtype, count=int(idx.lengths[seq]), offset=int(idx.pointers[seq]))
    lut = _G["lut"]
    t = toks.astype(np.int64)
    bad = (t < 0) | (t >= len(lut))
    c = lut[np.where(bad, 0, t)]
    c[bad] = -1
    c = torch.from_numpy(c[c >= 0])
    hashes = {}                                   # one hash pass per n, shared by the segments
    out = []
    for seg in seglist:
        n = int(_G["seg_n"][seg])
        if n not in hashes:
            hashes[n] = ngram_hashes(c, n).numpy()
        out.append((seq, seg, *longest_run(hashes[n], n, _G["seg_hashes"][seg])))
    return key, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--abs-run", type=int, default=200)
    ap.add_argument("--min-run", type=int, default=30)
    ap.add_argument("--procs", type=int, default=min(os.cpu_count() or 1, 64))
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()
    from scan import shard_key
    import sentencepiece as spm

    tdir = os.path.join(args.work, "table")
    segs = np.load(os.path.join(tdir, "segments.npz"))
    meta = json.load(open(os.path.join(tdir, "sources.json")))
    n_tok, seg_n = segs["n_tokens"].astype(np.int64), segs["n"].astype(np.int64)
    need = np.maximum(args.min_run, np.minimum(np.ceil(args.frac * n_tok), args.abs_run))
    need[seg_n == 0] = np.iinfo(np.int64).max
    prefixes = {shard_key(os.path.join(args.root, line.split()[1]), args.root): os.path.join(args.root, line.split()[1])
                for line in open(os.path.join(args.root, "datamix.txt")) if line.strip()}

    files = sorted(glob.glob(os.path.join(args.work, "hits", "*", "*.npz")))
    cand = []                                   # (file, row) of candidate pairs
    cand_segs = set()
    for f in files:
        h = np.load(f)["hits"]
        if len(h):
            rows = np.flatnonzero(h[:, 2] >= need[h[:, 1]])
            if len(rows):
                cand.append((f, rows))
                cand_segs.update(h[rows, 1].tolist())
    print(f"{sum(len(r) for _, r in cand)} candidate pairs, {len(cand_segs)} segments, in {len(files)} unit files",
          flush=True)

    # corrected n-gram sets of the candidate segments whose variants changed
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    lut = np.load(os.path.join(tdir, "lut.npy"))
    table = np.load(os.path.join(tdir, "table.npz"))["hash"]
    texts = {}
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in cand_segs:
                texts[i] = json.loads(line)["text"]
    old_variants = lambda t: list(dict.fromkeys([t, json.dumps(t, ensure_ascii=False)[1:-1], json.dumps(t)[1:-1]]))
    seg_hashes, changed = {}, set()
    for s, t in texts.items():
        if variants(t) == old_variants(t):
            continue
        changed.add(s)
        hs = []
        for ids in sp.encode(variants(t)):
            c = lut[np.asarray(ids, dtype=np.int64)]
            c = c[c >= 0]
            if len(c) >= seg_n[s]:
                hs.append(ngram_hashes(torch.from_numpy(c), int(seg_n[s])).numpy())
        h = np.unique(np.concatenate(hs)) if hs else np.zeros(0, np.int64)
        j = np.clip(np.searchsorted(table, h), 0, len(table) - 1)
        seg_hashes[s] = h[table[j] == h]                      # only what the scan could match (pruned table)
    print(f"{len(changed)} candidate segments with corrected n-gram sets", flush=True)

    jobs = defaultdict(list)                    # (shard key, seq) -> segments to recheck
    for f, rows in cand:
        key = os.path.basename(os.path.dirname(f))
        h = np.load(f)["hits"]
        for r in rows:
            if int(h[r, 1]) in changed:
                jobs[(key, int(h[r, 0]))].append(int(h[r, 1]))
    print(f"{sum(len(v) for v in jobs.values())} pairs in {len(jobs)} sequences to recheck", flush=True)
    fixed = {}
    _G.update(lut=lut, seg_n=seg_n, seg_hashes=seg_hashes, prefixes=prefixes)
    del table
    with Pool(args.procs, initializer=_init) as pool:
        for j, (key, res) in enumerate(pool.imap_unordered(_recheck, [(k, s, v) for (k, s), v in jobs.items()],
                                                         chunksize=16)):
            if j % 5000 == 0:
                print(f"  {j}/{len(jobs)} sequences rechecked", flush=True)
            for seq, seg, run, w, nr in res:
                fixed[(key, seq, seg)] = (run, w, nr)

    out_root = os.path.join(args.work, "hits_fixed")
    stats = {"candidates": int(sum(len(r) for _, r in cand)), "rechecked": len(fixed), "still_candidates": 0,
             "rows_dropped": 0}
    changed_arr = np.array(sorted(changed), dtype=np.int64)
    by_file = defaultdict(dict)
    for (key, seq, seg), v in fixed.items():
        by_file[key][(seq, seg)] = v
    for f in files:
        key = os.path.basename(os.path.dirname(f))
        z = np.load(f)
        h = z["hits"].copy()
        keep = np.ones(len(h), bool)
        fk = by_file.get(key)
        if fk and len(h):
            for i in np.flatnonzero(np.isin(h[:, 1], changed_arr)):
                v = fk.get((int(h[i, 0]), int(h[i, 1])))
                if v is not None:
                    h[i, 2:5] = v
                    keep[i] = h[i, 2] >= args.min_run
                    stats["still_candidates"] += int(h[i, 2] >= need[h[i, 1]])
        stats["rows_dropped"] += int((~keep).sum())
        dst = os.path.join(out_root, key, os.path.basename(f))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez(dst, hits=h[keep], seq_range=z["seq_range"])
    with open(os.path.join(out_root, "recheck_report.json"), "w") as fh:
        json.dump({**stats, "frac": args.frac, "abs_run": args.abs_run, "min_run": args.min_run,
                   "segments_corrected": len(changed)}, fh, indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
