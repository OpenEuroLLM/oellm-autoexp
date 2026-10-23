"""Sample recorded matches of the scan and show them: benchmark segment vs. the document around it.

    python scripts/flag_decontam/inspect_hits.py --work $WORK --shard mixture-vitae-1.0 \
        [--n 20] [--min-cov 0.5] [--max-cov 1.0] [--bench gsm8k] [--seed 0] [--context 60]

--shard selects hit directories by substring of the shard key (e.g. "mixture-vitae-1.0__megatron-lm__shard_0004",
"agenttrove"). Coverage = the pair's longest verbatim run / the segment's canonical tokens (finalize
drops a document at coverage >= --frac, or a run >= --abs-run tokens). Prints first a summary of the selected pairs (coverage
histogram, benchmark sources by pair count), then --n random pairs: the benchmark source, coverage,
the segment text, and the decoded document text around its longest run (in >>> <<<, --context
tokens around it). Small enough for a login node: run it in the training image.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Index, ngram_hashes, runs  # noqa: E402
from scan import shard_key  # noqa: E402

TOKENIZER = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k/tokenizer.model"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/gpfs/scratch/ehpc533/users/korbip/flag_decon_v2")
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--shard", required=True, help="substring of the shard key")
    ap.add_argument("--bench", default="", help="substring of the benchmark source (e.g. gsm8k, mmlu)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-cov", type=float, default=0.0)
    ap.add_argument("--max-cov", type=float, default=10.0)   # > 1 possible: escaped copies are longer
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", type=int, default=60, help="tokens shown before / after the match")
    ap.add_argument("--max-files", type=int, default=400, help="hit files read (random subset)")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--fixes", default="recheck_fixes.npz", help="recheck.py's corrections, applied if present")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    tdir = os.path.join(args.work, "table")
    segs = np.load(os.path.join(tdir, "segments.npz"))
    meta = json.load(open(os.path.join(tdir, "sources.json")))
    src_names, n_tok = meta["sources"], segs["n_tokens"].astype(np.int64)
    prefixes = {shard_key(os.path.join(args.root, line.split()[1]), args.root): os.path.join(args.root, line.split()[1])
                for line in open(os.path.join(args.root, "datamix.txt")) if line.strip()}

    files = []
    for key in prefixes:
        if args.shard in key:
            files += [(key, f) for f in glob.glob(os.path.join(args.work, "hits", key, "*.npz"))]
    if not files:
        sys.exit(f"no hit files for shards matching {args.shard!r}")
    if len(files) > args.max_files:
        files = [files[i] for i in rng.choice(len(files), args.max_files, replace=False)]
    keys, rows = [], []
    from finalize import apply_fixes, load_fixes
    fixes = load_fixes(args.work, args.fixes) if os.path.exists(os.path.join(args.work, args.fixes)) else {}
    for i, (key, f) in enumerate(files):
        h = apply_fixes(np.load(f)["hits"], fixes.get(key), 30)
        rows.append(h)
        keys.append(np.full(len(h), i, np.int32))
    hits, which = np.concatenate(rows), np.concatenate(keys)
    cov = hits[:, 2] / np.maximum(n_tok[hits[:, 1]], 1)
    sel = (cov >= args.min_cov) & (cov <= args.max_cov)
    if args.bench:   # any attribution (refs.attrib.jsonl.gz), else the one source extraction kept
        ap_ = meta["refs"].replace(".jsonl.gz", ".attrib.jsonl.gz")
        if os.path.exists(ap_):
            seg_ok = np.zeros(len(n_tok), bool)
            with gzip.open(ap_, "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    seg_ok[i] = any(args.bench.lower() in a[0].lower() for a in json.loads(line)["also_in"] if a[3])
            sel &= seg_ok[hits[:, 1]]
        else:
            bench_ok = np.array([args.bench.lower() in s.lower() for s in src_names])
            sel &= bench_ok[segs["src"][hits[:, 1]]]
    hits, which, cov = hits[sel], which[sel], cov[sel]
    print(f"# {len(hits)} pairs in {len(files)} hit files of shards matching {args.shard!r} "
          f"(coverage {args.min_cov}..{args.max_cov}{', bench ' + args.bench if args.bench else ''})")
    if not len(hits):
        return
    edges = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 10]
    hist = np.histogram(cov, edges)[0]
    print("# coverage histogram: " + "  ".join(f"[{a:g},{b:g}): {n}" for a, b, n in zip(edges, edges[1:], hist)))
    bench = Counter(src_names[s] for s in segs["src"][hits[:, 1]])
    hi = Counter(src_names[s] for s in segs["src"][hits[cov >= 0.9, 1]])
    print("# benchmark sources by pairs (all / coverage >= 0.9):")
    for k, v in bench.most_common(15):
        print(f"#   {k:50s} {v:9d} {hi.get(k, 0):9d}")

    pick = rng.choice(len(hits), size=min(args.n, len(hits)), replace=False)
    want = {int(hits[i, 1]) for i in pick}
    texts = {}
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in want:
                r = json.loads(line)
                texts[i] = (r["text"], r.get("split", ""), r.get("file", ""))
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    lut = np.load(os.path.join(tdir, "lut.npy"))
    tab = np.load(os.path.join(tdir, "table.npz"))
    table, csr_off, csr_seg = tab["hash"], tab["csr_off"], tab["csr_seg"]
    idx_cache = {}
    for k, i in enumerate(pick):
        seq, seg, run = (int(x) for x in hits[i, :3])
        key = files[which[i]][0]
        prefix = prefixes[key]
        idx = idx_cache.setdefault(prefix, Index(prefix + ".idx"))
        toks = np.fromfile(prefix + ".bin", dtype=idx.dtype, count=int(idx.lengths[seq]), offset=int(idx.pointers[seq]))
        canon = lut[np.clip(toks.astype(np.int64), 0, len(lut) - 1)]
        canon[(toks < 0) | (toks >= len(lut))] = -1
        pos = np.flatnonzero(canon >= 0)
        n = int(segs["n"][seg])
        seg_hashes = table[np.searchsorted(csr_off, np.flatnonzero(csr_seg == seg), "right") - 1]
        h = ngram_hashes(torch.from_numpy(canon[pos]), n).numpy()
        m = torch.from_numpy(np.flatnonzero(np.isin(h, seg_hashes)))
        text, split, rfile = texts[seg]
        print(f"\n=== [{k + 1}/{len(pick)}] {src_names[segs['src'][seg]]} ({split}, {rfile[:60]}) | "
              f"coverage {cov[i]:.2f} (longest run {run} of {n_tok[seg]} tokens, {int(hits[i, 4])} runs, {n}-grams) | "
              f"{key} seq {seq} ({len(toks)} tokens)")
        print("--- benchmark segment:")
        print(text[:700] + (" ..." if len(text) > 700 else ""))
        if len(m):
            z = torch.zeros(len(m), dtype=torch.int64)
            _, _, ra, rb, _ = runs(m, z, z, torch.tensor([n]))
            best = int(torch.argmax(rb - ra))
            a, b = int(pos[int(ra[best])]), int(pos[min(int(rb[best]) - 1, len(pos) - 1)]) + 1
            pre = sp.decode(toks[max(0, a - args.context):a].tolist())
            mid = sp.decode(toks[a:b].tolist())
            post = sp.decode(toks[b:b + args.context].tolist())
            mid = mid if len(mid) <= 1500 else mid[:700] + f" [... {len(mid) - 1400} chars ...] " + mid[-700:]
            print(f"--- document, longest run (tokens {a}-{b} of {len(toks)}):")
            print(f"{pre}>>>{mid}<<<{post}")
        else:
            print("--- (match not re-found in the sequence)")


if __name__ == "__main__":
    main()
