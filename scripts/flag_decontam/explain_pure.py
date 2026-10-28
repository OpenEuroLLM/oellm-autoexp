"""Explain the differences between our decontamination and another pipeline's (after compare_pure.py).

    python scripts/flag_decontam/explain_pure.py --work $WORK --orig <flag prefix> --cmp <compare_pure out dir> \
        [--control 5000] [--procs 64]

The other pipeline (OpenEuroLLM pretraining-decontamination, NeMo Curator TaskDecontamination) removes
a document when it contains ANY benchmark word n-gram (8-13 words) that occurs <= --match-threshold
(10) times in the dataset. Ours removes it when one benchmark item is >= 70 % covered by a single
verbatim run (or the run is >= 200 tokens). This applies THEIR rule to OUR references: for every
document of the groups of compare_pure.py (both / theirs only / ours only, and a random sample of
documents both kept as a control), the distinct table n-grams it contains (any n-gram, no run
requirement), the segments and benchmarks they belong to, and the largest number of n-grams of a
single segment. Documents of "theirs only" with no table n-gram at all were removed for a benchmark
we do not have (or a variant we prune as common).

Writes <cmp>/explain.json and prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NGRAM_SIZES, Index, ngram_hashes  # noqa: E402
from finalize import load_attrib  # noqa: E402

_G = {}


def _doc_hits(d):
    """Distinct table n-gram indices contained in document d (all n of the table)."""
    idx, prefix, lut, table = _G["idx"], _G["prefix"], _G["lut"], _G["table"]
    a, b = int(idx.doc_idx[d]), int(idx.doc_idx[d + 1])
    t = np.fromfile(prefix + ".bin", np.int32, count=int(idx.lengths[a:b].sum()), offset=int(idx.pointers[a]))
    tt = t.astype(np.int64)
    c = lut[np.clip(tt, 0, len(lut) - 1)]
    c = c[(c >= 0) & (tt < len(lut))]
    out = []
    ct = torch.from_numpy(c.astype(np.int64))
    for n in NGRAM_SIZES:
        h = ngram_hashes(ct, n).numpy()
        if not len(h):
            continue
        j = np.clip(np.searchsorted(table, h), 0, len(table) - 1)
        out.append(np.unique(j[table[j] == h]))
    return d, (np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--control", type=int, default=5000)
    ap.add_argument("--procs", type=int, default=min(os.cpu_count() or 1, 64))
    args = ap.parse_args()
    torch.set_num_threads(1)

    tdir = os.path.join(args.work, "table")
    tab = np.load(os.path.join(tdir, "table.npz"))
    segs = np.load(os.path.join(tdir, "segments.npz"))
    meta = json.load(open(os.path.join(tdir, "sources.json")))
    csr_off, csr_seg = tab["csr_off"], tab["csr_seg"]
    attrib = load_attrib(meta["refs"], len(segs["n"]))
    if attrib is None:                        # no --attribute extraction: the source each segment was kept from
        attrib = [[meta["sources"][int(s)]] for s in segs["src"]]

    z = np.load(os.path.join(args.cmp, "docs.npz"))
    theirs, ours = z["theirs"], z["ours"]
    rng = np.random.default_rng(0)
    neither = np.flatnonzero(~theirs & ~ours)
    groups = {"both": np.flatnonzero(theirs & ours), "theirs_only": np.flatnonzero(theirs & ~ours),
              "ours_only": np.flatnonzero(ours & ~theirs),
              "control_kept_by_both": np.sort(rng.choice(neither, size=min(args.control, len(neither)), replace=False))}
    _G.update(idx=Index(args.orig + ".idx"), prefix=args.orig, lut=np.load(os.path.join(tdir, "lut.npy")),
              table=tab["hash"])
    todo = np.unique(np.concatenate(list(groups.values())))
    hits = {}
    with Pool(args.procs) as pool:            # fork: the table is shared copy-on-write
        for d, g in pool.imap_unordered(_doc_hits, todo.tolist(), chunksize=8):
            hits[d] = g
    print(f"{len(todo)} documents looked up", flush=True)

    report = {}
    for name, ids in groups.items():
        n_grams = np.array([len(hits[d]) for d in ids])
        best_seg_grams, benches = [], Counter()
        for d in ids:
            g = hits[d]
            if not len(g):
                best_seg_grams.append(0)
                continue
            k = csr_off[g + 1] - csr_off[g]
            segl = csr_seg[np.repeat(csr_off[g], k) + np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)]
            sv, sc = np.unique(segl, return_counts=True)
            best_seg_grams.append(int(sc.max()))
            for b in attrib[int(sv[np.argmax(sc)])]:
                benches[b] += 1
        b = np.array(best_seg_grams)
        report[name] = {
            "docs": int(len(ids)),
            "with_any_table_ngram_pct": round(100 * float((n_grams > 0).mean()), 2) if len(ids) else 0,
            "with_>=3_ngrams_of_one_segment_pct": round(100 * float((b >= 3).mean()), 2) if len(ids) else 0,
            "with_>=10_ngrams_of_one_segment_pct": round(100 * float((b >= 10).mean()), 2) if len(ids) else 0,
            "median_distinct_ngrams": float(np.median(n_grams)) if len(ids) else 0,
            "top_benchmarks_of_best_segment": benches.most_common(12),
        }
    json.dump(report, open(os.path.join(args.cmp, "explain.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
