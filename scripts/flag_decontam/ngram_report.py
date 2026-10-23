"""Per-benchmark n-gram hit rates over the whole mix (comparable to n-gram-overlap contamination tables).

    python scripts/flag_decontam/ngram_report.py --work $WORK [--out $WORK/_report_ngrams.tsv]

From the `grams` arrays scan.py stores per unit (the distinct table n-grams the unit contains):
an n-gram is "hit" if it occurs anywhere in the mix. Per benchmark (every source that holds the
segment, refs.attrib.jsonl.gz, if present; else the one source extraction kept): segments, total
and hit n-grams of its matchable segments, hit %, and the share of segments with >= 1 / >= 50 %
of their n-grams hit. N-grams dropped as template / common (build, prune) are not in the table
and count neither as total nor as hit; the escaped variants of a segment count as its n-grams.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finalize import load_attrib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    tdir = os.path.join(args.work, "table")
    tab = np.load(os.path.join(tdir, "table.npz"))
    segs = np.load(os.path.join(tdir, "segments.npz"))
    meta = json.load(open(os.path.join(tdir, "sources.json")))
    G, S = len(tab["hash"]), len(segs["n"])

    hit = np.zeros(G, bool)
    files = glob.glob(os.path.join(args.work, "hits", "*", "*.npz"))
    for f in files:
        g = np.load(f)["grams"]
        hit[g] = True
    print(f"{len(files)} unit files, {int(hit.sum())} of {G} n-grams hit", flush=True)

    nseg = np.diff(tab["csr_off"])
    seg_of = tab["csr_seg"]                              # entries of the CSR, grouped by n-gram
    total = np.bincount(seg_of, minlength=S)
    hits = np.bincount(seg_of[np.repeat(hit, nseg)], minlength=S)

    attrib = load_attrib(meta["refs"], S)
    names = meta["sources"]
    acc = defaultdict(lambda: np.zeros(5, np.int64))     # segments, total, hit, segs >= 1 hit, segs >= 50 %
    for sg in np.flatnonzero((segs["n"] > 0) & (total > 0)):
        row = np.array([1, total[sg], hits[sg], hits[sg] > 0, 2 * hits[sg] >= total[sg]], np.int64)
        for b in (attrib[sg] if attrib is not None else [names[segs["src"][sg]]]):
            acc[b] += row
    out = args.out or os.path.join(args.work, "_report_ngrams.tsv")
    with open(out, "w") as fh:
        fh.write("benchmark_source\tsegments\ttotal_ngrams\thit_ngrams\thit_pct\tsegments_any_hit_pct\tsegments_half_hit_pct\n")
        for b, (n, t, h, a, half) in sorted(acc.items(), key=lambda x: -x[1][2] / max(x[1][1], 1)):
            fh.write(f"{b}\t{n}\t{t}\t{h}\t{100 * h / max(t, 1):.4f}\t{100 * a / max(n, 1):.2f}\t{100 * half / max(n, 1):.2f}\n")
    print(open(out).read())


if __name__ == "__main__":
    main()
