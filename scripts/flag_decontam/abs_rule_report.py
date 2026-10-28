"""Which documents does the absolute-run rule (--abs-run) drop on its own, and for which segments?

    python scripts/flag_decontam/abs_rule_report.py --work $WORK [--frac 0.7] [--abs-run 200] [--top 40]

finalize.py drops a document when a segment's longest verbatim run covers >= --frac of the segment
OR is >= --abs-run tokens. The second branch is meant for long excerpts of passages, but it also
fires on long boilerplate inside a benchmark item (an Asymptote plotting helper shared by many
MATH-style problems, licence headers, test harnesses). Over all hits of the scan: the documents
flagged only through the absolute branch (no segment reaches --frac), per source and per segment,
with the segment text, so boilerplate segments can be listed as generic.

Writes $WORK/_report_abs_rule.json and prints it.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Index  # noqa: E402
from scan import shard_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--abs-run", type=int, default=200)
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    segs = np.load(os.path.join(args.work, "table", "segments.npz"))
    meta = json.load(open(os.path.join(args.work, "table", "sources.json")))
    n_tok = segs["n_tokens"].astype(np.int64)
    frac_need = np.maximum(args.min_run, np.ceil(args.frac * n_tok))
    frac_need[segs["n"] == 0] = np.iinfo(np.int64).max
    rows = [l.split()[1] for l in open(os.path.join(args.root, "datamix.txt")) if l.strip()]

    seg_docs, seg_tok = Counter(), Counter()
    by_src = defaultdict(lambda: [0, 0])
    total_docs = total_tok = 0
    for rel in rows:
        p = os.path.join(args.root, rel)
        files = glob.glob(os.path.join(args.work, "hits", shard_key(p, args.root), "*.npz"))
        if not files:
            continue
        h = np.concatenate([np.load(f)["hits"] for f in files])
        if not len(h):
            continue
        idx = Index(p + ".idx")
        doc = np.searchsorted(idx.doc_idx, h[:, 0], "right") - 1
        by_frac = h[:, 2] >= frac_need[h[:, 1]]
        by_abs = (h[:, 2] >= args.abs_run) & ~by_frac & (segs["n"][h[:, 1]] > 0)
        only = np.setdiff1d(np.unique(doc[by_abs]), np.unique(doc[by_frac]))
        if not len(only):
            continue
        tok = idx.lengths[idx.doc_idx[only]].astype(np.int64)
        by_src[rel.split("/")[0]][0] += len(only)
        by_src[rel.split("/")[0]][1] += int(tok.sum())
        total_docs += len(only)
        total_tok += int(tok.sum())
        sel = by_abs & np.isin(doc, only)
        for s, dd in set(zip(h[sel, 1].tolist(), doc[sel].tolist())):
            seg_docs[s] += 1
        print(f"{rel}: {len(only)} docs only by the absolute rule", flush=True)

    top = seg_docs.most_common(args.top)
    want = {s for s, _ in top}
    texts = {}
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in want:
                r = json.loads(line)
                texts[i] = (r["src"], r["text"][:200].replace("\n", " | "))
    report = {"frac": args.frac, "abs_run": args.abs_run, "docs_only_by_abs_rule": total_docs,
              "tokens_only_by_abs_rule": total_tok,
              "by_source": {k: {"docs": v[0], "tokens": v[1]} for k, v in sorted(by_src.items(), key=lambda x: -x[1][1])},
              "top_segments": [{"seg": s, "docs": n, "n_tokens": int(n_tok[s]), "src": texts.get(s, ("", ""))[0],
                                "text": texts.get(s, ("", ""))[1]} for s, n in top]}
    json.dump(report, open(os.path.join(args.work, "_report_abs_rule.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
