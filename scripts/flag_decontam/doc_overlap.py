"""How much of one dataset is contained in another (e.g. openthoughts-3 inside mixture-vitae), with the
decontamination matcher: word-level canonical stream, longest verbatim runs (formatting-independent).

    # 1. a random sample of documents of A as references (one segment per document)
    python scripts/flag_decontam/doc_overlap.py refs --prefix <A prefix> --n 5000 --out $WORK/refs/refs.jsonl.gz
    # 2. build_table.py --refs ..., scan.py --mode scan --shards <file with B's prefixes>  (run_step.sh)
    # 3. per sampled document of A: the longest run found in B, as a share of the document
    python scripts/flag_decontam/doc_overlap.py report --work $WORK

The report: coverage histogram of the sampled documents (best single run / document tokens, canonical),
the share >= 0.7 (contained, by the decontamination rule) and >= 0.3, and a few examples.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Index  # noqa: E402

TOKENIZER = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k/tokenizer.model"


def cmd_refs(a):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    V = sp.get_piece_size()
    idx = Index(a.prefix + ".idx")
    docs = np.sort(np.random.default_rng(a.seed).choice(idx.doc_count - 1, size=a.n, replace=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as f:
        for d in docs:
            s0, s1 = int(idx.doc_idx[d]), int(idx.doc_idx[d + 1])
            t = np.fromfile(a.prefix + ".bin", np.int32, count=int(idx.lengths[s0:s1].sum()),
                            offset=int(idx.pointers[s0]))
            text = sp.decode([int(x) for x in t if 2 < x < V])[:a.max_chars]
            f.write(json.dumps({"src": f"{a.name}:{int(d)}", "split": "test", "text": text}) + "\n")
    print(f"{a.n} documents of {a.prefix} -> {a.out}")


def cmd_report(a):
    segs = np.load(os.path.join(a.work, "table", "segments.npz"))
    meta = json.load(open(os.path.join(a.work, "table", "sources.json")))
    n_tok = segs["n_tokens"].astype(np.int64)
    best = np.zeros(len(n_tok), np.int64)
    for f in glob.glob(os.path.join(a.work, "hits", "*", "*.npz")):
        h = np.load(f)["hits"]
        if len(h):
            np.maximum.at(best, h[:, 1], h[:, 2])
    ok = segs["n"] > 0
    cov = np.where(ok, best / np.maximum(n_tok, 1), 0.0)[ok]
    bins = [0, 1e-9, 0.1, 0.3, 0.5, 0.7, 0.9, 10]
    hist = np.histogram(cov, bins)[0]
    rep = {"sampled_documents": int(len(n_tok)), "matchable": int(ok.sum()),
           "coverage_hist": {f"{lo:g}-{hi:g}": int(c) for lo, hi, c in zip(bins[:-1], bins[1:], hist)},
           "contained_ge_0.7_pct": round(100 * float((cov >= 0.7).mean()), 2),
           "ge_0.3_pct": round(100 * float((cov >= 0.3).mean()), 2),
           "any_run_pct": round(100 * float((cov > 0).mean()), 2),
           "refs": meta["refs"]}
    json.dump(rep, open(os.path.join(a.work, "_report_doc_overlap.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["refs", "report"])
    ap.add_argument("--prefix")
    ap.add_argument("--name", default="docs")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=100_000, help="extract_refs.py's segment limit")
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    a = ap.parse_args()
    {"refs": cmd_refs, "report": cmd_report}[a.mode](a)


if __name__ == "__main__":
    main()
