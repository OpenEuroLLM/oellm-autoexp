"""Checks of the flag decontamination (docs/flag_post_decontamination.md).

    python scripts/flag_decontam/verify.py roundtrip --shards s.txt [--docs 300]
    python scripts/flag_decontam/verify.py tree --out $OUT [--docs 200]
    python scripts/flag_decontam/verify.py audit --work $WORK --shard <prefix> [--docs 20000]

roundtrip: the premise of token-level matching. Decode documents of the .bin with the tokenizer,
           re-encode them with SentencePiece (as build_table.py encodes the references) and
           compare the canonical streams. Mismatches would be text the references could never
           match token-wise.
tree:      every rewritten .idx loads with Megatron's IndexedDataset; kept documents are
           token-identical to the original documents, dropped ones are gone, counts add up.
audit:     recall of the token matcher against a plain text matcher: decode --docs documents
           of one shard, flag them with lower-cased word 13-grams (>= 50 % of a segment's
           word n-grams, math / code / QA benchmark segments) and compare with the documents the
           scan flagged (needs the scan's hits for that shard).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Index  # noqa: E402

TOKENIZER = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k/tokenizer.model"
ROOT = "/gpfs/projects/ehpc533/training-data-collection/flag"
AUDIT_SOURCES = re.compile(r"gsm8k|MATH500|AIME|AMC|gpqa|human-eval|HumanEval|mbpp|MBPP|mgsm|PolyMath|jeebench|IFEval|mmlu",
                           re.I)


def doc_tokens(prefix, idx, d):
    s0, s1 = idx.doc_idx[d], idx.doc_idx[d + 1]
    parts = []
    for s in range(s0, s1):
        parts.append(np.fromfile(prefix + ".bin", dtype=idx.dtype, count=int(idx.lengths[s]),
                                 offset=int(idx.pointers[s])))
    return np.concatenate(parts) if parts else np.zeros(0, idx.dtype)


def roundtrip(args):
    import sentencepiece as spm
    from common import build_lut
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    lut, _ = build_lut(sp)
    rng = np.random.default_rng(0)
    tot = same = tok = diff = 0
    for prefix in [line.split()[-1] for line in open(args.shards) if line.strip()]:
        idx = Index(prefix + ".idx")
        for d in rng.choice(idx.doc_count - 1, size=min(args.docs, idx.doc_count - 1), replace=False):
            t = doc_tokens(prefix, idx, int(d))[:20000]
            a = lut[t.astype(np.int64)]
            a = a[a >= 0]
            b = lut[np.asarray(sp.encode(sp.decode(t.tolist())), dtype=np.int64)]
            b = b[b >= 0]
            tot += 1
            same += int(np.array_equal(a, b))
            tok += len(a)
            m = min(len(a), len(b))
            diff += int((a[:m] != b[:m]).sum()) + abs(len(a) - len(b))
        print(f"{os.path.relpath(prefix, ROOT)}: {same}/{tot} documents identical so far", flush=True)
    print(json.dumps({"documents": tot, "identical": same, "canonical_tokens": tok,
                      "positional_mismatch_frac": diff / max(tok, 1)}))


def tree(args):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "submodules", "Megatron-LM"))
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    rng = np.random.default_rng(0)
    removed = sorted(glob.glob(os.path.join(args.out, "**", "*.removed.npz"), recursive=True))
    print(f"{len(removed)} rewritten shards", flush=True)
    for r in removed:
        new_p = r.removesuffix(".removed.npz")
        old_p = os.path.join(args.root, os.path.relpath(new_p, args.out))
        old, new = IndexedDataset(old_p), IndexedDataset(new_p)
        dropped = np.unique(np.load(r)["doc"])
        n_old, n_new = len(old.document_indices) - 1, len(new.document_indices) - 1
        assert n_new == n_old - len(dropped), (r, n_old, n_new, len(dropped))
        kept = np.setdiff1d(np.arange(n_old), dropped)
        for i in rng.choice(n_new, size=min(args.docs, n_new), replace=False):
            a, b = new.get(int(new.document_indices[i])), old.get(int(old.document_indices[kept[i]]))
            assert np.array_equal(a, b), (r, i)
        print(f"ok {os.path.relpath(new_p, args.out)}: {n_old} -> {n_new} docs", flush=True)
    print("tree ok")


def word_grams(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def audit(args):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    meta = json.load(open(os.path.join(args.work, "table", "sources.json")))
    seg_grams, gram_segs = {}, {}
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if AUDIT_SOURCES.search(r["src"]):
                g = word_grams(r["text"])
                if len(g) >= 3:
                    seg_grams[i] = len(g)
                    for h in g:
                        gram_segs.setdefault(h, []).append(i)
    print(f"audit references: {len(seg_grams)} segments", flush=True)
    idx = Index(args.shard + ".idx")
    rng = np.random.default_rng(0)
    docs = np.sort(rng.choice(idx.doc_count - 1, size=min(args.docs, idx.doc_count - 1), replace=False))
    text_flag = set()
    for d in docs:
        counts = {}
        for h in word_grams(sp.decode(doc_tokens(args.shard, idx, int(d)).tolist())):
            for s in gram_segs.get(h, ()):
                counts[s] = counts.get(s, 0) + 1
        if any(c >= max(3, 0.5 * seg_grams[s]) for s, c in counts.items()):
            text_flag.add(int(d))
    from scan import shard_key
    segs = np.load(os.path.join(args.work, "table", "segments.npz"))
    need = np.maximum(3, np.ceil(0.5 * segs["n_grams"]))
    tok_flag = set()
    for f in glob.glob(os.path.join(args.work, "hits", shard_key(args.shard, args.root), "*.npz")):
        h = np.load(f)["hits"]
        h = h[h[:, 2] >= need[h[:, 1]]]
        tok_flag |= set((np.searchsorted(idx.doc_idx, h[:, 0], "right") - 1).tolist())
    tok_flag &= set(docs.tolist())
    both = text_flag & tok_flag
    print(json.dumps({"docs": len(docs), "text_flagged": len(text_flag), "token_flagged": len(tok_flag),
                      "both": len(both), "text_only": sorted(text_flag - tok_flag)[:20],
                      "recall_vs_text": len(both) / max(len(text_flag), 1)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["roundtrip", "tree", "audit"])
    ap.add_argument("--shards")
    ap.add_argument("--shard")
    ap.add_argument("--work")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="/gpfs/projects/ehpc533/training-data-collection/flag-decon-v1")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--docs", type=int, default=300)
    args = ap.parse_args()
    {"roundtrip": roundtrip, "tree": tree, "audit": audit}[args.cmd](args)


if __name__ == "__main__":
    main()
