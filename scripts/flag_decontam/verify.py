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
           of one shard, undo JSON escapes, and flag them when their longest common contiguous
           run of lower-cased words with a (math / code / QA benchmark) segment covers >= --frac
           of it or is >= --abs-words long; compare with the documents the scan flagged.
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


def _tree_one(job):
    """Check one rewritten shard; returns a status line (raises on a mismatch)."""
    r, out, root, docs = job
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "submodules", "Megatron-LM"))
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    rng = np.random.default_rng(abs(hash(r)) % (1 << 32))
    new_p = r.removesuffix(".removed.npz")
    old_p = os.path.join(root, os.path.relpath(new_p, out))
    old, new = IndexedDataset(old_p), IndexedDataset(new_p)
    dropped = np.unique(np.load(r)["doc"])
    n_old, n_new = len(old.document_indices) - 1, len(new.document_indices) - 1
    assert n_new == n_old - len(dropped), (r, n_old, n_new, len(dropped))
    kept = np.setdiff1d(np.arange(n_old), dropped)
    for i in rng.choice(n_new, size=min(docs, n_new), replace=False):
        a, b = new.get(int(new.document_indices[i])), old.get(int(old.document_indices[kept[i]]))
        assert np.array_equal(a, b), (r, i)
    return f"ok {os.path.relpath(new_p, out)}: {n_old} -> {n_new} docs"


def tree(args):
    """Every rewritten shard of --out; split over nodes (--part / --nparts, from SLURM) and --procs."""
    from multiprocessing import Pool
    removed = sorted(glob.glob(os.path.join(args.out, "**", "*.removed.npz"), recursive=True))
    mine = removed[args.part::args.nparts]
    print(f"{len(removed)} rewritten shards, part {args.part}/{args.nparts}: {len(mine)}", flush=True)
    with Pool(args.procs) as pool:
        for line in pool.imap_unordered(_tree_one, [(r, args.out, args.root, args.docs) for r in mine]):
            print(line, flush=True)
    print(f"tree ok (part {args.part}/{args.nparts}, {len(mine)} shards)")


_ESC = re.compile(r"\\u([0-9a-fA-F]{4})|\\([nrt\"\\/])")


def unescape(text):
    """Undo JSON string escapes (agent trajectories store their text JSON-escaped)."""
    return _ESC.sub(lambda m: chr(int(m.group(1), 16)) if m.group(1) else
                    {"n": "\n", "r": "\r", "t": "\t"}.get(m.group(2), m.group(2)), text)


def words(text):
    return re.findall(r"\w+", text.lower())


def word_grams(w, n=13):
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def longest_common_run(a, b):
    from difflib import SequenceMatcher
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


_A = {}


def _audit_init(tokenizer, shard):
    import sentencepiece as spm   # the reference dicts come with the fork (module global _A)
    _A.update(sp=spm.SentencePieceProcessor(model_file=tokenizer), shard=shard, idx=Index(shard + ".idx"))


def _audit_doc(d):
    w = words(unescape(_A["sp"].decode(doc_tokens(_A["shard"], _A["idx"], int(d)).tolist())))
    cand = {}
    for h in word_grams(w):
        for s in _A["gram_segs"].get(h, ()):
            cand[s] = cand.get(s, 0) + 1
    for s, c in cand.items():
        if c >= 3:
            run = longest_common_run(w, _A["seg_words"][s])
            if run >= _A["frac"] * len(_A["seg_words"][s]) or run >= _A["abs_words"]:
                return int(d)
    return None


def audit(args):
    """Recall of the token matcher against a text matcher (JSON-unescaped, lower-cased words):
    a document is text-flagged when its longest common contiguous word run with a segment
    covers >= --frac of the segment's words, or is >= --abs-words words."""
    from multiprocessing import Pool
    meta = json.load(open(os.path.join(args.work, "table", "sources.json")))
    seg_words, gram_segs = {}, {}
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if AUDIT_SOURCES.search(r["src"]) and len(r["text"]) >= 80:
                w = words(r["text"])
                if len(w) >= 13:
                    seg_words[i] = w
                    for h in word_grams(w):
                        gram_segs.setdefault(h, set()).add(i)
    print(f"audit references: {len(seg_words)} segments", flush=True)
    idx = Index(args.shard + ".idx")
    rng = np.random.default_rng(0)
    docs = np.sort(rng.choice(idx.doc_count - 1, size=min(args.docs, idx.doc_count - 1), replace=False))
    _A.update(seg_words=seg_words, gram_segs=gram_segs, frac=args.frac, abs_words=args.abs_words)
    with Pool(args.procs, initializer=_audit_init, initargs=(args.tokenizer, args.shard)) as pool:
        text_flag = {d for d in pool.imap_unordered(_audit_doc, docs.tolist(), chunksize=64) if d is not None}
    from scan import shard_key
    segs = np.load(os.path.join(args.work, "table", "segments.npz"))
    n_tok = segs["n_tokens"].astype(np.int64)
    need = np.maximum(20, np.minimum(np.ceil(args.frac * n_tok), 200))
    from finalize import apply_fixes, load_fixes
    fixes = load_fixes(args.work, args.fixes) if os.path.exists(os.path.join(args.work, args.fixes)) else {}
    key = shard_key(args.shard, args.root)
    tok_flag = set()
    for f in glob.glob(os.path.join(args.work, args.hits_dir, key, "*.npz")):
        h = apply_fixes(np.load(f)["hits"], fixes.get(key), 20)
        h = h[h[:, 2] >= need[h[:, 1]]]
        tok_flag |= set((np.searchsorted(idx.doc_idx, h[:, 0], "right") - 1).tolist())
    tok_flag &= set(docs.tolist())
    both = text_flag & tok_flag
    print(json.dumps({"docs": len(docs), "text_flagged": len(text_flag), "token_flagged": len(tok_flag),
                      "both": len(both), "text_only": sorted(text_flag - tok_flag)[:20],
                      "token_only": sorted(tok_flag - text_flag)[:20],
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
    ap.add_argument("--frac", type=float, default=0.9, help="audit: run / segment length that flags")
    ap.add_argument("--abs-words", type=int, default=150, help="audit: a run of this many words flags")
    ap.add_argument("--hits-dir", default="hits")
    ap.add_argument("--fixes", default="recheck_fixes.npz", help="recheck.py's corrections, applied if present")
    ap.add_argument("--procs", type=int, default=min(os.cpu_count() or 1, 64))
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_NODEID", 0)))
    ap.add_argument("--nparts", type=int, default=int(os.environ.get("SLURM_NNODES", 1)))
    args = ap.parse_args()
    {"roundtrip": roundtrip, "tree": tree, "audit": audit}[args.cmd](args)


if __name__ == "__main__":
    main()
