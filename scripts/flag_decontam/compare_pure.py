"""Compare our decontamination of one shard with an independently filtered ("pure") re-tokenisation of it.

    python scripts/flag_decontam/compare_pure.py --orig <flag prefix> --pure <pure prefix> \
        --work $WORK --out <dir> [--frac 0.7] [--procs 64] [--samples 40]

The pure shard was filtered by another pipeline and re-tokenised from text (possibly in another
order, without BOS), so documents are matched by content, not position:
  1. every document of both shards is hashed twice -- its exact token sequence (leading <bos> and
     trailing <eos> stripped) and its word-level canonical stream (common.build_lut(words=True):
     case, punctuation and whitespace-insensitive, for re-extraction differences);
  2. an original document is "kept by them" when its exact hash occurs in the pure shard (counted
     as a multiset: duplicates are matched one by one), else when its canonical hash does; the
     rest are "removed by them". Pure documents matching no original ones are counted ("new or
     modified").
Our decision for the original shard is recomputed from the scan's hits (finalize.py's rule:
longest verbatim run >= max(--min-run, min(ceil(--frac * segment tokens), --abs-run))), with the
best coverage per document kept for the analysis.

Writes <out>/summary.json (confusion matrix in documents and tokens, match statistics, our
coverage histogram for their-only removals, benchmark sources of our-only / both removals),
<out>/docs.npz (per original doc: theirs / ours / best coverage / best segment) and
<out>/samples.txt (decoded excerpts of disagreements).
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Index  # noqa: E402

TOKENIZER = "/gpfs/projects/ehpc533/tokenizers/tokenizer-256k/tokenizer.model"
BOS, EOS = 1, 2
_G = {}


def _init(lut_path):
    _G["lut"] = np.load(lut_path)
    _G["idx"] = {}


def _hash_range(job):
    """(prefix, d0, d1) -> exact and canonical 64-bit hashes of documents d0..d1-1."""
    prefix, d0, d1 = job
    idx = _G["idx"].get(prefix) or _G["idx"].setdefault(prefix, Index(prefix + ".idx"))
    lut = _G["lut"]
    s0, s1 = int(idx.doc_idx[d0]), int(idx.doc_idx[d1])
    start = int(idx.pointers[s0])
    end = int(idx.pointers[s1 - 1]) + int(idx.lengths[s1 - 1]) * 4
    buf = np.fromfile(prefix + ".bin", dtype=np.int32, count=(end - start) // 4, offset=start)
    ex = np.empty(d1 - d0, np.uint64)
    ca = np.empty(d1 - d0, np.uint64)
    for k, d in enumerate(range(d0, d1)):
        a, b = int(idx.doc_idx[d]), int(idx.doc_idx[d + 1])
        o = (int(idx.pointers[a]) - start) // 4
        n = int(idx.lengths[a:b].sum())
        t = buf[o:o + n]
        i, j = 0, len(t)
        while i < j and t[i] == BOS:
            i += 1
        while j > i and t[j - 1] == EOS:
            j -= 1
        t = t[i:j]
        ex[k] = int.from_bytes(hashlib.blake2b(t.tobytes(), digest_size=8).digest(), "little")
        tt = t.astype(np.int64)
        c = lut[np.clip(tt, 0, len(lut) - 1)]
        c = c[(c >= 0) & (tt < len(lut))]
        ca[k] = int.from_bytes(hashlib.blake2b(c.astype(np.int32).tobytes(), digest_size=8).digest(), "little")
    return d0, ex, ca


def hash_docs(prefix, pool, chunk=2000):
    idx = Index(prefix + ".idx")
    n = idx.doc_count - 1
    ex, ca = np.empty(n, np.uint64), np.empty(n, np.uint64)
    for d0, e, c in pool.imap_unordered(_hash_range, [(prefix, a, min(a + chunk, n)) for a in range(0, n, chunk)]):
        ex[d0:d0 + len(e)], ca[d0:d0 + len(c)] = e, c
    tok = np.add.reduceat(idx.lengths.astype(np.int64), idx.doc_idx[:-1]) if n else np.zeros(0, np.int64)
    return ex, ca, tok


def multiset_match(orig, pure):
    """Boolean per original element: matched against the pure multiset (one-to-one per value)."""
    if len(orig) == 0 or len(pure) == 0:
        return np.zeros(len(orig), bool)
    order = np.argsort(orig, kind="stable")
    so = orig[order]
    pv, pc = np.unique(pure, return_counts=True)
    first = np.ones(len(so), bool)
    first[1:] = so[1:] != so[:-1]
    grp = np.cumsum(first) - 1
    rank = np.arange(len(so)) - np.flatnonzero(first)[grp]       # occurrence number within equal values
    j = np.searchsorted(pv, so)
    j = np.clip(j, 0, len(pv) - 1)
    avail = np.where(pv[j] == so, pc[j], 0)
    m = np.zeros(len(orig), bool)
    m[order] = rank < avail
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--pure", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--abs-run", type=int, default=200)
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--procs", type=int, default=min(os.cpu_count() or 1, 64))
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from scan import shard_key

    with Pool(args.procs, initializer=_init, initargs=(os.path.join(args.work, "table", "lut.npy"),)) as pool:
        o_ex, o_ca, o_tok = hash_docs(args.orig, pool)
        p_ex, p_ca, p_tok = hash_docs(args.pure, pool)
    print(f"hashed {len(o_ex)} original, {len(p_ex)} pure documents", flush=True)

    kept_exact = multiset_match(o_ex, p_ex)
    p_used_exact = multiset_match(p_ex, o_ex)
    kept_canon = np.zeros(len(o_ex), bool)
    rest = ~kept_exact
    kept_canon[rest] = multiset_match(o_ca[rest], p_ca[~p_used_exact])
    theirs = ~(kept_exact | kept_canon)                         # removed by them
    p_unmatched = int((~p_used_exact).sum()) - int(kept_canon.sum())

    segs = np.load(os.path.join(args.work, "table", "segments.npz"))
    meta = json.load(open(os.path.join(args.work, "table", "sources.json")))
    n_tok = segs["n_tokens"].astype(np.int64)
    need = np.maximum(args.min_run, np.minimum(np.ceil(args.frac * n_tok), args.abs_run))
    need[segs["n"] == 0] = np.iinfo(np.int64).max
    idx = Index(args.orig + ".idx")
    hits = [np.load(f)["hits"] for f in glob.glob(os.path.join(args.work, "hits", shard_key(args.orig, args.root), "*.npz"))]
    h = np.concatenate(hits) if hits else np.zeros((0, 5), np.int64)
    assert len(hits), f"no hits for {args.orig}: scan it first"
    doc = np.searchsorted(idx.doc_idx, h[:, 0], "right") - 1
    cov = h[:, 2] / np.maximum(n_tok[h[:, 1]], 1)
    flag = h[:, 2] >= need[h[:, 1]]
    ours = np.zeros(len(o_ex), bool)
    ours[doc[flag]] = True
    best_cov = np.zeros(len(o_ex))
    best_seg = np.full(len(o_ex), -1, np.int64)
    order = np.lexsort((cov, doc))                              # last per doc = max coverage
    last = np.ones(len(order), bool)
    last[:-1] = doc[order][1:] != doc[order][:-1]
    best_cov[doc[order][last]] = cov[order][last]
    best_seg[doc[order][last]] = h[order][last][:, 1]

    attrib = None
    ap_ = meta["refs"].replace(".jsonl.gz", ".attrib.jsonl.gz")
    if os.path.exists(ap_):
        attrib = [sorted({a[0] for a in json.loads(l)["also_in"] if a[3]}) for l in gzip.open(ap_, "rt", encoding="utf-8")]
    name = (lambda s: "|".join(attrib[s]) if attrib is not None else meta["sources"][segs["src"][s]])

    def cell(m):
        return {"docs": int(m.sum()), "tokens": int(o_tok[m].sum())}

    both, t_only, o_only = theirs & ours, theirs & ~ours, ours & ~theirs
    hist_edges = [0, 1e-9, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001, 100]
    t_hist = np.histogram(best_cov[t_only], hist_edges)[0].tolist()
    summary = {
        "orig": args.orig, "pure": args.pure, "rule": {"frac": args.frac, "abs_run": args.abs_run, "min_run": args.min_run},
        "orig_docs": int(len(o_ex)), "pure_docs": int(len(p_ex)),
        "orig_tokens": int(o_tok.sum()), "pure_tokens": int(p_tok.sum()),
        "matched_exact": int(kept_exact.sum()), "matched_canonical_only": int(kept_canon.sum()),
        "pure_unmatched_new_or_modified": p_unmatched,
        "removed_by_them": cell(theirs), "removed_by_us": cell(ours),
        "both": cell(both), "theirs_only": cell(t_only), "ours_only": cell(o_only), "neither": cell(~theirs & ~ours),
        "theirs_only_our_best_coverage_hist": dict(zip(["none (no run >= 20 tok)", "(0,0.1)", "[0.1,0.3)", "[0.3,0.5)",
                                                         "[0.5,0.7)", "[0.7,0.9)", "[0.9,1]", ">1"], t_hist)),
        "ours_only_benchmarks": Counter(name(s) for s in best_seg[o_only] if s >= 0).most_common(20),
        "both_benchmarks": Counter(name(s) for s in best_seg[both] if s >= 0).most_common(20),
        "theirs_only_partial_benchmarks": Counter(name(s) for s in best_seg[t_only] if s >= 0).most_common(20),
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    np.savez(os.path.join(args.out, "docs.npz"), theirs=theirs, ours=ours, best_cov=best_cov, best_seg=best_seg,
             tokens=o_tok, kept_exact=kept_exact, kept_canon=kept_canon)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("benchmarks")}, indent=1), flush=True)

    if args.samples <= 0:
        return
    # decoded excerpts of the disagreements (and a few agreements)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    rng = np.random.default_rng(0)
    texts = {}
    want = {}
    groups = {"theirs_only": np.flatnonzero(t_only), "ours_only": np.flatnonzero(o_only), "both": np.flatnonzero(both)}
    for g, ids in groups.items():
        pick = rng.choice(ids, size=min(args.samples if g != "both" else args.samples // 4, len(ids)), replace=False)
        want[g] = sorted(pick.tolist())
    segs_wanted = {int(best_seg[d]) for ids in want.values() for d in ids if best_seg[d] >= 0}
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in segs_wanted:
                texts[i] = json.loads(line)["text"]
    with open(os.path.join(args.out, "samples.txt"), "w") as fh:
        for g, ids in want.items():
            for d in ids:
                a, b = int(idx.doc_idx[d]), int(idx.doc_idx[d + 1])
                t = np.fromfile(args.orig + ".bin", np.int32, count=int(idx.lengths[a:b].sum()), offset=int(idx.pointers[a]))
                s = int(best_seg[d])
                fh.write(f"=== {g} doc {d} ({len(t)} tokens) | our best coverage {best_cov[d]:.2f}"
                         + (f" on {name(s)} segment {s}" if s >= 0 else "") + "\n")
                if s >= 0:
                    fh.write("--- benchmark segment: " + texts.get(s, "")[:500].replace("\n", " | ") + "\n")
                fh.write("--- document start: " + sp.decode(t[:400].tolist())[:1500].replace("\n", " | ") + "\n\n")
    print(f"wrote {args.out}/summary.json, docs.npz, samples.txt", flush=True)


if __name__ == "__main__":
    main()
