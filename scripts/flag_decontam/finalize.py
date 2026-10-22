"""Decide which documents to drop and write the decontaminated tree (docs/flag_post_decontamination.md).

    python scripts/flag_decontam/finalize.py --work $WORK --out $OUT [--frac 0.5] [--dry-run]

A (sequence, segment) pair from scan.py is contamination when the sequence contains at least
max(--min-hits, ceil(--frac * n_grams)) of the segment's distinct table n-grams. A document is
dropped when any of its sequences has one. Segments that would drop more than --max-docs
documents over the whole mix are reported and not applied (generic text that survived pruning:
check them in the report before raising the limit).

Output tree $OUT mirrors the source root for every shard of <root>/datamix.txt:
  <prefix>.bin        symlink to the original .bin (tokens are never copied)
  <prefix>.idx        new index of the kept documents (original pointers into the same .bin),
                      or a symlink to the original .idx when nothing is dropped
  <shard>.info.json   the original's, with the new .idx size / md5 / counts
  <prefix>.removed.npz   dropped documents: doc, seq, seg, matched, n_grams (per flagged pair)
plus datamix.txt (unchanged weights) and _report/{report.json, sources.tsv, benchmarks.tsv,
top_segments.tsv, generic_segments.tsv}.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Index, filter_index, write_index  # noqa: E402
from scan import shard_key  # noqa: E402


def load_hits(work, key, seq_count):
    files = sorted(glob.glob(os.path.join(work, "hits", key, "*.npz")))
    hits, covered = [], np.zeros(seq_count + 1, dtype=np.int64)
    for f in files:
        z = np.load(f)
        s0, s1 = z["seq_range"]
        covered[s0] += 1
        covered[s1] -= 1
        hits.append(z["hits"])
    cov = np.cumsum(covered)[:-1]
    complete = bool(seq_count == 0 or (cov == 1).all())
    h = np.concatenate(hits) if hits else np.zeros((0, 3), np.int64)
    return h, complete


def md5(path):
    m = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            m.update(b)
    return m.hexdigest()


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--out", default="/gpfs/projects/ehpc533/training-data-collection/flag-decon-v1")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true", help="report only, write no tree")
    args = ap.parse_args()

    tdir = os.path.join(args.work, "table")
    segs = np.load(os.path.join(tdir, "segments.npz"))
    meta = json.load(open(os.path.join(tdir, "sources.json")))
    src_names, n_grams = meta["sources"], segs["n_grams"].astype(np.int64)
    need = np.maximum(args.min_hits, np.ceil(args.frac * n_grams)).astype(np.int64)
    need[n_grams < args.min_hits] = np.iinfo(np.int64).max     # unmatchable segments never flag

    rows = []
    for line in open(os.path.join(args.root, "datamix.txt")):
        if line.strip():
            w, prefix = line.split()
            rows.append((float(w), prefix))

    # pass 1: flagged (doc, seg) pairs per shard
    per_shard, seg_docs = {}, defaultdict(int)
    incomplete = []
    for _, rel in rows:
        p = os.path.join(args.root, rel)
        idx = Index(p + ".idx")
        assert idx.multimodal_tail == 0, f"multimodal index {p}"
        hits, complete = load_hits(args.work, shard_key(p, args.root), idx.seq_count)
        if not complete:
            incomplete.append(rel)
        flag = hits[hits[:, 2] >= need[hits[:, 1]]] if len(hits) else hits
        seq2doc = np.searchsorted(idx.doc_idx, flag[:, 0], "right") - 1
        per_shard[rel] = (idx, flag, seq2doc)
        for s in np.unique(np.stack([seq2doc, flag[:, 1]], 1), axis=0)[:, 1] if len(flag) else []:
            seg_docs[int(s)] += 1
    if incomplete:
        sys.exit(f"scan incomplete for {len(incomplete)} shards, e.g. {incomplete[:3]}: rerun scan.py --mode scan")

    generic = {s for s, n in seg_docs.items() if n > args.max_docs}
    refs_text = {}
    want = set(sorted(seg_docs, key=lambda s: -seg_docs[s])[:300]) | generic
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                refs_text[i] = json.loads(line)["text"][:300].replace("\t", " ").replace("\n", " ")

    # pass 2: drop decisions, tree, report
    by_src = defaultdict(lambda: [0, 0, 0, 0, 0])        # docs, tokens, docs dropped, tokens dropped, shards
    by_bench = defaultdict(lambda: [set(), 0])            # flagged segments, docs
    total = [0, 0, 0, 0]
    for _, rel in rows:
        idx, flag, seq2doc = per_shard[rel]
        if generic and len(flag):
            ok = ~np.isin(flag[:, 1], list(generic))
            flag, seq2doc = flag[ok], seq2doc[ok]
        n_docs = idx.doc_count - 1
        doc_tok = np.add.reduceat(idx.lengths.astype(np.int64), idx.doc_idx[:-1]) if n_docs else np.zeros(0, np.int64)
        drop = np.zeros(n_docs, dtype=bool)
        drop[seq2doc] = True
        src = rel.split("/")[0]
        s = by_src[src]
        s[0] += n_docs
        s[1] += int(doc_tok.sum())
        s[2] += int(drop.sum())
        s[3] += int(doc_tok[drop].sum())
        s[4] += 1
        for d, sg in np.unique(np.stack([seq2doc, flag[:, 1]], 1), axis=0) if len(flag) else []:
            b = by_bench[src_names[segs["src"][sg]]]
            b[0].add(int(sg))
            b[1] += 1
        if args.dry_run:
            continue
        src_p, dst_p = os.path.join(args.root, rel), os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst_p), exist_ok=True)
        link(src_p + ".bin", dst_p + ".bin")
        info_src = src_p.removesuffix("_text_document") + ".info.json"
        info = json.load(open(info_src)) if os.path.exists(info_src) else {"idx": {}, "bin": {}}
        if drop.any():
            lengths, pointers, doc_idx = filter_index(idx, ~drop)
            if os.path.islink(dst_p + ".idx"):
                os.remove(dst_p + ".idx")
            write_index(dst_p + ".idx", idx.dtype_code, lengths, pointers, doc_idx)
            np.savez(dst_p + ".removed.npz", doc=seq2doc, seq=flag[:, 0], seg=flag[:, 1], matched=flag[:, 2],
                     n_grams=n_grams[flag[:, 1]])
            info["idx"].update(size=os.path.getsize(dst_p + ".idx"), md5sum=md5(dst_p + ".idx"),
                               sequence_count=int(len(lengths)), document_count=int(len(doc_idx)),
                               total_tokens=int(lengths.astype(np.int64).sum()))
            info["decontamination"] = {"dropped_documents": int(drop.sum()),
                                       "dropped_tokens": int(doc_tok[drop].sum()),
                                       "source_idx": src_p + ".idx"}
        else:
            link(src_p + ".idx", dst_p + ".idx")
        with open(dst_p.removesuffix("_text_document") + ".info.json", "w") as f:
            json.dump(info, f, indent=2)
    for k, s in by_src.items():
        for i in range(4):
            total[i] += s[i]

    rdir = os.path.join(args.out if not args.dry_run else args.work, "_report")
    os.makedirs(rdir, exist_ok=True)
    if not args.dry_run:
        shutil.copy(os.path.join(args.root, "datamix.txt"), os.path.join(args.out, "datamix.txt"))
    with open(os.path.join(rdir, "sources.tsv"), "w") as f:
        f.write("source\tshards\tdocs\ttokens\tdocs_dropped\ttokens_dropped\tdoc_pct\ttoken_pct\n")
        for k, (d, t, dd, td, ns) in sorted(by_src.items(), key=lambda x: -x[1][3]):
            f.write(f"{k}\t{ns}\t{d}\t{t}\t{dd}\t{td}\t{100 * dd / max(d, 1):.4f}\t{100 * td / max(t, 1):.4f}\n")
    with open(os.path.join(rdir, "benchmarks.tsv"), "w") as f:
        f.write("benchmark_source\tsegments_in_table\tsegments_found\tdoc_segment_pairs\n")
        seg_count = np.bincount(segs["src"], weights=(n_grams >= args.min_hits), minlength=len(src_names))
        for k, (ss, nd) in sorted(by_bench.items(), key=lambda x: -x[1][1]):
            f.write(f"{k}\t{int(seg_count[src_names.index(k)])}\t{len(ss)}\t{nd}\n")
    with open(os.path.join(rdir, "top_segments.tsv"), "w") as f:
        f.write("segment\tbenchmark_source\tdocs\tn_grams\ttext\n")
        for sg in sorted(seg_docs, key=lambda s: -seg_docs[s])[:300]:
            f.write(f"{sg}\t{src_names[segs['src'][sg]]}\t{seg_docs[sg]}\t{n_grams[sg]}\t{refs_text.get(sg, '')}\n")
    with open(os.path.join(rdir, "generic_segments.tsv"), "w") as f:
        f.write("segment\tbenchmark_source\tdocs\ttext\n")
        for sg in sorted(generic, key=lambda s: -seg_docs[s]):
            f.write(f"{sg}\t{src_names[segs['src'][sg]]}\t{seg_docs[sg]}\t{refs_text.get(sg, '')}\n")
    report = {"frac": args.frac, "min_hits": args.min_hits, "max_docs": args.max_docs,
              "shards": len(rows), "docs": total[0], "tokens": total[1], "docs_dropped": total[2],
              "tokens_dropped": total[3], "generic_segments_skipped": len(generic),
              "segments_found": len(seg_docs), "dry_run": args.dry_run}
    with open(os.path.join(rdir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print(open(os.path.join(rdir, "sources.tsv")).read())


if __name__ == "__main__":
    main()
