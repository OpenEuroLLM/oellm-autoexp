"""Decide which documents to drop and write the decontaminated tree (docs/flag_post_decontamination.md).

    python scripts/flag_decontam/finalize.py --work $WORK --out $OUT [--frac 0.7] [--abs-run 200]
    python scripts/flag_decontam/finalize.py --work $WORK --dry-run --frac 0.7,0.8,0.9,1.0

A (sequence, segment) pair from scan.py is contamination when its longest verbatim run (in
canonical tokens) covers >= --frac of the segment, or is >= --abs-run tokens long (a long 1:1
copy of part of a passage), and in any case >= --min-run tokens. Rephrased or modified items
(a changed word splits the run) are kept on purpose. A document is dropped when any of its
sequences has such a pair. Segments that would drop more than --max-docs
documents over the whole mix are reported and not applied (generic text that survived pruning:
check them in the report before raising the limit).

--dry-run writes only the reports, to $WORK/_report/frac_<f>/ (several comma-separated --frac
values in one pass); otherwise one --frac and the tree $OUT, which mirrors the source root for
every shard of <root>/datamix.txt:
  <prefix>.bin        symlink to the original .bin (tokens are never copied)
  <prefix>.idx        new index of the kept documents (original pointers into the same .bin),
                      or a symlink to the original .idx when nothing is dropped
  <shard>.info.json   the original's, with the new .idx size / md5 / counts
  <prefix>.removed.npz   dropped documents: doc, seq, seg, run, windows, runs, n_tokens (per flagged pair)
plus datamix.txt (unchanged weights) and _report/{report.json, sources.tsv, benchmarks.tsv,
benchmarks_all.tsv, top_segments.tsv, generic_segments.tsv}. benchmarks.tsv credits a segment to the
one source extraction kept; benchmarks_all.tsv (when refs.attrib.jsonl.gz exists next to the refs)
to every benchmark that holds the exact text.
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


def load_hits(work, key, seq_count, hits_dir="hits"):
    files = sorted(glob.glob(os.path.join(work, hits_dir, key, "*.npz")))
    hits, covered = [], np.zeros(seq_count + 1, dtype=np.int64)   # hits: seq, seg, run, windows, runs
    for f in files:
        z = np.load(f)
        s0, s1 = z["seq_range"]
        covered[s0] += 1
        covered[s1] -= 1
        hits.append(z["hits"])
    cov = np.cumsum(covered)[:-1]
    complete = bool(seq_count == 0 or (cov == 1).all())
    h = np.concatenate(hits) if hits else np.zeros((0, 5), np.int64)
    return h, complete


def load_fixes(work, fixes):
    """{shard key: (seq, seg) -> (run, windows, runs)} from recheck.py's corrections file, or {}."""
    if not fixes:
        return {}
    z = np.load(os.path.join(work, fixes))
    out = defaultdict(dict)
    for k, seq, seg, run, w, nr in z["fixes"]:
        out[str(z["keys"][k])][(int(seq), int(seg))] = (int(run), int(w), int(nr))
    return out


def apply_fixes(hits, fix, min_run):
    if not fix or not len(hits):
        return hits
    h = hits.copy()
    keep = np.ones(len(h), bool)
    for i in range(len(h)):
        v = fix.get((int(h[i, 0]), int(h[i, 1])))
        if v is not None:
            h[i, 2:5] = v
            keep[i] = v[0] >= min_run
    return h[keep]


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


def doc_tokens(idx, docs):
    """Tokens of the given documents (reads only their entries of the index)."""
    s0, s1 = idx.doc_idx[docs], idx.doc_idx[docs + 1]
    if np.all(s1 - s0 == 1):
        return idx.lengths[s0].astype(np.int64)
    return np.array([int(idx.lengths[a:b].astype(np.int64).sum()) for a, b in zip(s0, s1)], np.int64)


def shard_totals(root, rel, idx):
    info_p = os.path.join(root, rel).removesuffix("_text_document") + ".info.json"
    if os.path.exists(info_p):
        t = json.load(open(info_p)).get("idx", {}).get("total_tokens")
        if t is not None:
            return int(t)
    return int(idx.lengths.astype(np.int64).sum())


def load_attrib(refs_path, n_segments):
    """Every included source of each segment (refs.attrib.jsonl.gz from extract_refs.py --attribute), or None."""
    p = refs_path.replace(".jsonl.gz", ".attrib.jsonl.gz")
    if not os.path.exists(p):
        return None
    out = []
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            out.append(sorted({a[0] for a in json.loads(line)["also_in"] if a[3]}))
    assert len(out) == n_segments, f"{p}: {len(out)} lines for {n_segments} segments"
    return out


ROLE_ORDER = ["answer", "tests", "question", "options", "context", "other"]   # most severe first
_ROLE_RULES = [  # (role, substrings of the field path, lower-cased) -- first match wins
    ("answer", ["canonical_solution", "solution", "answer", "target", "completion", "output", "label_text",
                "rationale", "explanation", "final", "gold", "response"]),
    ("tests", ["test", "assert", "check"]),
    ("question", ["question", "problem", "prompt", "query", "goal", "instruction", "input", "sentence",
                  "premise", "ctx", "activity_label", "task", "description", "statement", "claim"]),
    ("options", ["choice", "option", "ending", "endings", "mc", "distractor", "candidates", "sol1", "sol2"]),
    ("context", ["context", "passage", "story", "article", "text", "document", "paragraph", "background",
                 "abstract", "flores", "sentence_", "src", "tgt", "source", "translation"]),
]


def role_of(path: str) -> str:
    p = path.lower()
    for role, keys in _ROLE_RULES:
        if any(k in p for k in keys):
            return role
    return "other"


def load_roles(refs_path, n_segments):
    """Most severe role of each segment over all its fields (refs.attrib.jsonl.gz with field paths), or None."""
    p = refs_path.replace(".jsonl.gz", ".attrib.jsonl.gz")
    if not os.path.exists(p):
        return None
    rank = {r: i for i, r in enumerate(ROLE_ORDER)}
    out = np.full(n_segments, len(ROLE_ORDER) - 1, np.int8)
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            paths = [a[5] for a in json.loads(line)["also_in"] if a[3] and len(a) > 5]
            if paths:
                out[i] = min(rank[role_of(x)] for x in paths)
    return out


def decide(shards, segs, frac, args):
    """Per shard (flagged pairs, their docs); the generic segments left out; docs per segment."""
    n_tok = segs["n_tokens"].astype(np.int64)
    need = np.maximum(args.min_run, np.minimum(np.ceil(frac * n_tok), args.abs_run)).astype(np.int64)
    need[segs["n"] == 0] = np.iinfo(np.int64).max         # unmatchable segments never flag
    S, max_docs = len(n_tok), args.max_docs
    flagged, seg_docs = {}, np.zeros(S, np.int64)
    for rel, (idx, hits, seq2doc) in shards.items():
        sel = hits[:, 2] >= need[hits[:, 1]] if len(hits) else np.zeros(0, bool)
        f, d = hits[sel], seq2doc[sel]
        flagged[rel] = (f, d)
        if len(f):
            pairs = np.unique(d * S + f[:, 1])                 # distinct (doc, segment)
            seg_docs += np.bincount(pairs % S, minlength=S)
    generic = np.flatnonzero(seg_docs > max_docs)
    if len(generic):
        for rel, (f, d) in flagged.items():
            ok = ~np.isin(f[:, 1], generic)
            flagged[rel] = (f[ok], d[ok])
    return flagged, set(generic.tolist()), seg_docs


def report(rdir, frac, args, rows, shards, totals, flagged, generic, seg_docs, segs, src_names, refs_text, attrib=None,
           roles=None):
    os.makedirs(rdir, exist_ok=True)
    S, nsrc = len(segs["n"]), len(src_names)
    n_tok = segs["n_tokens"]
    by_src = defaultdict(lambda: [0, 0, 0, 0, 0])        # docs, tokens, docs dropped, tokens dropped, shards
    pair_per_src = np.zeros(nsrc, np.int64)               # (doc, segment) pairs per benchmark source
    found = np.zeros(S, bool)
    seg_pairs = np.zeros(S, np.int64)
    for _, rel in rows:
        idx = shards[rel][0]
        f, d = flagged[rel]
        dropped = np.unique(d)
        s = by_src[rel.split("/")[0]]
        s[0] += idx.doc_count - 1
        s[1] += totals[rel]
        s[2] += len(dropped)
        s[3] += int(doc_tokens(idx, dropped).sum()) if len(dropped) else 0
        s[4] += 1
        if len(f):
            seg = np.unique(d * S + f[:, 1]) % S
            found[seg] = True
            seg_pairs += np.bincount(seg, minlength=S)
            pair_per_src += np.bincount(segs["src"][seg], minlength=nsrc)
    total = [sum(v[i] for v in by_src.values()) for i in range(4)]
    if roles is not None:     # dropped docs / tokens by the most severe role among a doc's flagged segments
        by_role = np.zeros((len(ROLE_ORDER), 2), np.int64)
        for _, rel in rows:
            idx = shards[rel][0]
            f, d = flagged[rel]
            if not len(f):
                continue
            r = roles[f[:, 1]]
            order = np.lexsort((r, d))
            dd, rr = d[order], r[order]
            first = np.ones(len(dd), bool)
            first[1:] = dd[1:] != dd[:-1]
            docs, rmin = dd[first], rr[first]
            tok = doc_tokens(idx, docs)
            np.add.at(by_role[:, 0], rmin, 1)
            np.add.at(by_role[:, 1], rmin, tok)
        with open(os.path.join(rdir, "roles.tsv"), "w") as fh:
            fh.write("role\tdocs_dropped\ttokens_dropped\n")
            for i, r in enumerate(ROLE_ORDER):
                fh.write(f"{r}\t{by_role[i, 0]}\t{by_role[i, 1]}\n")
    with open(os.path.join(rdir, "sources.tsv"), "w") as fh:
        fh.write("source\tshards\tdocs\ttokens\tdocs_dropped\ttokens_dropped\tdoc_pct\ttoken_pct\n")
        for k, (dd, t, ddr, tdr, ns) in sorted(by_src.items(), key=lambda x: -x[1][3]):
            fh.write(f"{k}\t{ns}\t{dd}\t{t}\t{ddr}\t{tdr}\t{100 * ddr / max(dd, 1):.4f}\t{100 * tdr / max(t, 1):.4f}\n")
    matchable = np.bincount(segs["src"], weights=(segs["n"] > 0), minlength=nsrc)
    seg_found = np.bincount(segs["src"][found], minlength=nsrc)
    with open(os.path.join(rdir, "benchmarks.tsv"), "w") as fh:
        fh.write("benchmark_source\tsegments_matchable\tsegments_found\tpct_found\tdoc_segment_pairs\n")
        for k in np.argsort(-pair_per_src):
            if pair_per_src[k] == 0:
                break
            m = int(matchable[k])
            fh.write(f"{src_names[k]}\t{m}\t{int(seg_found[k])}\t{100 * seg_found[k] / max(m, 1):.2f}\t{int(pair_per_src[k])}\n")
    if attrib is not None:   # every benchmark holding the exact text (MGSM-en questions are GSM8K test questions)
        allm, allf, allp = defaultdict(int), defaultdict(int), defaultdict(int)
        for sg in np.flatnonzero(segs["n"] > 0):
            for a in attrib[sg]:
                allm[a] += 1
        for sg in np.flatnonzero(found):
            for a in attrib[sg]:
                allf[a] += 1
                allp[a] += int(seg_pairs[sg])
        with open(os.path.join(rdir, "benchmarks_all.tsv"), "w") as fh:
            fh.write("benchmark_source\tsegments_matchable\tsegments_found\tpct_found\tdoc_segment_pairs\n")
            for k in sorted(allp, key=lambda x: -allp[x]):
                fh.write(f"{k}\t{allm[k]}\t{allf[k]}\t{100 * allf[k] / max(allm[k], 1):.2f}\t{allp[k]}\n")
    applied = seg_docs.copy()
    applied[list(generic)] = 0
    top = np.argsort(-applied)[:300]
    with open(os.path.join(rdir, "top_segments.tsv"), "w") as fh:
        fh.write("segment\tbenchmark_source\tdocs\tn_tokens\ttext\n")
        for sg in top[applied[top] > 0]:
            fh.write(f"{sg}\t{src_names[segs['src'][sg]]}\t{applied[sg]}\t{n_tok[sg]}\t{refs_text.get(int(sg), '')}\n")
    with open(os.path.join(rdir, "generic_segments.tsv"), "w") as fh:
        fh.write("segment\tbenchmark_source\tdocs\ttext\n")
        for sg in sorted(generic, key=lambda x: -seg_docs[x]):
            fh.write(f"{sg}\t{src_names[segs['src'][sg]]}\t{seg_docs[sg]}\t{refs_text.get(sg, '')}\n")
    rep = {"frac": frac, "abs_run": args.abs_run, "min_run": args.min_run, "max_docs": args.max_docs, "shards": len(rows),
           "docs": total[0], "tokens": total[1], "docs_dropped": total[2], "tokens_dropped": total[3],
           "token_pct_dropped": 100 * total[3] / max(total[1], 1), "segments_found": int((applied > 0).sum()),
           "generic_segments_skipped": len(generic), "dry_run": args.dry_run}
    with open(os.path.join(rdir, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep), flush=True)


def write_tree(args, rows, shards, flagged, segs):
    for _, rel in rows:
        idx = shards[rel][0]
        f, d = flagged[rel]
        drop = np.zeros(idx.doc_count - 1, dtype=bool)
        drop[d] = True
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
            np.savez(dst_p + ".removed.npz", doc=d, seq=f[:, 0], seg=f[:, 1], run=f[:, 2], windows=f[:, 3], runs=f[:, 4],
                     n_tokens=segs["n_tokens"][f[:, 1]])
            dropped = np.flatnonzero(drop)
            info["idx"].update(size=os.path.getsize(dst_p + ".idx"), md5sum=md5(dst_p + ".idx"),
                               sequence_count=int(len(lengths)), document_count=int(len(doc_idx)),
                               total_tokens=int(lengths.astype(np.int64).sum()))
            info["decontamination"] = {"dropped_documents": int(len(dropped)),
                                       "dropped_tokens": int(doc_tokens(idx, dropped).sum()),
                                       "source_idx": src_p + ".idx", "frac": args.frac[0], "abs_run": args.abs_run,
                                       "min_run": args.min_run, "max_docs": args.max_docs}
        else:
            if os.path.exists(dst_p + ".removed.npz"):
                os.remove(dst_p + ".removed.npz")
            link(src_p + ".idx", dst_p + ".idx")
        with open(dst_p.removesuffix("_text_document") + ".info.json", "w") as fh:
            json.dump(info, fh, indent=2)
        print(f"{'rewrote' if drop.any() else 'linked '} {rel}: -{int(drop.sum())} docs", flush=True)
    shutil.copy(os.path.join(args.root, "datamix.txt"), os.path.join(args.out, "datamix.txt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--root", default="/gpfs/projects/ehpc533/training-data-collection/flag")
    ap.add_argument("--out", default="/gpfs/projects/ehpc533/training-data-collection/flag-decon-v1")
    ap.add_argument("--frac", type=lambda s: [float(x) for x in s.split(",")], default=[0.7],   # user 2026-10-24
                    help="longest run / segment tokens (comma list with --dry-run)")
    ap.add_argument("--abs-run", type=int, default=200, help="a run this long flags whatever the segment length")
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--max-docs", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true", help="reports only ($WORK/_report/frac_<f>), no tree")
    ap.add_argument("--hits-dir", default="hits")
    ap.add_argument("--fixes", default="", help="recheck.py's corrections (recheck_fixes.npz), applied to the hits")
    args = ap.parse_args()
    if not args.dry_run and len(args.frac) != 1:
        sys.exit("writing the tree takes exactly one --frac")

    tdir = os.path.join(args.work, "table")
    segs = np.load(os.path.join(tdir, "segments.npz"))
    meta = json.load(open(os.path.join(tdir, "sources.json")))
    src_names = meta["sources"]
    rows = []
    for line in open(os.path.join(args.root, "datamix.txt")):
        if line.strip():
            w, prefix = line.split()
            rows.append((float(w), prefix))

    fixes = load_fixes(args.work, args.fixes)
    shards, totals, incomplete = {}, {}, []
    for _, rel in rows:
        p = os.path.join(args.root, rel)
        idx = Index(p + ".idx")
        assert idx.multimodal_tail == 0, f"multimodal index {p}"
        key = shard_key(p, args.root)
        hits, complete = load_hits(args.work, key, idx.seq_count, args.hits_dir)
        hits = apply_fixes(hits, fixes.get(key), args.min_run)
        if not complete:
            incomplete.append(rel)
        seq2doc = np.searchsorted(idx.doc_idx, hits[:, 0], "right") - 1
        shards[rel] = (idx, hits, seq2doc)
        totals[rel] = shard_totals(args.root, rel, idx)
    if incomplete:
        sys.exit(f"scan incomplete for {len(incomplete)} shards, e.g. {incomplete[:3]}: rerun scan.py --mode scan")
    print(f"hits loaded: {sum(len(v[1]) for v in shards.values())} (sequence, segment) pairs", flush=True)

    attrib = load_attrib(meta["refs"], len(segs["n"]))
    roles = load_roles(meta["refs"], len(segs["n"]))
    decisions = {fr: decide(shards, segs, fr, args) for fr in args.frac}
    want = set()
    for _, generic, seg_docs in decisions.values():
        want |= set(np.argsort(-seg_docs)[:300].tolist()) | generic
    refs_text = {}
    with gzip.open(meta["refs"], "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in want:
                refs_text[i] = json.loads(line)["text"][:300].replace("\t", " ").replace("\n", " ")

    for fr, (flagged, generic, seg_docs) in decisions.items():
        rdir = (os.path.join(args.work, "_report" + ("_fixed" if args.fixes else ""), f"frac_{fr:g}")
                if args.dry_run else os.path.join(args.out, "_report"))
        report(rdir, fr, args, rows, shards, totals, flagged, generic, seg_docs, segs, src_names, refs_text, attrib,
               roles)
        print(open(os.path.join(rdir, "sources.tsv")).read(), flush=True)
    if not args.dry_run:
        write_tree(args, rows, shards, decisions[args.frac[0]][0], segs)


if __name__ == "__main__":
    main()
