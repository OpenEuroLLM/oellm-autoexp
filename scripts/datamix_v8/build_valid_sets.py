"""Fixed held-out validation sets for the v8 + multilingual mix: one shard per set (Megatron
--fixed-valid-data-path needs one unweighted prefix per set), exactly one batch (2^26 tokens) each -- six 4k
sets (16,384 x 4,096) and long_64k (1,024 x 65,536, documents >= 32k tokens) -- evaluated completely at every
evaluation (--fixed-valid-seq-lengths 4096 x 6, 65536).

    python build_valid_sets.py --datamix datamix_v8ml_stage3_mn5.txt --report datamix_v8ml_mn5_report.tsv \\
        --out <data>/v8ml-valid

Held-out documents only: documents entirely inside the validation part of every stage-3 shard view under
--split 998,2,0 -- Megatron splits by sequence: sequences [round(0.998 S), S) -- which no stage trains on (stage 3 holds them
out, stages 1 / 2 hold other documents). Sets (category = the report's category column):
  all           every source            web_en        category web          multilingual  category multilingual
  code          category code           math          category math         reasoning     mv1-instruct + other
  long_64k      every source, documents >= 32k tokens only, at 64k
Within a set every source gets tokens in proportion to its mix weight; a source with fewer held-out tokens
gives all it has and the rest goes to the others. Documents whole (all their sequences), shuffled (seeded).
Writes <out>/<set>_text_document.{bin,idx}, <set>.info.json (per-source tokens / documents),
<set>.selection.npz (stage-3 view prefix index, document) and valid_sets.txt ("name seq_length prefix").
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flag_decontam"))
from common import Index, write_index  # noqa: E402

# name -> (categories (None = all), sequence length, minimum document tokens)
SETS = {"all": (None, 4096, 0), "web_en": ({"web"}, 4096, 0), "multilingual": ({"multilingual"}, 4096, 0),
        "code": ({"code"}, 4096, 0), "math": ({"math"}, 4096, 0),
        "reasoning": ({"mv1-instruct", "other"}, 4096, 0),
        "long_64k": (None, 65536, 32768)}           # long documents only, evaluated at 64k


def read_datamix(path):
    out = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            w, p = line.split()[:2]
            out.append((float(w), p))
    return out


def read_report(path):
    """(dir, source, category), longest dir first."""
    rows = []
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if f[0] != "category":
            rows.append((f[6].rstrip("/") + "/", f[1], f[0]))
    return sorted(rows, key=lambda r: -len(r[0]))


def stage_view_dir(dir_, stage_root, data_root):
    """A source's dir in the stage-3 views (v8ml-stages/s3/<path relative to the data root>)."""
    return os.path.join(stage_root, os.path.relpath(dir_, data_root)).rstrip("/") + "/"


def heldout_docs(idx, split=(0.998, 1.0)):
    """Documents entirely inside Megatron's validation split. GPTDataset splits by SEQUENCE: sequences
    round(0.998 S) .. S (S sequences); a document whose sequences straddle the boundary is left out."""
    beg = int(round(split[0] * idx.seq_count))
    first = np.asarray(idx.doc_idx[:-1], np.int64)
    return np.flatnonzero(first >= beg).astype(np.int64)


def doc_tokens(idx, docs):
    lens = idx.lengths.astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(lens)])
    d0 = np.asarray(idx.doc_idx, np.int64)
    return cum[d0[docs + 1]] - cum[d0[docs]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datamix", required=True, help="the stage-3 datamix (v8ml-stages/s3 views)")
    ap.add_argument("--report", required=True, help="build_datamix.py's report of the full v8ml mix")
    ap.add_argument("--stage-root", default="/gpfs/projects/ehpc533/users/korbip/data/v8ml-stages/s3")
    ap.add_argument("--data-root", default="/gpfs/projects/ehpc533/users/korbip/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-length", type=int, default=0, help="override every set's length (tests)")
    ap.add_argument("--batch-tokens", type=int, default=1 << 26, help="tokens per set (one batch)")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    rows = read_datamix(a.datamix)
    report = [(stage_view_dir(d, a.stage_root, a.data_root), s, c) for d, s, c in read_report(a.report)]
    src_w, src_cat, shards = {}, {}, {}
    for w, p in rows:
        hit = next(((s, c) for d, s, c in report if p.startswith(d)), None)
        assert hit is not None, f"no source for {p}"
        s, c = hit
        src_w[s] = src_w.get(s, 0.0) + w
        src_cat[s] = c
        shards.setdefault(s, []).append(p)
    # held-out pool per source: (shard number, document, tokens)
    prefixes = [p for _, p in rows]
    pid = {p: i for i, p in enumerate(prefixes)}
    pool, idx_cache = {}, {}
    for s, ps in shards.items():
        items = []
        for p in ps:
            idx = idx_cache.setdefault(p, Index(p + ".idx"))
            docs = heldout_docs(idx)
            items.append(np.stack([np.full(len(docs), pid[p]), docs, doc_tokens(idx, docs)], 1))
        pool[s] = np.concatenate(items) if items else np.zeros((0, 3), np.int64)
        np.random.default_rng([a.seed, len(s)] + [ord(ch) for ch in s]).shuffle(pool[s])
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for name in a.sets.split(","):
        cats, seq_length, min_doc = SETS[name]
        seq_length = a.seq_length or seq_length
        min_doc = min(min_doc, seq_length // 2) if a.seq_length else min_doc
        target = (a.batch_tokens // seq_length) * seq_length + 1   # a sample spans seq_length + 1 tokens
        srcs = [s for s in src_w if cats is None or src_cat[s] in cats]
        spool = {s: pool[s][pool[s][:, 2] >= min_doc] for s in srcs}
        avail = {s: int(spool[s][:, 2].sum()) for s in srcs}
        quota = {}
        left, open_ = target, [s for s in srcs if avail[s] > 0]
        while left > 0 and open_:                            # proportional, capped by what a source holds
            wsum = sum(src_w[s] for s in open_)
            give = {s: min(avail[s] - quota.get(s, 0), int(np.ceil(left * src_w[s] / wsum))) for s in open_}
            for s, g in give.items():
                quota[s] = quota.get(s, 0) + g
            left = target - sum(quota.values())
            open_ = [s for s in open_ if quota[s] < avail[s]]
        assert left <= 0, f"{name}: only {target - left} held-out tokens for {target}"
        picked, per_src = [], {}
        for s in srcs:
            if not quota.get(s):
                continue
            cum = np.cumsum(spool[s][:, 2])
            k = int(np.searchsorted(cum, quota[s])) + 1      # whole documents, the last one may overshoot
            sel = spool[s][:k]
            picked.append(sel)
            per_src[s] = {"documents": int(len(sel)), "tokens": int(sel[:, 2].sum()), "weight": src_w[s]}
        sel = np.concatenate(picked)
        np.random.default_rng([a.seed, 7] + [ord(ch) for ch in name]).shuffle(sel)
        prefix = os.path.join(a.out, f"{name}_text_document")
        seq_lens, doc_starts = [], [0]
        with open(prefix + ".bin.tmp", "wb") as fo:
            for p_i, d, n in sel:                            # all sequences of the document, in order
                idx = idx_cache[prefixes[p_i]]
                got = 0
                for q in range(int(idx.doc_idx[d]), int(idx.doc_idx[d + 1])):
                    t = np.fromfile(prefixes[p_i] + ".bin", np.int32, count=int(idx.lengths[q]),
                                    offset=int(idx.pointers[q]))
                    fo.write(t.tobytes())
                    seq_lens.append(len(t))
                    got += len(t)
                assert got == n
                doc_starts.append(len(seq_lens))
        os.replace(prefix + ".bin.tmp", prefix + ".bin")
        lengths = np.asarray(seq_lens, np.int32)
        pointers = np.concatenate([[0], np.cumsum(lengths[:-1].astype(np.int64) * 4)])
        write_index(prefix + ".idx", 4, lengths, pointers, np.asarray(doc_starts, np.int64))
        np.savez(os.path.join(a.out, f"{name}.selection.npz"), shard=sel[:, 0], doc=sel[:, 1], tokens=sel[:, 2],
                 prefixes=np.array(prefixes))
        info = {"set": name, "categories": sorted(cats) if cats else "all", "documents": int(len(sel)),
                "tokens": int(lengths.astype(np.int64).sum()), "target_tokens": target,
                "min_document_tokens": min_doc,
                "sequences_at": {"seq_length": seq_length,
                                 "samples": int((lengths.astype(np.int64).sum() - 1) // seq_length)},
                "heldout": "documents within sequences [round(0.998 S), S) of the stage-3 views (Megatron --split 998,2,0)",
                "sources": per_src}
        json.dump(info, open(os.path.join(a.out, f"{name}.info.json"), "w"), indent=1)
        lines.append(f"{name} {seq_length} {prefix}")
        print(f"{name}: {len(per_src)} sources, {len(sel)} documents, {info['tokens'] / 1e6:.1f}M tokens, "
              f"{info['sequences_at']['samples']} samples at {seq_length}", flush=True)
    vs = os.path.join(a.out, "valid_sets.txt")             # "name seq_length prefix"; merged with earlier runs
    old_lines = {l.split()[0]: l.strip() for l in open(vs)} if os.path.exists(vs) else {}
    old_lines.update({l.split()[0]: l for l in lines})
    order = [n for n in SETS if n in old_lines] + [n for n in old_lines if n not in SETS]
    with open(vs, "w") as f:
        f.write("\n".join(old_lines[n] for n in order) + "\n")


if __name__ == "__main__":
    main()
