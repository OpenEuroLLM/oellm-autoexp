"""Split a datamix by document into sequence-length curriculum stages (e.g. 4k / 16k / 64k = 2 / 8 / 90 %), so every
stage has its own frozen dataset: the same per-source ratios, no document in two stages, one dataset index per stage
(a stage with another sequence length / schedule never reshuffles another stage's data).

    # per shard (split over nodes): one .idx view per stage, the .bin hard-linked (a symlink across filesystems)
    python split_stages.py split --datamix datamix_v8ml_mn5.txt --out <root> --fractions 0.02,0.08,0.90 --part i --nparts n
    # the stage datamix files: per source the same total weight, spread over its shards by the stage's tokens
    python split_stages.py datamix --datamix datamix_v8ml_mn5.txt --report datamix_v8ml_mn5_report.tsv --out <root> ...

A document's stage: a hash of (shard path relative to --data-root, document index) mapped to [0, 1), compared
with the cumulative fractions -- deterministic and independent of the node split. Output prefixes:
<out>/s<k>/<shard path relative to --data-root>; <prefix>.stage.json holds documents / tokens per stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flag_decontam"))
from common import Index, filter_index, write_index  # noqa: E402

DATA = "/gpfs/projects/ehpc533/users/korbip/data"


def read_datamix(path):
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            w, p = line.split()[:2]
            rows.append((float(w), p))
    return rows


def stage_of_docs(rel: str, n_docs: int, fractions) -> np.ndarray:
    """Stage index per document: splitmix64 of (crc32(rel) << 32) ^ doc, top 53 bits as a uniform in [0, 1)."""
    x = (np.uint64(zlib.crc32(rel.encode())) << np.uint64(32)) ^ np.arange(n_docs, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return np.searchsorted(np.cumsum(fractions)[:-1], u, side="right")


def link_bin(src, dst):
    real = os.path.realpath(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if os.path.exists(dst) and os.path.samefile(dst, real):
            return
        os.remove(dst)
    try:
        os.link(real, dst)
    except OSError:
        os.symlink(real, dst)


def split_shard(prefix, rel, out, fractions):
    meta_p = os.path.join(out, "s1", rel) + ".stage.json"
    if os.path.exists(meta_p):
        return json.load(open(meta_p))
    idx = Index(prefix + ".idx")
    stage = stage_of_docs(rel, idx.doc_count - 1, fractions)
    seq_tokens = idx.lengths.astype(np.int64)
    doc_tokens = np.add.reduceat(seq_tokens, idx.doc_idx[:-1]) if len(seq_tokens) else np.zeros(0, np.int64)
    meta = {"source": prefix, "fractions": list(fractions), "stages": []}
    for k in range(len(fractions)):
        keep = stage == k
        dst = os.path.join(out, f"s{k + 1}", rel)
        link_bin(prefix + ".bin", dst + ".bin")
        write_index(dst + ".idx", idx.dtype_code, *filter_index(idx, keep))
        meta["stages"].append({"documents": int(keep.sum()), "tokens": int(doc_tokens[keep].sum())})
    for k in range(len(fractions)):                 # the same record next to every stage's shard
        p = os.path.join(out, f"s{k + 1}", rel) + ".stage.json"
        with open(p + ".tmp", "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(p + ".tmp", p)
    return meta


def cmd_split(a):
    fr = [float(x) for x in a.fractions.split(",")]
    assert abs(sum(fr) - 1) < 1e-9, fr
    rows = read_datamix(a.datamix)
    mine = rows[a.part::a.nparts]
    for i, (_, p) in enumerate(mine):
        rel = os.path.relpath(p, a.data_root)
        assert not rel.startswith(".."), p
        m = split_shard(p, rel, a.out, fr)
        print(f"[split {a.part}] {i + 1}/{len(mine)} {rel}: " +
              " / ".join(f"{s['tokens'] / 1e9:.2f}B" for s in m["stages"]), flush=True)


def source_dirs(report):
    """(dir, source name) from build_datamix.py's report, longest first (a shard belongs to the longest prefix)."""
    out = []
    for line in open(report):
        f = line.rstrip("\n").split("\t")
        if f[0] == "category":
            continue
        out.append((f[6].rstrip("/") + "/", f[1]))
    return sorted(out, key=lambda x: -len(x[0]))


def cmd_datamix(a):
    fr = [float(x) for x in a.fractions.split(",")]
    rows = read_datamix(a.datamix)
    dirs = source_dirs(a.report)
    src_w, src_t = {}, {}
    per = []
    for w, p in rows:
        src = next((n for d, n in dirs if p.startswith(d)), None)
        assert src is not None, f"no source for {p}"
        rel = os.path.relpath(p, a.data_root)
        m = json.load(open(os.path.join(a.out, "s1", rel) + ".stage.json"))
        per.append((src, rel, m))
        src_w[src] = src_w.get(src, 0.0) + w
        for k, s in enumerate(m["stages"]):
            src_t[(src, k)] = src_t.get((src, k), 0) + s["tokens"]
    for k in range(len(fr)):
        lines = []
        for src, rel, m in per:
            t = m["stages"][k]["tokens"]
            if t:
                lines.append(f"{src_w[src] * t / src_t[(src, k)]:.10f} {os.path.join(a.out, f's{k + 1}', rel)}")
        out = a.datamix_out.replace("{k}", str(k + 1))
        with open(out + ".tmp", "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(out + ".tmp", out)
        tot = sum(t for (s, kk), t in src_t.items() if kk == k)
        print(f"stage {k + 1}: {len(lines)} shards, {tot / 1e9:.1f}B tokens, weight sum "
              f"{sum(float(l.split()[0]) for l in lines):.4f} -> {out}", flush=True)
    with open(os.path.join(a.out, "stages_report.tsv"), "w") as f:
        f.write("source\tweight_pct\t" + "\t".join(f"s{k + 1}_tokens_B" for k in range(len(fr))) + "\n")
        for src in sorted(src_w, key=lambda s: -src_w[s]):
            f.write(f"{src}\t{src_w[src]:.4f}\t" + "\t".join(f"{src_t[(src, k)] / 1e9:.3f}" for k in range(len(fr))) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["split", "datamix"])
    ap.add_argument("--datamix", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datamix-out", default="", help="stage datamix files, '{k}' = stage number")
    ap.add_argument("--data-root", default=DATA)
    ap.add_argument("--fractions", default="0.02,0.08,0.90")
    ap.add_argument("--part", type=int, default=int(os.environ.get("SLURM_NODEID", 0)))
    ap.add_argument("--nparts", type=int, default=int(os.environ.get("SLURM_NNODES", 1)))
    a = ap.parse_args()
    {"split": cmd_split, "datamix": cmd_datamix}[a.mode](a)


if __name__ == "__main__":
    main()
