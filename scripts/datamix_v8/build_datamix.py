"""datamix v8 on MareNostrum 5 as a Megatron ``data_args_path`` file + a per-source report.

    python scripts/datamix_v8/build_datamix.py [--out $HYBRID_ENV/data/datamix_v8_mn5.txt]

Every row of docs/datamix_v8/datamix_v8.tsv (ratio_pct) maps to its shards in the final trees:
  jsc "flag/X"       -> <data>/flag-decon-v2-agenttok/X             (the flag sources, as in the current default)
  jsc "mv1sub/X"     -> <data>/v8-decon-agenttok/mv1sub/X/megatron-lm
  lumi "vtok/X"      -> <data>/v8-decon-agenttok/vtok/<X without .bin / _text_document / .dedup.clean>/megatron-lm
  hf only            -> <data>/v8-decon-agenttok/hftok/<name>/megatron-lm
  decisions (docs/datamix_v8/metadata.yaml): common-pile-stackv2-0.1 -> hftok/common-pile-stackv2-0.1-full (full
  propella-filtered stage), dolma-3-math -> vtok/dolmino-math (both ingredients).
A shard's weight is ratio_pct * shard tokens / source tokens (weights sum to 100); tokens from the shard's .idx.
The report (--report) has per source: ratio, tokens, shards, epochs at --total-tokens. Fails on a missing source.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "flag_decontam"))
from common import Index  # noqa: E402

DATA = "/gpfs/projects/ehpc533/users/korbip/data"
FLAG = f"{DATA}/flag-decon-v2-agenttok"
V8 = f"{DATA}/v8-decon-agenttok"
OVERRIDE = {"common-pile-stackv2-0.1": f"{V8}/hftok/common-pile-stackv2-0.1-full/megatron-lm",
            "dolma-3-math": f"{V8}/vtok/dolmino-math/megatron-lm"}


def source_dir(row):
    name, jsc, lumi = row["name"], row["jsc"], row["lumi"]
    if name in OVERRIDE:
        return OVERRIDE[name]
    if jsc.startswith("flag/"):
        return os.path.join(FLAG, jsc[len("flag/"):])
    if jsc.startswith("mv1sub/"):
        return f"{V8}/mv1sub/{jsc[len('mv1sub/'):]}/megatron-lm"
    if lumi.startswith("vtok/"):
        x = os.path.basename(lumi).removesuffix(".bin").removesuffix("_text_document").removesuffix(".dedup.clean")
        return f"{V8}/vtok/{x}/megatron-lm"
    return f"{V8}/hftok/{name}/megatron-lm"


def tokens(prefix):
    return int(Index(prefix + ".idx").lengths.astype(np.int64).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=os.path.join(HERE, "..", "..", "docs", "datamix_v8", "datamix_v8.tsv"))
    ap.add_argument("--out", default=os.path.join(os.environ.get("HYBRID_ENV", "/gpfs/projects/ehpc533/users/korbip/hybrid_v019"),
                                                   "data", "datamix_v8_mn5.txt"))
    ap.add_argument("--report", default="")
    ap.add_argument("--total-tokens", type=float, default=12e12)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader((l for l in open(a.table) if not l.startswith("#") and l.strip("\t\n")),
                                      delimiter="\t") if r.get("name") and r.get("ratio_pct")]
    lines, rep, missing = [], [], []
    for r in rows:
        d = source_dir(r)
        prefixes = sorted(p[:-4] for p in glob.glob(os.path.join(d, "**", "*_text_document.idx"), recursive=True))
        if not prefixes:
            missing.append((r["name"], d))
            continue
        tok = [tokens(p) for p in prefixes]
        total, ratio = sum(tok), float(r["ratio_pct"])
        lines += [f"{ratio * t / total:.10f} {p}" for p, t in zip(prefixes, tok) if t]
        rep.append((r["category"], r["name"], ratio, total, len(prefixes), ratio / 100 * a.total_tokens / total, d))
        print(f"{r['name']:50s} {ratio:8.4f} % {total / 1e9:9.2f}B {len(prefixes):4d} shards "
              f"{rep[-1][5]:6.2f} epochs", flush=True)
    if missing:
        sys.exit("missing sources:\n" + "\n".join(f"  {n}: {d}" for n, d in missing))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(a.out + ".tmp", a.out)
    print(f"{len(lines)} shards, {len(rows)} sources, weights sum {sum(float(l.split()[0]) for l in lines):.4f} -> {a.out}")
    if a.report:
        with open(a.report, "w") as f:
            f.write("category\tname\tratio_pct\ttokens_B\tshards\tepochs_at_total\tdir\n")
            for c, n, ratio, total, ns, ep, d in rep:
                f.write(f"{c}\t{n}\t{ratio:.4f}\t{total / 1e9:.2f}\t{ns}\t{ep:.3f}\t{d}\n")


if __name__ == "__main__":
    main()
