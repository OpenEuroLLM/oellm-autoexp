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
--ml <flag-ml file> --en-share 0.8: the v8 table becomes 80 % and the multilingual part (flag-ml, as in flag-option4 / 5)
the other 20 % (report rows "multilingual", per collection source).
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
LUMI_FLAG = "/scratch/project_465002530/training/collection/flag/"
OVERRIDE = {"common-pile-stackv2-0.1": f"{V8}/hftok/common-pile-stackv2-0.1-full/megatron-lm",
            "dolma-3-math": f"{V8}/vtok/dolmino-math/megatron-lm"}
# v8.1: the full HF versions replace partial LUMI data_tok copies (tokenize_gt.py on MareNostrum, 2026-10-31)
OVERRIDE.update({n: f"{V8}/hftok/{n}/megatron-lm" for n in [
    "hermes-agent-reasoning-traces", "Themis-CodePreference", "openthoughts-4-code-qwen3-32b-32k-annotated",
    "Nemotron-Cascade-SFT-Stage-1", "Nemotron-Math-v2"]})


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
    ap.add_argument("--ml", default="", help="multilingual part: 'weight LUMI-path' lines (docs/datamix_v8/flag-ml_lumi.txt), "
                    "mapped to flag-decon-v2-agenttok; their weights are rescaled to 1 - --en-share")
    ap.add_argument("--en-share", type=float, default=1.0, help="share of the v8 (English) table when --ml is given, e.g. 0.8")
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
        total, ratio = sum(tok), float(r["ratio_pct"]) * (a.en_share if a.ml else 1.0)
        lines += [f"{ratio * t / total:.10f} {p}" for p, t in zip(prefixes, tok) if t]
        rep.append((r["category"], r["name"], ratio, total, len(prefixes), ratio / 100 * a.total_tokens / total, d))
        print(f"{r['name']:50s} {ratio:8.4f} % {total / 1e9:9.2f}B {len(prefixes):4d} shards "
              f"{rep[-1][5]:6.2f} epochs", flush=True)
    if a.ml:                                  # the multilingual part, its own relative weights, scaled to 1 - en_share
        ml = [(float(w), p.rstrip("/")) for w, p in (l.split()[:2] for l in open(a.ml) if l.strip() and not l.startswith("#"))]
        wsum = sum(w for w, _ in ml)
        by_src = {}
        for w, p in ml:
            assert p.startswith(LUMI_FLAG), p
            q = os.path.join(FLAG, p[len(LUMI_FLAG):])
            if not os.path.exists(q + ".idx"):
                missing.append(("ml " + p, q))
                continue
            share = 100 * (1 - a.en_share) * w / wsum
            lines.append(f"{share:.10f} {q}")
            src = p[len(LUMI_FLAG):].split("/")[0]
            b = by_src.setdefault(src, [0.0, 0, 0])
            b[0] += share; b[1] += tokens(q); b[2] += 1
        for src, (share, t, n) in sorted(by_src.items(), key=lambda x: -x[1][0]):
            rep.append(("multilingual", src, share, t, n, share / 100 * a.total_tokens / t, f"{FLAG}/{src}"))
            print(f"ml {src:47s} {share:8.4f} % {t / 1e9:9.2f}B {n:4d} shards {rep[-1][5]:6.2f} epochs", flush=True)
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
