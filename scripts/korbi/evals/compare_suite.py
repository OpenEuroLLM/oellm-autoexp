#!/usr/bin/env python3
"""Compare a full-suite eval directory against the 32B-dense and Qwen references.

Sources
  ours        <eval dir>/summary_shard*.tsv        (row, task, metric, value, ...)
  qwen        <ext dir>/summary_shard*.tsv         same format, one dir per model
  32B dense   JUPITER b1cons_90k / b1consTW_300k .flag-evals-438.tasks.csv

Per task one metric is chosen from a preference list, because a task reports several
(acc / acc_norm / exact_match / f1 / stderr). The filter suffix after the comma is dropped
("acc,none" -> "acc").

    python3 scripts/korbi/evals/compare_suite.py --ours <dir> [--min-common 1]

Prints one row per task present in `ours`, with the references that have it, then a summary
(wins / losses against each reference and the mean delta). Tasks a reference lacks print "–".
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
from collections import defaultdict

EVAL_ROOT = "/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm"
HOST = "marenostrum_new"
JUPITER = "jupiter"
JUPITER_DIR = "/e/scratch/e-sta-openeurollm/poeppel1/flag-evals/results"
JUPITER_REFS = {
    "32B@90k": "b1cons_90k.flag-evals-438.tasks.csv",
    "32B@300k": "b1consTW_300k.flag-evals-438.tasks.csv",
}
QWEN_REFS = {
    "Qwen3-14B": "ext_Qwen3-14B-Base",
    "Qwen3-30B-A3B": "ext_Qwen3-30B-A3B-Base",
    "Qwen3.5-35B-A3B": "ext_Qwen3.5-35B-A3B-Base",
    "Qwen2.5-32B": "ext_Qwen2.5-32B",
}

# Higher is better for every entry; the first metric present per task wins.
METRIC_PREFERENCE = [
    "acc",
    "accuracy",
    "exact_match",
    "f1",
    "em",
    "pass@1",
    "acc_norm",
]


def base_metric(name: str) -> str:
    return name.split(",")[0].strip()


def pick(metrics: dict[str, float]) -> tuple[str, float] | None:
    for want in METRIC_PREFERENCE:
        for key, value in metrics.items():
            if key == want:
                return key, value
    # fall back to the first non-stderr metric
    for key, value in metrics.items():
        if "stderr" not in key:
            return key, value
    return None


def read_summaries(directory: str) -> dict[str, tuple[str, float]]:
    """task -> (metric, value) from summary_shard*.tsv files."""
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(directory, "summary_shard*.tsv"))):
        with open(path) as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                task, metric, value = parts[1], base_metric(parts[2]), parts[3]
                try:
                    per_task[task][metric] = float(value)
                except ValueError:
                    continue
    out = {}
    for task, metrics in per_task.items():
        chosen = pick(metrics)
        if chosen:
            out[task] = chosen
    return out


def read_jupiter() -> dict[str, dict[str, float]]:
    """ref -> task -> value, one ssh round trip."""
    script = ["import csv, sys"]
    script.append("refs = %r" % (JUPITER_REFS,))
    script.append("base = %r" % JUPITER_DIR)
    script.append("keep = set(%r)" % (METRIC_PREFERENCE,))
    script.append("for label, fname in refs.items():")
    script.append("    rows = list(csv.DictReader(open(base + '/' + fname)))")
    script.append("    best = {}")
    script.append("    for r in rows:")
    script.append("        metric = (r['metric'] or '').split(',')[0].strip()")
    script.append("        if metric not in keep:")
    script.append("            continue")
    script.append("        try:")
    script.append("            value = float(r['value'])")
    script.append("        except (TypeError, ValueError):")
    script.append("            continue")
    script.append("        task = r['task']")
    script.append("        rank = %r.index(metric)" % (METRIC_PREFERENCE,))
    script.append("        if task not in best or rank < best[task][0]:")
    script.append("            best[task] = (rank, value)")
    script.append("    for task, (rank, value) in best.items():")
    script.append("        print('%%s\\t%%s\\t%%s' %% (label, task, value))")
    program = "\n".join(script)
    out = subprocess.run(["ssh", JUPITER, "python3 -"], input=program, capture_output=True,
                         text=True, timeout=900, check=False)
    if out.returncode != 0:
        print(f"warning: jupiter exited {out.returncode}: {out.stderr[-200:]}", file=sys.stderr)
    refs: dict[str, dict[str, float]] = defaultdict(dict)
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            refs[parts[0]][parts[1]] = float(parts[2])
    return refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="our eval dir (summary_shard*.tsv)")
    ap.add_argument("--label", default="ours")
    ap.add_argument("--min-common", type=int, default=1, help="only tasks with at least this many references")
    ap.add_argument("--csv", default="", help="also write the table as CSV")
    args = ap.parse_args()

    ours_dir = args.ours if os.path.isdir(args.ours) else os.path.join(EVAL_ROOT, args.ours)
    ours = read_summaries(ours_dir)
    if not ours:
        print(f"no summary rows in {ours_dir}", file=sys.stderr)
        return 1

    refs: dict[str, dict[str, float]] = defaultdict(dict)
    for label, sub in QWEN_REFS.items():
        d = os.path.join(EVAL_ROOT, sub)
        if os.path.isdir(d):
            for task, (_metric, value) in read_summaries(d).items():
                refs[label][task] = value
    for label, values in read_jupiter().items():
        refs[label].update(values)

    columns = list(JUPITER_REFS) + list(QWEN_REFS)
    columns = [c for c in columns if c in refs]
    rows = []
    for task in sorted(ours):
        metric, value = ours[task]
        cells = [refs[c].get(task) for c in columns]
        if sum(1 for c in cells if c is not None) < args.min_common:
            continue
        rows.append((task, metric, value, cells))

    width = max([len(r[0]) for r in rows] + [4])
    header = f"{'task':{width}} {'metric':<12} {args.label:>7} " + " ".join(f"{c:>14}" for c in columns)
    print(header)
    print("-" * len(header))
    for task, metric, value, cells in rows:
        line = f"{task:{width}} {metric:<12} {value*100:7.2f} "
        line += " ".join("            –" if c is None else f"{c*100:14.2f}" for c in cells)
        print(line)

    print("\nsummary over %d tasks" % len(rows))
    for i, col in enumerate(columns):
        pairs = [(r[2], r[3][i]) for r in rows if r[3][i] is not None]
        if not pairs:
            continue
        wins = sum(1 for a, b in pairs if a > b)
        losses = sum(1 for a, b in pairs if a < b)
        delta = sum(a - b for a, b in pairs) / len(pairs) * 100
        print(f"  vs {col:16s} n={len(pairs):3d}  wins {wins:3d}  losses {losses:3d}  "
              f"mean delta {delta:+6.2f} pp")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["task", "metric", args.label] + columns)
            for task, metric, value, cells in rows:
                w.writerow([task, metric, f"{value:.6f}"] + ["" if c is None else f"{c:.6f}" for c in cells])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
