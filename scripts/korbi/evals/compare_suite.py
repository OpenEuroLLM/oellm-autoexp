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
import json
import re
import glob
import os
import subprocess
import sys
from collections import defaultdict

EVAL_ROOT = "/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm"
# The JUPITER reference CSVs are copied to MN5 once (the MN5 login nodes cannot reach JUPITER):
#   ssh jupiter 'cat <csv>' > /tmp/x.csv && ssh marenostrum_new 'cat > $R/refs_jupiter/<csv>' < /tmp/x.csv
JUPITER_DIR = "/gpfs/projects/ehpc533/users/korbip/hybrid_v019/refs_jupiter"
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


def to_pct(value: float) -> float:
    """The JUPITER CSVs mix units (coqa f1 is a fraction, squadv2 f1 is already percent)."""
    return value * 100.0 if value <= 1.5 else value


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


def read_results(directory: str) -> dict[str, tuple[str, float]]:
    """task -> (metric, value) from results/*/samples_*.jsonl (works mid-run, before summaries).

    lm-eval writes the per-sample metric names in ``metrics`` and the values as top-level keys.
    """
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    for task_dir in sorted(glob.glob(os.path.join(directory, "results", "*"))):
        task = re.sub(r"_\d+$", "", os.path.basename(task_dir))
        files = glob.glob(os.path.join(task_dir, "**", "samples_*.jsonl"), recursive=True)
        if not files:
            continue
        totals: dict[str, float] = defaultdict(float)
        n = 0
        for path in files:
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    n += 1
                    for key in (row.get("metrics") or []):
                        value = row.get(key)
                        # squad-style tasks store a list of metric dicts, not a scalar; skip them here
                        # (their summary TSV has the aggregate once the session finishes)
                        if not isinstance(value, (int, float)):
                            continue
                        totals[key] += float(value)
        if n:
            for key, value in totals.items():
                per_task[task][base_metric(key)] = value / n
    out = {}
    for task, metrics in per_task.items():
        chosen = pick(metrics)
        if chosen:
            out[task] = chosen
    return out


def read_lighteval(directory: str) -> dict[str, tuple[str, float]]:
    """task -> (metric, value) from lighteval's results JSONs (they live outside results/).

    ``results["flores200:eng_Latn-bul_Cyrl|4"]`` maps metric names to numbers; the ``|n`` suffix is
    the few-shot count, not part of the task name.
    """
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    pattern = os.path.join(directory, "lighteval", "lighteval_group_*", "results", "**", "results_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            payload = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        for raw_task, metrics in (payload.get("results") or {}).items():
            task = raw_task.split("|")[0]
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                metric = base_metric(key)
                if "stderr" in metric:
                    continue
                try:
                    per_task[task][metric] = float(value)
                except (TypeError, ValueError):
                    continue
    out = {}
    for task, metrics in per_task.items():
        chosen = pick(metrics)
        if chosen:
            out[task] = chosen
    return out


def read_evalchemy(directory: str) -> dict[str, tuple[str, float]]:
    """task -> (metric, value) from evalchemy's per-task result JSONs (also outside results/)."""
    per_task: dict[str, dict[str, float]] = defaultdict(dict)
    pattern = os.path.join(directory, "evalchemy", "evalchemy_*", "**", "results_*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            payload = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        for task, metrics in (payload.get("results") or {}).items():
            if not isinstance(metrics, dict):
                continue
            for key, value in metrics.items():
                metric = base_metric(key)
                if "stderr" in metric:
                    continue
                try:
                    per_task[task][metric] = float(value)
                except (TypeError, ValueError):
                    continue
    out = {}
    for task, metrics in per_task.items():
        chosen = pick(metrics)
        if chosen:
            out[task] = chosen
    return out


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
    """ref -> task -> value from the local copies of the JUPITER 438-task CSVs."""
    refs: dict[str, dict[str, float]] = defaultdict(dict)
    for label, fname in JUPITER_REFS.items():
        path = os.path.join(JUPITER_DIR, fname)
        if not os.path.exists(path):
            print(f"warning: missing {path}", file=sys.stderr)
            continue
        best: dict[str, tuple[int, float]] = {}
        with open(path) as handle:
            for row in csv.DictReader(handle):
                metric = base_metric(row.get("metric") or "")
                if metric not in METRIC_PREFERENCE:
                    continue
                try:
                    value = float(row["value"])
                except (TypeError, ValueError):
                    continue
                task = row["task"]
                rank = METRIC_PREFERENCE.index(metric)
                if task not in best or rank < best[task][0]:
                    best[task] = (rank, value)
        refs[label] = {t: v for t, (_r, v) in best.items()}
    return refs


def read_jupiter_samples() -> dict[str, dict[str, int]]:
    """ref -> task -> n_samples, to drop tasks too small to mean anything (a 4-item task swings 50 pp)."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for label, fname in JUPITER_REFS.items():
        path = os.path.join(JUPITER_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            for row in csv.DictReader(handle):
                try:
                    out[label][row["task"]] = int(float(row.get("n_samples") or 0))
                except (TypeError, ValueError):
                    continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="our eval dir (summary_shard*.tsv)")
    ap.add_argument("--label", default="ours")
    ap.add_argument("--min-common", type=int, default=1, help="only tasks with at least this many references")
    ap.add_argument("--csv", default="", help="also write the table as CSV")
    ap.add_argument("--worst", type=int, default=0, help="also list the N largest negative deltas")
    ap.add_argument("--ref", default="32B@300k", help="reference for --worst")
    ap.add_argument("--min-samples", type=int, default=30,
                    help="drop tasks with fewer reference samples (default 30; 0 keeps all)")
    args = ap.parse_args()

    ours_dir = args.ours if os.path.isdir(args.ours) else os.path.join(EVAL_ROOT, args.ours)
    # summaries only cover the rows of the session that wrote them: merge instead of preferring
    ours = read_results(ours_dir)
    ours.update(read_lighteval(ours_dir))
    ours.update(read_evalchemy(ours_dir))
    ours.update(read_summaries(ours_dir))
    if not ours:
        print(f"no results in {ours_dir}", file=sys.stderr)
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
    sample_sizes = read_jupiter_samples()
    rows = []
    for task in sorted(ours):
        metric, value = ours[task]
        cells = [refs[c].get(task) for c in columns]
        if args.min_samples:
            sizes = [sample_sizes.get(c, {}).get(task) for c in columns]
            sizes = [n for n in sizes if n is not None]
            if sizes and max(sizes) < args.min_samples:
                continue
        if sum(1 for c in cells if c is not None) < args.min_common:
            continue
        rows.append((task, metric, value, cells))

    width = max([len(r[0]) for r in rows] + [4])
    header = f"{'task':{width}} {'metric':<12} {args.label:>7} " + " ".join(f"{c:>14}" for c in columns)
    print(header)
    print("-" * len(header))
    for task, metric, value, cells in rows:
        line = f"{task:{width}} {metric:<12} {to_pct(value):7.2f} "
        line += " ".join("            –" if c is None else f"{to_pct(c):14.2f}" for c in cells)
        print(line)

    print("\nsummary over %d tasks" % len(rows))
    for i, col in enumerate(columns):
        pairs = [(r[2], r[3][i]) for r in rows if r[3][i] is not None]
        if not pairs:
            continue
        wins = sum(1 for a, b in pairs if a > b)
        losses = sum(1 for a, b in pairs if a < b)
        delta = sum(to_pct(a) - to_pct(b) for a, b in pairs) / len(pairs)
        print(f"  vs {col:16s} n={len(pairs):3d}  wins {wins:3d}  losses {losses:3d}  "
              f"mean delta {delta:+6.2f} pp")

    if args.worst and args.ref in columns:
        idx = columns.index(args.ref)
        gaps = sorted(
            ((to_pct(value) - to_pct(cells[idx]), task, metric, value, cells[idx])
             for task, metric, value, cells in rows if cells[idx] is not None),
            key=lambda g: g[0],
        )[: args.worst]
        print(f"\nworst {len(gaps)} vs {args.ref}")
        for delta, task, metric, value, ref_value in gaps:
            print(f"  {delta:+7.2f} pp  {task:44s} {metric:<12} ours {to_pct(value):6.2f}  ref {to_pct(ref_value):6.2f}")

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
