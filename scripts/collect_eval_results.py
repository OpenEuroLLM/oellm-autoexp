#!/usr/bin/env python3
"""Collect lm-eval results into a CSV grouped by run and iteration.

Usage:
    python collect_eval_results.py /shared_silo/scratch/$USER/eval_results --filter bilingual
"""

import argparse
import csv
import json
import sys
from pathlib import Path

TASKS = [
    ("arc_challenge", "acc_norm,none"),
    ("arc_challenge_mt_fi", "acc_norm,none"),
    ("hellaswag", "acc_norm,none"),
    ("ogx_hellaswagx_fi", "acc_norm,none"),
    ("goldenswag", "acc_norm,none"),
    ("ogx_goldenswagx_fi", "acc_norm,none"),
    ("mmlu", "acc,none"),
    ("ogx_mmlux_FI", "acc,none"),
    ("truthfulqa_mc2", "acc,none"),
    ("ogx_truthfulqax_mc2_fi", "acc,none"),
    ("gsm8k", "exact_match,strict-match"),
    ("ogx_gsm8kx_fi", "acc,none"),
    ("finbench_v2", "acc_norm,none"),
]


def find_results(eval_dir: Path) -> dict[str, dict[str, float]]:
    """Return {run/iter: {task: score}} from all results JSONs."""
    rows: dict[str, dict[str, float]] = {}

    for rfile in sorted(eval_dir.rglob("results_*.json")):
        with open(rfile) as f:
            data = json.load(f)

        model_path = data.get("model_name", "")
        if not model_path:
            continue

        p = Path(model_path)
        if p.name.startswith("iter_"):
            key = f"{p.parent.name}/{p.name}"
        else:
            key = p.name

        results = data.get("results", {})
        if key not in rows:
            rows[key] = {}

        for task_name, metric_key in TASKS:
            if task_name in results and metric_key in results[task_name]:
                rows[key][task_name] = results[task_name][metric_key]

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_dir",
        type=Path,
        help="Directory containing eval_results",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only include rows whose key contains this string",
    )
    parser.add_argument(
        "-o", "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output CSV file (default: stdout)",
    )
    args = parser.parse_args()

    rows = find_results(args.eval_dir)

    header_tasks = [t for t, _ in TASKS]
    writer = csv.writer(args.output, delimiter="\t")
    writer.writerow(["run", "iter"] + header_tasks)

    for key in sorted(rows):
        if args.filter and args.filter not in key:
            continue
        if "/" in key:
            run, iter_part = key.rsplit("/", 1)
        else:
            run, iter_part = key, ""

        row = [run, iter_part]
        for task_name, _ in TASKS:
            val = rows[key].get(task_name)
            row.append(f"{val:.4f}" if val is not None else "")
        writer.writerow(row)


if __name__ == "__main__":
    main()
