#!/usr/bin/env python3
"""
Extract the final validation loss from the latest .log file in each run subdirectory
and write results to a CSV.

Usage:
    python collect_val_loss.py <runs_dir> [--output <output.csv>]
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

VAL_LOSS_RE = re.compile(
    r"validation loss at iteration\s+(\d+)\s+on validation set\s*\|.*?lm loss value:\s*([\d.E+\-]+)\s*\|.*?lm loss PPL:\s*([\d.E+\-]+)"
)


def find_latest_log(run_dir: Path) -> Optional[Path]:
    logs = [f for f in run_dir.iterdir() if f.suffix == ".log" and f.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda f: f.stat().st_mtime)


def extract_last_val_loss(log_file: Path) -> Optional[Tuple[int, float, float]]:
    last_match = None
    with log_file.open(errors="replace") as f:
        for line in f:
            m = VAL_LOSS_RE.search(line)
            if m:
                last_match = m
    if last_match is None:
        return None
    iteration = int(last_match.group(1))
    loss = float(last_match.group(2))
    ppl = float(last_match.group(3))
    return iteration, loss, ppl


def main():
    parser = argparse.ArgumentParser(description="Collect final validation loss from run directories.")
    parser.add_argument("runs_dir", type=Path, help="Directory containing run subdirectories")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output CSV file (default: <runs_dir>/val_loss_summary.csv)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.runs_dir / "val_loss_summary.csv"

    if not args.runs_dir.is_dir():
        print(f"Error: {args.runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No subdirectories found in {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for run_dir in run_dirs:
        log_file = find_latest_log(run_dir)
        if log_file is None:
            print(f"  [skip] {run_dir.name}: no .log files found")
            continue

        result = extract_last_val_loss(log_file)
        if result is None:
            print(f"  [skip] {run_dir.name}: no validation loss found in {log_file.name}")
            continue

        iteration, loss, ppl = result
        print(f"  {run_dir.name}: iter={iteration}  loss={loss:.6E}  PPL={ppl:.6E}  ({log_file.name})")
        rows.append({
            "run_name": run_dir.name,
            "iteration": iteration,
            "lm_loss": loss,
            "lm_loss_ppl": ppl,
        })

    if not rows:
        print("No results collected — nothing written.", file=sys.stderr)
        sys.exit(1)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "iteration", "lm_loss", "lm_loss_ppl"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
