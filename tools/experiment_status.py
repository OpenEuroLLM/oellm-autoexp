#!/usr/bin/env python3
"""Summarise the status of training experiments in a directory.

For each run subdirectory the script reports:
  - Stop reason: TIME_LIMIT, SEGFAULT, or UNKNOWN
  - Latest iteration reached (from stdout logs)
  - eval_interval configured (from config yaml)
  - Whether the latest iteration is aligned with eval_interval

Usage:
    python experiment_status.py <runs_dir>
"""

import argparse
import re
import sys
from pathlib import Path

ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*\d+")
EVAL_INTERVAL_RE = re.compile(r"^\s*eval_interval:\s*(\d+)", re.MULTILINE)
TIME_LIMIT_RE = re.compile(r"DUE TO TIME LIMIT", re.IGNORECASE)
SEGFAULT_RE = re.compile(r"Segmentation fault|segfault", re.IGNORECASE)


def find_logs(run_dir: Path, prefix: str) -> list[Path]:
    """Return all logs/<prefix>-*.log files, sorted by modification time."""
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return []
    files = sorted(
        logs_dir.glob(f"{prefix}-*.log"),
        key=lambda f: f.stat().st_mtime,
    )
    return files


def detect_stop_reason(stderr_logs: list[Path]) -> str:
    """Return TIME_LIMIT, SEGFAULT, or UNKNOWN by scanning all stderr files."""
    has_segfault = False
    has_time_limit = False
    for log in stderr_logs:
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        if TIME_LIMIT_RE.search(text):
            has_time_limit = True
        if SEGFAULT_RE.search(text):
            has_segfault = True
    if has_time_limit and has_segfault:
        return "TIME_LIMIT+SEGFAULT"
    if has_time_limit:
        return "TIME_LIMIT"
    if has_segfault:
        return "SEGFAULT"
    return "UNKNOWN"


def latest_iteration(stdout_logs: list[Path]) -> int | None:
    """Return the highest iteration number seen across all stdout logs."""
    last = None
    for log in stdout_logs:
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for m in ITER_RE.finditer(text):
            val = int(m.group(1))
            if last is None or val > last:
                last = val
    return last


def get_eval_interval(run_dir: Path) -> int | None:
    """Extract eval_interval from the config-*.yaml file in the run
    directory."""
    configs = sorted(run_dir.glob("config-*.yaml"), key=lambda f: f.stat().st_mtime)
    for cfg in reversed(configs):  # most recent first
        try:
            text = cfg.read_text(errors="replace")
        except OSError:
            continue
        m = EVAL_INTERVAL_RE.search(text)
        if m:
            return int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Summarise experiment run status in a directory.")
    parser.add_argument("runs_dir", type=Path, help="Directory containing run subdirectories")
    args = parser.parse_args()

    runs_dir: Path = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        print(f"Error: {runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No subdirectories found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for run_dir in run_dirs:
        stderr_logs = find_logs(run_dir, "stderr")
        stdout_logs = find_logs(run_dir, "stdout")

        if not stderr_logs and not stdout_logs:
            rows.append(
                {
                    "name": run_dir.name,
                    "stop_reason": "NO_LOGS",
                    "latest_iter": None,
                    "eval_interval": None,
                    "iter_aligned": None,
                }
            )
            continue

        stop_reason = detect_stop_reason(stderr_logs)
        last_iter = latest_iteration(stdout_logs)
        eval_int = get_eval_interval(run_dir)

        if last_iter is not None and eval_int is not None:
            aligned = last_iter % eval_int == 0
        else:
            aligned = None

        rows.append(
            {
                "name": run_dir.name,
                "stop_reason": stop_reason,
                "latest_iter": last_iter,
                "eval_interval": eval_int,
                "iter_aligned": aligned,
            }
        )

    # ---- print summary table ----
    col_name = max(len(r["name"]) for r in rows) + 2
    col_reason = max(len(r["stop_reason"]) for r in rows) + 2
    col_iter = 12
    col_eval = 13
    col_align = 8

    header = (
        f"{'Run':<{col_name}}"
        f"{'Stop reason':<{col_reason}}"
        f"{'Latest iter':>{col_iter}}"
        f"{'eval_interval':>{col_eval}}"
        f"{'Aligned':>{col_align}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    counts = {"TIME_LIMIT": 0, "SEGFAULT": 0, "TIME_LIMIT+SEGFAULT": 0, "UNKNOWN": 0, "NO_LOGS": 0}
    aligned_count = 0
    misaligned_count = 0

    for r in rows:
        iter_str = str(r["latest_iter"]) if r["latest_iter"] is not None else "N/A"
        eval_str = str(r["eval_interval"]) if r["eval_interval"] is not None else "N/A"
        if r["iter_aligned"] is None:
            align_str = "N/A"
        elif r["iter_aligned"]:
            align_str = "YES"
            aligned_count += 1
        else:
            align_str = "NO"
            misaligned_count += 1

        print(
            f"{r['name']:<{col_name}}"
            f"{r['stop_reason']:<{col_reason}}"
            f"{iter_str:>{col_iter}}"
            f"{eval_str:>{col_eval}}"
            f"{align_str:>{col_align}}"
        )
        key = r["stop_reason"] if r["stop_reason"] in counts else "UNKNOWN"
        counts[key] += 1

    print(sep)
    print("\nSummary:")
    for reason, count in counts.items():
        if count:
            print(f"  {reason:<24} {count}")
    print(f"  {'Iter aligned with eval':24} {aligned_count}")
    print(f"  {'Iter NOT aligned':24} {misaligned_count}")
    print(sep)


if __name__ == "__main__":
    main()
