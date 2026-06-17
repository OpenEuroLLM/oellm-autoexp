#!/usr/bin/env python3
"""
Find and optionally open the log file for a training run.

Accepts either a SLURM job ID (numeric) or a run name pattern (substring match).
Defaults to stderr / current.log; use --stdout for the stdout log.

Usage:
    # By SLURM job ID
    python get_logfile.py 41827746

    # By run name pattern (partial match OK)
    python get_logfile.py gbsz128_stable
    python get_logfile.py lr0.0005_gbsz64

    # Just print the path, don't open
    python get_logfile.py 41827746 --no-open

    # Stdout instead of stderr
    python get_logfile.py gbsz128_stable --stdout

    # Explicit results dir
    python get_logfile.py gbsz128_stable -d /path/to/training
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


_DEFAULT_SEARCH_ROOTS = [
    "/leonardo_work/OELLM_prod2026/experiments/multilingual_scaling",
    "/leonardo_work/OELLM_prod2026/{user}/multilingual_scaling",
]


def _training_dirs(roots):
    result = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("training")):
            if p.is_dir():
                result.append(p)
        result.append(root)
    return result


def find_by_slurm_id(search_dirs, job_id, stdout):
    prefix = "stdout" if stdout else "stderr"
    for d in search_dirs:
        if not d.is_dir():
            continue
        for run_dir in sorted(d.iterdir()):
            if not run_dir.is_dir():
                continue
            log = run_dir / "logs" / ("{}-{}.log".format(prefix, job_id))
            if log.exists():
                return log
    return None


def find_by_name(search_dirs, pattern, stdout):
    matches: list[Path] = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        for run_dir in sorted(d.iterdir()):
            if run_dir.is_dir() and pattern in run_dir.name:
                matches.append(run_dir)

    if not matches:
        return None

    if len(matches) > 1:
        print("Multiple matches — be more specific:", file=sys.stderr)
        for m in sorted(matches):
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    run_dir = matches[0]

    # current.log symlink always points to latest stderr
    if not stdout:
        current = run_dir / "current.log"
        if current.exists():
            return current

    # Fall back to latest log file in logs/
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        prefix = "stdout" if stdout else "stderr"
        files = sorted(logs_dir.glob(f"{prefix}-*.log"))
        if files:
            return files[-1]

    return None


def open_in_vscode(path):
    # Inside a Singularity container SINGULARITY_CONTAINER is set;
    # 'code' may or may not be reachable — try anyway, fail silently.
    try:
        subprocess.run(["code", str(path)], check=False)
    except FileNotFoundError:
        print("(VSCode 'code' CLI not found — copy the path above to open manually)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("query", help="SLURM job ID (numeric) or run name substring")
    ap.add_argument(
        "-d", "--results-dir",
        type=Path,
        default=None,
        help="Directory containing run subdirectories (auto-discovered if omitted)",
    )
    ap.add_argument("--stderr", action="store_true", help="Return stderr log instead of stdout")
    ap.add_argument(
        "--open", dest="open_file", action="store_true", default=True,
        help="Open in VSCode after printing path (default: on)",
    )
    ap.add_argument("--no-open", dest="open_file", action="store_false", help="Just print the path")
    args = ap.parse_args()

    if args.results_dir is not None:
        search_dirs = [args.results_dir]
    else:
        user = os.environ.get("USER", "")
        roots = [Path(t.format(user=user)) for t in _DEFAULT_SEARCH_ROOTS]
        search_dirs = _training_dirs(roots)

    is_slurm_id = bool(re.fullmatch(r"\d+", args.query))
    use_stdout = not args.stderr

    log_path = (
        find_by_slurm_id(search_dirs, args.query, use_stdout)
        if is_slurm_id
        else find_by_name(search_dirs, args.query, use_stdout)
    )

    if log_path is None:
        print(f"No log found for: {args.query}", file=sys.stderr)
        sys.exit(1)

    log_path = log_path.resolve()
    print(log_path)

    if args.open_file:
        open_in_vscode(log_path)


if __name__ == "__main__":
    main()
