#!/usr/bin/env python3
"""Collect the final validation loss from the latest .log file in each run
subdirectory and write results to per-stage CSVs.

Runs are classified by the stage marker in their directory name
(``..._branch<N>BT``, ``..._decay<N>BT``, ``..._stable<N>BT`` /
``..._end<N>BT``) and written to three separate files:

- predecay.csv: branch runs
- decay.csv: decay runs
- stable.csv: stable and end runs

Usage:
    python collect_val_loss.py <runs_dir> --output-dir <dir>
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Make sibling modules importable when run as a standalone script.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_guard import guard_write  # noqa: E402

VAL_LOSS_RE = re.compile(
    r"validation loss at iteration\s+(\d+)\s+on validation set\s*\|.*?lm loss value:\s*([\d.E+\-]+)\s*\|.*?lm loss PPL:\s*([\d.E+\-]+)"
)

STAGE_RE = re.compile(r"_(branch|decay|stable|end)\d+BT")

STAGE_TO_FILE = {
    "branch": "predecay",
    "decay": "decay",
    "stable": "stable",
    "end": "stable",
}


def find_latest_log(run_dir: Path) -> Path | None:
    logs = [
        f
        for f in run_dir.rglob("*.log")
        if "wandb" not in f.relative_to(run_dir).parts and f.is_file()
    ]
    if not logs:
        return None
    return max(logs, key=lambda f: f.stat().st_mtime)


def extract_last_val_loss(log_file: Path) -> tuple[int, float, float] | None:
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


def classify_run(run_name: str) -> str | None:
    """Return the output basename ("predecay", "decay", "stable") for a run,
    or None if the run name has no recognized stage marker."""
    m = STAGE_RE.search(run_name)
    if m is None:
        return None
    return STAGE_TO_FILE[m.group(1)]


def read_existing_rows(csv_path: Path) -> dict[str, dict]:
    """Read a previously written CSV into a dict keyed by run_name."""
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {row["run_name"]: row for row in reader}


def main():
    parser = argparse.ArgumentParser(
        description="Collect final validation loss from run directories, split by stage."
    )
    parser.add_argument("runs_dir", type=Path, help="Directory containing run subdirectories")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to write predecay.csv, decay.csv, and stable.csv into",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.runs_dir.is_dir():
        print(f"Error: {args.runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    if not run_dirs:
        print(f"No subdirectories found in {args.runs_dir}", file=sys.stderr)
        sys.exit(1)

    rows_by_stage: dict[str, list[dict]] = {"predecay": [], "decay": [], "stable": []}

    for run_dir in run_dirs:
        log_file = find_latest_log(run_dir)
        if log_file is None:
            print(f"  [skip] {run_dir.name}: no .log files found")
            continue

        result = extract_last_val_loss(log_file)
        if result is None:
            print(f"  [skip] {run_dir.name}: no validation loss found in {log_file.name}")
            continue

        stage = classify_run(run_dir.name)
        if stage is None:
            print(f"  [skip] {run_dir.name}: no recognized stage marker (branch/decay/stable/end)")
            continue

        iteration, loss, ppl = result
        print(
            f"  {run_dir.name}: stage={stage}  iter={iteration}  loss={loss:.6E}  PPL={ppl:.6E}  ({log_file.name})"
        )
        rows_by_stage[stage].append(
            {
                "run_name": run_dir.name,
                "iteration": iteration,
                "lm_loss": loss,
                "lm_loss_ppl": ppl,
            }
        )

    if not any(rows_by_stage.values()):
        print("No results collected — nothing written.", file=sys.stderr)
        sys.exit(1)

    for stage, filename in (("predecay", "predecay.csv"), ("decay", "decay.csv"), ("stable", "stable.csv")):
        new_rows = rows_by_stage[stage]
        out_path = output_dir / filename
        merged = read_existing_rows(out_path)
        n_before = len(merged)

        if not new_rows and n_before == 0:
            print(f"\nNo {stage} results collected — skipping {filename}.")
            continue

        n_new = sum(1 for row in new_rows if row["run_name"] not in merged)
        n_updated = len(new_rows) - n_new
        for row in new_rows:
            merged[row["run_name"]] = row

        guard_write(out_path)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run_name", "iteration", "lm_loss", "lm_loss_ppl"])
            writer.writeheader()
            writer.writerows(sorted(merged.values(), key=lambda r: r["run_name"]))

        print(
            f"\nWrote {len(merged)} rows to {out_path} "
            f"({n_new} new, {n_updated} updated, {n_before} preexisting)"
        )


if __name__ == "__main__":
    main()
