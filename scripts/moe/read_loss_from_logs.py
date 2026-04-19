"""
Walk eval result directories, read current.log (or slurm logs), extract the last
validation loss line, and write a summary CSV under results/evals.

The CSV lists the full grid for non-GQA and GQA sweeps (240 rows: GQA no/yes ×120
cells). Parent folders named moe_GQA_* are GQA=yes; all other grid folders are
GQA=no. Missing runs have NaN for iteration and lm loss.

Expected run folder names like:
  moe_130M_eval_nexp_64_lr0.0005_gbsz1024_seed1234_decay120BT
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

# Repo-local default: oellm-autoexp/results/evals
EVALS_ROOT = Path(__file__).resolve().parents[2] / "results" / "evals"
CSV_NAME = "validation_loss_summary.csv"

# Full grid: 5 * 3 * 4 * 2 = 120 rows
GRID_LR = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3]
GRID_BS = [1024, 512, 256]
GRID_N_EXP = [8, 16, 32, 64]
GRID_DECAY_BT = [80, 120]

# Optional "[rank]: " or "[nid123:456]: " prefix; short or long metric tail
VAL_LINE = re.compile(
    r"(?:\[[^\]]+\]:\s*)?"
    r"validation loss at iteration (?P<iteration>\d+) on validation set \|"
    r".*?lm loss value:\s*(?P<lm_loss_value>[\d.E+-]+)",
)

# moe_130M_eval_nexp_64_lr0.0005_gbsz1024_seed1234_decay120BT
RUN_NAME = re.compile(
    r"nexp_(?P<n_exp>\d+)_lr(?P<lr>[\d.]+)_gbsz(?P<bsz>\d+)_seed\d+_decay(?P<decay>\d+)BT"
)


def lr_key(lr: float) -> float:
    """Stable key for matching parsed folder lr to grid values."""
    return round(float(lr), 12)


def gqa_from_grid_dir_name(grid_dir_name: str) -> str:
    """yes if sweep lives under moe_GQA_*, else no."""
    return "yes" if grid_dir_name.startswith("moe_GQA_") else "no"


def parse_run_name(name: str) -> dict[str, float | int] | None:
    m = RUN_NAME.search(name)
    if not m:
        return None
    return {
        "n_exp": int(m.group("n_exp")),
        "lr": float(m.group("lr")),
        "bsz": int(m.group("bsz")),
        "decay_bt": int(m.group("decay")),
    }


def last_validation_from_log(log_path: Path) -> dict[str, float | int] | None:
    """Return iteration and lm loss from the last matching line in the file."""
    last = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = VAL_LINE.search(line)
            if m:
                last = {
                    "iteration": int(m.group("iteration")),
                    "lm loss": float(m.group("lm_loss_value")),
                }
    return last


def resolve_log(run_dir: Path) -> Path | None:
    cur = run_dir / "current.log"
    if cur.is_file():
        return cur
    slurm = sorted(run_dir.glob("slurm-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return slurm[0] if slurm else None


GridKey = tuple[str, int, float, int, int]


def index_runs(evals_root: Path) -> tuple[dict[GridKey, dict[str, float | int]], list[str]]:
    """Map (GQA, bsz, lr_key, n_exp, decay_bt) -> {iteration, lm loss}."""
    indexed: dict[GridKey, dict[str, float | int]] = {}
    warnings: list[str] = []
    if not evals_root.is_dir():
        warnings.append(f"Missing or not a directory: {evals_root}")
        return indexed, warnings

    for grid_dir in sorted(evals_root.iterdir()):
        if not grid_dir.is_dir():
            continue
        gqa = gqa_from_grid_dir_name(grid_dir.name)
        for run_dir in sorted(grid_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            meta = parse_run_name(run_name)
            log_path = resolve_log(run_dir)
            if meta is None:
                warnings.append(f"skip (bad run name): {run_dir}")
                continue
            if log_path is None:
                warnings.append(f"skip (no log): {run_dir}")
                continue
            metrics = last_validation_from_log(log_path)
            if metrics is None:
                warnings.append(f"skip (no validation line): {log_path}")
                continue
            key: GridKey = (gqa, meta["bsz"], lr_key(meta["lr"]), meta["n_exp"], meta["decay_bt"])
            if key in indexed:
                warnings.append(f"duplicate grid point GQA={gqa} {key[1:]}: replacing with {run_dir}")
            indexed[key] = dict(metrics)
    return indexed, warnings


def build_full_grid(
    indexed: dict[GridKey, dict[str, float | int]],
) -> list[dict[str, float | int | str]]:
    """One row per (GQA × grid cell); NaN where no run produced metrics."""
    rows: list[dict[str, float | int | str]] = []
    nan = float("nan")
    for gqa in ("no", "yes"):
        for bsz in GRID_BS:
            for lr in GRID_LR:
                for n_exp in GRID_N_EXP:
                    for decay_bt in GRID_DECAY_BT:
                        k: GridKey = (gqa, bsz, lr_key(lr), n_exp, decay_bt)
                        row: dict[str, float | int | str] = {
                            "GQA": gqa,
                            "bsz": bsz,
                            "lr": lr,
                            "n_exp": n_exp,
                            "decay_bt": decay_bt,
                        }
                        if k in indexed:
                            row["iteration"] = indexed[k]["iteration"]
                            row["lm loss"] = indexed[k]["lm loss"]
                        else:
                            row["iteration"] = nan
                            row["lm loss"] = nan
                        rows.append(row)
    return rows


def main() -> None:
    evals_root = EVALS_ROOT
    if len(sys.argv) > 1:
        evals_root = Path(sys.argv[1]).resolve()

    indexed, warnings = index_runs(evals_root)
    rows = build_full_grid(indexed)
    out_cols = ["GQA", "bsz", "lr", "n_exp", "decay_bt", "iteration", "lm loss"]

    csv_path = evals_root / CSV_NAME
    evals_root.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} rows to {csv_path}")
    n_ok = sum(
        1
        for r in rows
        if not (isinstance(r["iteration"], float) and math.isnan(r["iteration"]))
    )
    print(f"Filled {n_ok} / {len(rows)} grid cells from logs")
    for line in warnings:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
