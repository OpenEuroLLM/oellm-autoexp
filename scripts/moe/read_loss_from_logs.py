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


def detect_gqa(name: str) -> str:
    """Detect GQA from a folder name. 'no_GQA' or no mention → no; 'GQA' → yes."""
    if "no_GQA" in name:
        return "no"
    if "GQA" in name:
        return "yes"
    return "no"


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


def index_runs(evals_root: Path) -> tuple[dict[GridKey, list[dict[str, float | int | str]]], list[str]]:
    """Map (GQA, bsz, lr_key, n_exp, decay_bt) -> list of {iteration, lm loss, run_name}.

    Keeps duplicates (e.g. a base sweep and a replication sweep that both map to GQA=no)
    so every run folder shows up as its own row in the final CSV.
    """
    indexed: dict[GridKey, list[dict[str, float | int | str]]] = {}
    warnings: list[str] = []
    if not evals_root.is_dir():
        warnings.append(f"Missing or not a directory: {evals_root}")
        return indexed, warnings

    for grid_dir in sorted(evals_root.iterdir()):
        if not grid_dir.is_dir():
            continue
        gqa = detect_gqa(grid_dir.name)
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
            entry: dict[str, float | int | str] = {**metrics, "run_name": run_name}
            indexed.setdefault(key, []).append(entry)
    return indexed, warnings


def build_full_grid(
    indexed: dict[GridKey, list[dict[str, float | int | str]]],
) -> list[dict[str, float | int | str]]:
    """One row per (GQA × grid cell), or one per matching run when a cell has multiples.

    Cells with no matching run produce a single NaN row.
    """
    rows: list[dict[str, float | int | str]] = []
    nan = float("nan")
    for gqa in ("no", "yes"):
        for bsz in GRID_BS:
            for lr in GRID_LR:
                for n_exp in GRID_N_EXP:
                    for decay_bt in GRID_DECAY_BT:
                        k: GridKey = (gqa, bsz, lr_key(lr), n_exp, decay_bt)
                        base: dict[str, float | int | str] = {
                            "GQA": gqa,
                            "bsz": bsz,
                            "lr": lr,
                            "n_exp": n_exp,
                            "decay_bt": decay_bt,
                        }
                        entries = indexed.get(k, [])
                        if not entries:
                            rows.append({**base, "iteration": nan, "lm loss": nan, "run_name": ""})
                            continue
                        for entry in entries:
                            rows.append({
                                **base,
                                "iteration": entry["iteration"],
                                "lm loss": entry["lm loss"],
                                "run_name": entry.get("run_name", ""),
                            })
    return rows


def is_single_grid_folder(path: Path) -> bool:
    """True when path looks like a single grid folder (children are run dirs, not grid dirs)."""
    for child in path.iterdir():
        if child.is_dir() and RUN_NAME.search(child.name):
            return True
    return False


def scan_single_folder(grid_dir: Path) -> tuple[list[dict], list[str]]:
    """Scan one grid folder and return rows + warnings (no full-grid padding)."""
    gqa = detect_gqa(grid_dir.name)
    rows: list[dict] = []
    warnings: list[str] = []
    for run_dir in sorted(grid_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_run_name(run_dir.name)
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
        rows.append({"GQA": gqa, **meta, **metrics, "run_name": run_dir.name})
    rows.sort(key=lambda r: (r["bsz"], r["lr"], r["n_exp"], r["decay_bt"]))
    return rows, warnings


def write_csv(rows: list[dict], out_cols: list[str], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    target = EVALS_ROOT
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()

    # Single grid folder mode: flat CSV of whatever runs are inside
    if target.is_dir() and is_single_grid_folder(target):
        rows, warnings = scan_single_folder(target)
        out_cols = ["GQA", "bsz", "lr", "n_exp", "decay_bt", "iteration", "lm loss", "run_name"]
        csv_path = target / CSV_NAME
        write_csv(rows, out_cols, csv_path)
        print(f"Wrote {len(rows)} rows to {csv_path}")
        for line in warnings:
            print(line, file=sys.stderr)
        return

    # Full grid mode: Cartesian product across all grid folders; duplicates (e.g.
    # replication sweeps) show up as extra rows on the same grid cell.
    indexed, warnings = index_runs(target)
    rows = build_full_grid(indexed)
    out_cols = ["GQA", "bsz", "lr", "n_exp", "decay_bt", "iteration", "lm loss", "run_name"]
    csv_path = target / CSV_NAME
    write_csv(rows, out_cols, csv_path)
    n_runs = sum(
        1
        for r in rows
        if not (isinstance(r["iteration"], float) and math.isnan(r["iteration"]))
    )
    n_empty = len(rows) - n_runs
    print(f"Wrote {len(rows)} rows to {csv_path} ({n_runs} from logs, {n_empty} empty grid cells)")
    for line in warnings:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
