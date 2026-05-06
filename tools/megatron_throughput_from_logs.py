#!/usr/bin/env python3
"""
Parse Megatron stdout logs for per-GPU TFLOP/s and token/s, then print a summary table.

Looks for ``qwen3_*`` experiment subdirectories under each given base directory, reads
``logs/stdout*.log`` first (fallback: ``*.log`` in the run directory). When multiple
``stdout-JOBID.log`` files exist, the largest ``JOBID`` is used (restarts / resubmits).
Entries that are
missing or dangling symlinks (e.g. ``current.log`` before the job writes) are skipped with an error row.

Requires ``log_throughput`` style lines, e.g.::

    iteration      51/91553 | ... elapsed time per iteration (ms): 370.3 | ...
    throughput per GPU (TFLOP/s/GPU): 198.6 | Tokens per second per GPU (Tok/s/GPU): 22120.2 | ...

Usage::

    python megatron_throughput_from_logs.py BASE_DIR [BASE_DIR ...]

    python megatron_throughput_from_logs.py \\
        /path/to/training_workers16_nodes4_disttime60 \\
        /path/to/training_workers16_nodes8_disttime60 \\
        --gpus-per-node 4 \\
        --skip-first-iters 50 --max-elapsed-ms 6000 --max-iters 500 \\
        --csv throughput.csv

``--gpus-per-node`` sets **physical GPU count = nodes × N** from each base path
(``…_nodes32_…`` → 32×4 GPUs if N=4). Without it, **workers × nodes** from the path is used
(often **ranks**, not GPUs).

See README_megatron_throughput_from_logs.md in this directory for details.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PATH_WORKERS_NODES = re.compile(r"training_workers(\d+)_nodes(\d+)_disttime")

SBATCH_NODES = re.compile(r"^#SBATCH\s+--nodes[= ](\d+)", re.MULTILINE)
SBATCH_GPUS_PER_NODE = re.compile(r"^#SBATCH\s+--gpus-per-node[= ](\d+)", re.MULTILINE)
SBATCH_GRES_GPU = re.compile(r"^#SBATCH\s+--gres=gpu[^:\n]*:(\d+)", re.MULTILINE)

ITER_LINE = re.compile(
    r"iteration\s+(\d+)/\s*\d+\s*\|.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+).*?"
    r"Tokens per second per GPU \(Tok/s/GPU\):\s*([\d.]+)",
    re.DOTALL,
)

NAME_LR = re.compile(r"lr([\d.]+)_")
NAME_GBS = re.compile(r"gbsz(\d+)_")


def filter_readable(paths: list[str]) -> list[str]:
    """Drop missing paths / broken symlinks (common for ``current.log`` before job writes)."""
    return sorted(p for p in paths if Path(p).is_file())


def pick_stdout_log(log_paths: list[str]) -> str:
    readable = filter_readable(log_paths)
    outs = [p for p in readable if "stdout" in Path(p).name]
    picks = outs if outs else readable
    if not picks:
        raise FileNotFoundError("no readable log files in candidate list")

    def score(path: str) -> tuple[int, int]:
        """Prefer latest Slurm job id (stdout-JOBID.log); tie-break on larger file."""
        path_obj = Path(path)
        m = re.search(r"stdout-(\d+)\.log$", path_obj.name, re.I)
        job_id = int(m.group(1)) if m else -1
        return (job_id, path_obj.stat().st_size)

    return max(picks, key=score)


def parse_log(
    log_path: Path,
    max_elapsed_ms: float,
    skip_first_iters: int,
    max_iters_used: int,
) -> tuple[dict[str, Any] | None, str | None]:
    text = log_path.read_text(errors="replace")
    rows: list[tuple[int, float, float, float]] = []
    for m in ITER_LINE.finditer(text):
        it_s, et_s, tflop_s, tok_s = m.groups()
        it = int(it_s)
        et = float(et_s)
        tflop = float(tflop_s)
        tok = float(tok_s)
        if et > max_elapsed_ms:
            continue
        rows.append((it, et, tflop, tok))

    if not rows:
        return None, "no matching iteration lines"

    rows.sort(key=lambda x: x[0])
    stable = [r for r in rows if r[0] > skip_first_iters]
    if not stable:
        stable = rows[skip_first_iters:] if len(rows) > skip_first_iters else rows
    if len(stable) > max_iters_used:
        stable = stable[:max_iters_used]
    if not stable:
        return None, "empty after filters"

    tflops = [r[2] for r in stable]
    toks = [r[3] for r in stable]
    out: dict[str, Any] = {
        "n_iters": len(stable),
        "it_first": stable[0][0],
        "it_last": stable[-1][0],
        "avg_tflop_per_gpu": mean(tflops),
        "avg_tok_per_gpu": mean(toks),
        "avg_elapsed_ms": mean(r[1] for r in stable),
    }
    return out, None


def parse_name_meta(exp_name: str) -> tuple[float | None, int | None]:
    lr = gbs = None
    m_lr = NAME_LR.search(exp_name)
    m_gbs = NAME_GBS.search(exp_name)
    if m_lr:
        lr = float(m_lr.group(1))
    if m_gbs:
        gbs = int(m_gbs.group(1))
    return lr, gbs


def gpu_count_from_path(
    workers: int | None,
    nodes: int | None,
    gpus_per_node: int | None,
) -> int | None:
    if workers is None or nodes is None:
        return None
    if gpus_per_node is not None:
        return nodes * gpus_per_node
    return workers * nodes


def parse_sbatch(path: Path) -> tuple[int | None, int | None]:
    """Return (nodes, gpus_per_node) from a SLURM sbatch file, or (None, None) if absent."""
    if not path.is_file():
        return None, None
    text = path.read_text(errors="replace")
    nodes = gpus = None
    m = SBATCH_NODES.search(text)
    if m:
        nodes = int(m.group(1))
    m = SBATCH_GPUS_PER_NODE.search(text)
    if m:
        gpus = int(m.group(1))
    if gpus is None:
        m = SBATCH_GRES_GPU.search(text)
        if m:
            gpus = int(m.group(1))
    return nodes, gpus


def gather_rows(
    bases: list[Path],
    max_elapsed_ms: float,
    skip_first_iters: int,
    max_iters_used: int,
    gpus_per_node: int | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for base in bases:
        if not base.is_dir():
            results.append(
                {
                    "folder": str(base),
                    "experiment": "",
                    "error": f"not a directory: {base}",
                }
            )
            continue

        m_path = PATH_WORKERS_NODES.search(str(base))
        path_workers = int(m_path.group(1)) if m_path else None
        path_nodes = int(m_path.group(2)) if m_path else None

        base_sbatch_nodes, base_sbatch_gpus = parse_sbatch(base / "job.sbatch")

        for exp in sorted(base.glob("qwen3_*")):
            raw_logs = sorted(glob.glob(str(exp / "logs" / "*.log")))
            logs_root = sorted(glob.glob(str(exp / "*.log")))
            logs_read = filter_readable(raw_logs)
            if not logs_read:
                logs_read = filter_readable(logs_root)

            if not logs_read:
                results.append(
                    {
                        "folder": base.name,
                        "experiment": exp.name,
                        "error": "no readable log (missing files or dangling symlinks such as current.log)",
                    }
                )
                continue

            exp_sbatch_nodes, exp_sbatch_gpus = parse_sbatch(exp / "job.sbatch")
            sbatch_nodes = exp_sbatch_nodes or base_sbatch_nodes
            sbatch_gpus = exp_sbatch_gpus or base_sbatch_gpus

            nodes = path_nodes or sbatch_nodes
            # workers_per_node from path regex; fall back to gpus-per-node from sbatch
            workers = path_workers or sbatch_gpus
            effective_gpus_per_node = gpus_per_node or sbatch_gpus

            try:
                log_path = Path(pick_stdout_log(logs_read))
                parsed, err = parse_log(log_path, max_elapsed_ms, skip_first_iters, max_iters_used)
            except FileNotFoundError as e:
                results.append(
                    {
                        "folder": base.name,
                        "experiment": exp.name,
                        "error": f"log read failed: {e}",
                    }
                )
                continue
            lr, gbs = parse_name_meta(exp.name)

            row: dict[str, Any] = {
                "folder": base.name,
                "experiment": exp.name,
                "log": log_path.name,
                "workers_per_node": workers,
                "nodes": nodes,
                "lr": lr,
                "global_batch": gbs,
            }
            if err:
                row["error"] = err
                results.append(row)
                continue

            assert parsed is not None
            n_gpu = gpu_count_from_path(workers, nodes, effective_gpus_per_node)
            row.update(parsed)
            if n_gpu is not None:
                row["n_gpus"] = n_gpu
            row["gpus_per_node_setting"] = gpus_per_node
            results.append(row)

    return results


def print_markdown_table(rows: list[dict[str, Any]]) -> None:
    errs = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]

    for r in errs:
        print(f"ERR {r['folder']}/{r.get('experiment', '')}: {r['error']}", file=sys.stderr)

    if not ok:
        print("\n(No successful runs.)", file=sys.stderr)
        return

    print()
    print(
        "| # | nodes | workers/node | GPUs | lr | global_bs | avg TFLOP/s/gpu | avg Tok/s/gpu | n_iters |"
    )
    print(
        "|--:|------:|-------------:|-----:|-----:|----------:|----------------:|--------------:|--------:|"
    )
    for i, r in enumerate(ok, start=1):
        nodes_s = str(r["nodes"]) if r.get("nodes") is not None else ""
        workers_s = str(r["workers_per_node"]) if r.get("workers_per_node") is not None else ""
        ngpu_s = str(r["n_gpus"]) if r.get("n_gpus") is not None else ""
        print(
            f"| {i} | {nodes_s} | {workers_s} | {ngpu_s} | {r['lr']} | {r['global_batch']} | "
            f"{r['avg_tflop_per_gpu']:.2f} | {r['avg_tok_per_gpu']:.0f} | {r['n_iters']} |"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if not r.get("error")]
    err_rows = [r for r in rows if r.get("error")]

    fieldnames = [
        "index",
        "folder",
        "experiment",
        "nodes",
        "workers_per_node",
        "n_gpus",
        "gpus_per_node_setting",
        "lr",
        "global_batch",
        "avg_tflop_per_gpu",
        "avg_tok_per_gpu",
        "n_iters",
        "it_first",
        "it_last",
        "avg_elapsed_ms",
        "log",
    ]
    err_fieldnames = ["folder", "experiment", "error"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(ok, start=1):
            row_out = {k: r.get(k) for k in fieldnames if k != "index"}
            row_out["index"] = i
            w.writerow(row_out)

        if err_rows:
            f.write("\n")
            we = csv.DictWriter(f, fieldnames=err_fieldnames, extrasaction="ignore")
            we.writeheader()
            for r in err_rows:
                we.writerow({k: r.get(k, "") for k in err_fieldnames})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize Megatron throughput from qwen3_* run logs under base directories."
    )
    p.add_argument(
        "bases",
        nargs="+",
        type=Path,
        help="One or more directories that each contain qwen3_* experiment subfolders",
    )
    p.add_argument(
        "--max-elapsed-ms",
        type=float,
        default=6000.0,
        help="Drop iterations with longer elapsed time (stragglers, checkpoint sync)",
    )
    p.add_argument(
        "--skip-first-iters",
        type=int,
        default=50,
        help="Average only iterations strictly after this index (Megatron warmup)",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=500,
        help="Maximum number of post-filter iterations to average",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write results to this CSV path (includes a second table of errors)",
    )
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=4,
        metavar="N",
        help="Physical GPUs per node: cluster totals use nodes×N from path (omit for legacy workers×nodes)",
    )
    args = p.parse_args()

    if args.gpus_per_node is None:
        print(
            "Note: using workers×nodes from path for GPU count (may be ranks, not GPUs). "
            "Pass --gpus-per-node if your nodes have a fixed GPU count.",
            file=sys.stderr,
        )

    rows = gather_rows(
        list(args.bases),
        args.max_elapsed_ms,
        args.skip_first_iters,
        args.max_iters,
        args.gpus_per_node,
    )
    print_markdown_table(rows)
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nWrote CSV: {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
