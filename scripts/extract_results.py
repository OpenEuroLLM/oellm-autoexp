#!/usr/bin/env python3
"""Extract TFLOPs / MFU / TPS / config from Titan and Megatron training logs.

Usage (run on JUPITER or locally if output dir is mounted):
    python scripts/extract_results.py \\
        --base-dirs \\
            /e/scratch/projectnucleus/poeppel1/output/titan_moe_30BA3B_comparable \\
            /e/scratch/projectnucleus/poeppel1/output/megatron_moe_30BA3B_comparable

Options:
    --warmup N      Skip first N iterations when computing medians (default: 10)
    --csv           Also write a CSV file next to the script
    --sort FIELD    Sort output by: tflops, tps, mfu, name (default: name)
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

# GH200 SXM BF16 peak TFLOPs — derived from Titan logs (tflops / mfu_frac).
GH200_PEAK_TFLOPS = 989.4

# ── regex patterns ────────────────────────────────────────────────────────────

# Titan per-step log line (one per rank; we filter to rank 0 only):
#   0: [titan] 2026-03-24 ... step: 80  loss: ...  memory:  5.71GiB(6.01%)  tps: 57,729  tflops: 78.76  mfu: 7.96%
TITAN_RE = re.compile(
    r"^0:.*step:\s*(\d+)"
    r".*?memory:\s*([\d.]+)GiB\(([\d.]+)%\)"
    r".*?tps:\s*([\d,]+)"
    r".*?tflops:\s*([\d.]+)"
    r".*?mfu:\s*([\d.]+)%"
)

# Megatron per-iteration log line (logged once per iteration from one rank):
#   [default3]: [2026-...] iteration  80/  100 | ... | elapsed time per iteration (ms): 1548.1 | mem usages: 0.8131 | throughput per GPU (TFLOP/s/GPU): 57.6 | Tokens per second per GPU (Tok/s/GPU): 2645.8 | ...
MEGATRON_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+"
    r".*?elapsed time per iteration \(ms\):\s*([\d.]+)"
    r".*?mem usages:\s*([\d.]+)"
    r".*?throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)"
    r".*?Tokens per second per GPU \(Tok/s/GPU\):\s*([\d.]+)"
)

OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|calloc.*bytes.*failed|Failed to CUDA calloc",
    re.IGNORECASE,
)

# ── config parsing from directory name ───────────────────────────────────────


def parse_config(name: str) -> dict:
    """Extract config fields from job directory name."""
    cfg = {"name": name}

    m = re.search(r"_ep(\d+)", name)
    cfg["ep"] = int(m.group(1)) if m else None

    m = re.search(r"_pp(\d+)", name)
    cfg["pp"] = int(m.group(1)) if m else None

    m = re.search(r"_tp(\d+)", name)
    cfg["tp"] = int(m.group(1)) if m else None

    m = re.search(r"_(?:mbs|bs)(\d+)", name)
    cfg["bs"] = int(m.group(1)) if m else None

    if "_nocompile" in name:
        cfg["compile"] = "no"
    elif "_compile" in name:
        cfg["compile"] = "yes"
    else:
        cfg["compile"] = None

    if "acfull" in name or "full_recompute" in name:
        cfg["ac"] = "full"
    elif "acnone" in name or "no_recompute" in name:
        cfg["ac"] = "none"
    else:
        cfg["ac"] = None

    if name.startswith("titan_"):
        cfg["backend"] = "titan"
    elif name.startswith("meg_"):
        cfg["backend"] = "megatron"
    else:
        cfg["backend"] = "unknown"

    return cfg


# ── log parsing ───────────────────────────────────────────────────────────────


def parse_log(log_path: Path, backend: str, warmup: int) -> dict:
    """Parse a single log file.

    Returns a metrics dict.
    """
    tflops, tps, mfu, mem_gib, mem_pct = [], [], [], [], []
    oom = False

    pattern = TITAN_RE if backend == "titan" else MEGATRON_RE

    with open(log_path, errors="replace") as f:
        for line in f:
            if OOM_RE.search(line):
                oom = True
            m = pattern.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if step <= warmup:
                continue

            if backend == "titan":
                mem_gib.append(float(m.group(2)))
                mem_pct.append(float(m.group(3)))
                tps.append(float(m.group(4).replace(",", "")))
                tflops.append(float(m.group(5)))
                mfu.append(float(m.group(6)))
            else:  # megatron
                # m.group(2) = elapsed_ms (unused here)
                mem_pct.append(float(m.group(3)) * 100.0)
                tflops.append(float(m.group(4)))
                tps.append(float(m.group(5)))

    if not tflops:
        return {"status": "oom" if oom else "no_data"}

    med_tflops = statistics.median(tflops)
    med_tps = statistics.median(tps)
    result = {
        "status": "oom" if oom else "ok",
        "tflops": med_tflops,
        "tps": med_tps,
        "mfu_pct": statistics.median(mfu) if mfu else med_tflops / GH200_PEAK_TFLOPS * 100,
        "mem_gib": statistics.median(mem_gib) if mem_gib else None,
        "mem_pct": statistics.median(mem_pct) if mem_pct else None,
        "n_iters": len(tflops),
    }
    return result


def find_log(job_dir: Path, backend: str) -> Path | None:
    """Find the most recent slurm log in a job directory."""
    logs = sorted(job_dir.glob("slurm-*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        required=True,
        metavar="DIR",
        help="Output directories to scan (one or more)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Iterations to skip at start (default: 10)"
    )
    parser.add_argument(
        "--csv", action="store_true", help="Print CSV instead of a human-readable table"
    )
    parser.add_argument(
        "--sort",
        default="name",
        choices=["tflops", "tps", "mfu", "name"],
        help="Sort column (default: name)",
    )
    args = parser.parse_args()

    rows = []
    for base in args.base_dirs:
        base_path = Path(base)
        if not base_path.is_dir():
            print(f"[WARN] Not a directory: {base}", file=sys.stderr)
            continue
        for job_dir in sorted(base_path.iterdir()):
            if not job_dir.is_dir():
                continue
            cfg = parse_config(job_dir.name)
            log = find_log(job_dir, cfg["backend"])
            if log is None:
                metrics = {"status": "no_log"}
            else:
                metrics = parse_log(log, cfg["backend"], args.warmup)
                # Attach job id from log filename
                cfg["job_id"] = re.search(r"slurm-(\d+)", log.name)
                cfg["job_id"] = cfg["job_id"].group(1) if cfg["job_id"] else ""
            rows.append({**cfg, **metrics})

    # Sort
    sort_key = {
        "tflops": lambda r: -(r.get("tflops") or 0),
        "tps": lambda r: -(r.get("tps") or 0),
        "mfu": lambda r: -(r.get("mfu_pct") or 0),
        "name": lambda r: r["name"],
    }[args.sort]
    rows.sort(key=sort_key)

    if args.csv:
        _print_csv(rows)
    else:
        _print_table(rows)


def _fmt(val, fmt=".1f", suffix="", na="—"):
    if val is None:
        return na
    return f"{val:{fmt}}{suffix}"


def _print_table(rows):
    # Header
    cols = [
        ("backend", 8, "backend"),
        ("ep", 4, "EP"),
        ("pp", 4, "PP"),
        ("compile", 8, "compile"),
        ("ac", 6, "AC"),
        ("bs", 4, "BS"),
        ("status", 8, "status"),
        ("tflops", 10, "TFLOPS/GPU"),
        ("mfu_pct", 10, "MFU%"),
        ("tps", 12, "TPS/GPU"),
        ("mem_gib", 10, "Mem(GiB)"),
        ("mem_pct", 10, "Mem%"),
        ("n_iters", 7, "iters"),
        ("job_id", 10, "job_id"),
    ]

    header = "  ".join(f"{h:{w}}" for _, w, h in cols)
    sep = "  ".join("-" * w for w, *_ in [(w,) for _, w, _ in cols])
    print(header)
    print(sep)

    for r in rows:

        def g(k):
            return r.get(k)

        vals = [
            f"{g('backend') or '':8}",
            f"{str(g('ep') or ''):4}",
            f"{str(g('pp') or ''):4}",
            f"{g('compile') or '':8}",
            f"{g('ac') or '':6}",
            f"{str(g('bs') or ''):4}",
            f"{g('status') or '':8}",
            _fmt(g("tflops"), ".1f", "", "OOM" if g("status") == "oom" else "—").rjust(10),
            _fmt(g("mfu_pct"), ".2f", "%", "OOM" if g("status") == "oom" else "—").rjust(10),
            _fmt(g("tps"), ",.0f", "", "OOM" if g("status") == "oom" else "—").rjust(12),
            _fmt(g("mem_gib"), ".2f", " GiB", "—").rjust(10),
            _fmt(g("mem_pct"), ".1f", "%", "—").rjust(10),
            f"{str(g('n_iters') or ''):7}",
            f"{g('job_id') or '':10}",
        ]
        print("  ".join(vals))


def _print_csv(rows):
    import csv

    fields = [
        "name",
        "backend",
        "ep",
        "pp",
        "compile",
        "ac",
        "bs",
        "status",
        "tflops",
        "mfu_pct",
        "tps",
        "mem_gib",
        "mem_pct",
        "n_iters",
        "job_id",
    ]
    w = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
