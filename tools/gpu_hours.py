#!/usr/bin/env python3
"""Calculate GPU hours for all experiments in a results directory. Scans for
slurm-<jobid>.log files, queries sacct, and reports per-job and per-experiment
GPU-hours.

Usage:
    python gpu_hours.py [results_dir]

Defaults to the current directory if no argument is given.
"""

import argparse
import csv
import math
import os
import re
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

# Make sibling modules importable when run as a standalone script.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from write_guard import guard_write  # noqa: E402


# Slurm counts GPUs as Graphics Compute Dies, but LUMI bills per physical
# MI250x module, which carries 2 GCDs.  A standard-g node therefore appears to
# Slurm as 8 GPUs while being billed as 4 GPU-hours per node-hour.  Reporting
# the raw GCD count doubles every GPU-hour figure relative to the allocation.
# See https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#gpu-billing
#
# Keyed by the model qualifier in the sacct TRES string ("gres/gpu:mi250=64").
# Clusters that report a bare "gres/gpu=" (MN5, Leonardo) have no qualifier and
# bill 1:1, so they are unaffected.
GPU_BILLING_DIVISOR: dict[str, float] = {
    "mi250": 2.0,
    "mi250x": 2.0,
}


def gpu_billing_divisor(model: str) -> float:
    """GCDs per billed GPU for *model* (1.0 when the GPU bills 1:1)."""
    return GPU_BILLING_DIVISOR.get((model or "").strip().lower(), 1.0)


def parse_elapsed(s):
    """Convert sacct elapsed string (D-HH:MM:SS or HH:MM:SS) to hours."""
    if "-" in s:
        days, rest = s.split("-", 1)
        h, m, sec = rest.split(":")
        return int(days) * 24 + int(h) + int(m) / 60 + int(sec) / 3600
    h, m, sec = s.split(":")
    return int(h) + int(m) / 60 + int(sec) / 3600


def parse_mem_gb(value, unit):
    """Convert a sacct AllocTRES mem quantity (e.g. "480", "G") to GiB."""
    factor = {"": 1 / 1024, "K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1, "T": 1024, "P": 1024 ** 2}
    return float(value) * factor.get(unit, 1 / 1024)


def gpu_hours_for_job(d, hours):
    """Billed GPU-hours for one job, per LUMI's billing policy.

    A GPU-hour is a full MI250x module (2 GCDs) for one hour. On standard-g
    (always full-node) it's simply GCDs/2 * hours. On small-g/dev-g, billing
    is 0.5 per GCD allocated, unless CPU or memory allocated per GCD exceeds
    8 cores / 64GB, in which case that share is billed instead:
        GPU-h = max(ceil(cpu/8), ceil(mem_GB/64), GCDs) * hours * 0.5
    Non-LUMI GPUs (no "mi250" TRES type) are billed 1:1 (gpus * hours), since
    sacct's gres/gpu there already counts physical GPUs, not GCDs.
    """
    if not d.get("is_lumi_gpu"):
        return d["gpus"] * hours

    gcds = d["gpus"]
    if d.get("partition") in _LUMI_GCD_BILLED_PARTITIONS:
        units = max(math.ceil(d["cpus"] / 8), math.ceil(d["mem_gb"] / 64), gcds)
        return units * hours * 0.5
    return (gcds / 2) * hours


def collect_job_ids(results_dir):
    """Return {experiment_name: [job_id, ...]} by scanning for log files.

    Supports two layouts:
      - <results_dir>/<exp_name>/slurm-JOBID.log
      - <results_dir>/.../logs/stderr-JOBID.log  (stdout-JOBID.log shares the same ID)
    """
    experiments = defaultdict(set)  # set deduplicates stderr/stdout entries for the same job

    for root, dirs, files in os.walk(results_dir):
        dirs.sort()
        rel_root = os.path.relpath(root, results_dir)

        for fname in sorted(files):
            # Layout 1: slurm-JOBID.log anywhere under an experiment subdir
            m = re.match(r"slurm-(\d+)\.log", fname)
            if m and rel_root != ".":
                exp_name = rel_root.split(os.sep)[0]
                experiments[exp_name].add(m.group(1))
                continue

            # Layout 2: stderr-JOBID.log inside a logs/ subdir
            m = re.match(r"stderr-(\d+)\.log", fname)
            if m and os.path.basename(root) == "logs":
                parent = os.path.dirname(root)
                exp_name = os.path.relpath(parent, results_dir)
                experiments[exp_name].add(m.group(1))

    return {k: sorted(v) for k, v in sorted(experiments.items())}


def query_sacct(job_ids):
    """Run sacct for the given job IDs.

    Returns {job_id: {"state": ..., "elapsed": ..., "gpus": int}}.
    Only the top-level job entry (no .batch / .extern / .N steps) is kept.
    """
    ids_str = ",".join(job_ids)
    cmd = [
        "sacct",
        "-j",
        ids_str,
        "--format=JobID,State,Elapsed,Partition,AllocTRES%80",
        "--noheader",
        "--parsable2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sacct error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    info = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id_field, state, elapsed, partition, alloc_tres = parts[:5]
        # Skip sub-steps (.batch, .extern, .0, .1, ...)
        if "." in job_id_field:
            continue
        job_id = job_id_field.strip()
        # Parse GPU count from TRES string, e.g. "gres/gpu=32" (MN5/Leonardo)
        # or "gres/gpu:mi250=64" (LUMI, which qualifies the resource with the
        # GPU model).  The optional ":<model>" is what makes LUMI jobs report
        # zero GPUs — and therefore zero GPU-hours — without it.
        gpus = 0
        m = re.search(r"gres/gpu(?::([^=]+))?=(\d+)", alloc_tres)
        model = ""
        if m:
            model = (m.group(1) or "").strip().lower()
            gpus = int(m.group(2))
        info[job_id] = {
            "state": state.strip(),
            "elapsed": elapsed.strip(),
            "partition": partition.strip(),
            "gpus": gpus,
            "gpus_billed": gpus / gpu_billing_divisor(model),
            "gpu_model": model,
        }
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Calculate GPU hours for all experiments in a results directory."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=".",
        help="Directory containing experiment subdirectories (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV file (default: <results_dir>/gpu_hours.csv)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    experiments = collect_job_ids(results_dir)
    if not experiments:
        print("No slurm-*.log files found.", file=sys.stderr)
        sys.exit(1)

    all_job_ids = [jid for jids in experiments.values() for jid in jids]
    sacct_info = query_sacct(all_job_ids)

    # ---- per-job table ----
    col_exp = max(len(e) for e in experiments) + 2
    col_job = 12
    col_state = 20
    col_ela = 14
    col_gpu = 6
    col_gpuh = 8

    header = (
        f"{'Experiment':<{col_exp}}  "
        f"{'JobID':>{col_job}}  "
        f"{'State':>{col_state}}  "
        f"{'Elapsed':>{col_ela}}  "
        f"{'GPUs':>{col_gpu}}  "
        f"{'GPU-h':>{col_gpuh}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    grand_total = 0.0
    exp_totals = {}
    csv_rows = []

    for exp_name, job_ids in experiments.items():
        exp_total = 0.0
        display_name = exp_name  # blanked out after first row for the terminal table only
        for job_id in job_ids:
            if job_id not in sacct_info:
                print(
                    f"{'':>{col_exp}}  {job_id:>{col_job}}  "
                    f"{'NOT FOUND':>{col_state}}  {'':>{col_ela}}  "
                    f"{'':>{col_gpu}}  {'':>{col_gpuh}}"
                )
                continue
            d = sacct_info[job_id]
            hours = parse_elapsed(d["elapsed"])
            gpu_h = hours * d["gpus_billed"]
            exp_total += gpu_h
            csv_rows.append(
                {
                    "experiment": exp_name,
                    "job_id": job_id,
                    "state": d["state"],
                    "elapsed": d["elapsed"],
                    "gpus": d["gpus"],
                    "gpu_hours": round(gpu_h, 1),
                }
            )
            print(
                f"{display_name:<{col_exp}}  "
                f"{job_id:>{col_job}}  "
                f"{d['state']:>{col_state}}  "
                f"{d['elapsed']:>{col_ela}}  "
                f"{d['gpus']:>{col_gpu}}  "
                f"{gpu_h:>{col_gpuh}.1f}"
            )
            display_name = ""  # only print name on first row of each experiment

        exp_totals[exp_name] = exp_total
        grand_total += exp_total

    # ---- per-experiment summary ----
    print(sep)
    print(f"\n{'Experiment summary':}")
    print(sep)
    for exp_name, job_ids in experiments.items():
        exp_gpu_h = sum(
            parse_elapsed(sacct_info[jid]["elapsed"]) * sacct_info[jid]["gpus_billed"]
            for jid in job_ids
            if jid in sacct_info
        )
        print(f"  {exp_name:<{col_exp - 2}}  {exp_gpu_h:>8.1f} GPU-h")

    print(sep)
    print(f"  {'TOTAL':<{col_exp - 2}}  {grand_total:>8.1f} GPU-h")
    print(sep)

    # ---- save to CSV ----
    csv_rows.append(
        {
            "experiment": "TOTAL",
            "job_id": "",
            "state": "",
            "elapsed": "",
            "gpus": "",
            "gpu_hours": round(grand_total, 1),
        }
    )
    output_csv = args.output if args.output else os.path.join(results_dir, "gpu_hours.csv")
    guard_write(output_csv)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "job_id", "state", "elapsed", "gpus", "gpu_hours"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {len(csv_rows) - 1} job rows (+1 total) to {output_csv}")


if __name__ == "__main__":
    main()
