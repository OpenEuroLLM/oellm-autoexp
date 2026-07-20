#!/usr/bin/env python3
"""
Calculate GPU hours for all experiments in a results directory.
Scans for slurm-<jobid>.log files, queries sacct, and reports per-job
and per-experiment GPU-hours.

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

# LUMI-G Slurm partitions billed under the GCD-granular formula (see
# https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/#gpu-billing).
# Any other GPU partition (standard-g, or a non-LUMI cluster's GPU partition)
# falls back to the "whole module" rate.
_LUMI_GCD_BILLED_PARTITIONS = {"small-g", "dev-g"}


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
    """
    Run sacct for the given job IDs.
    Returns {job_id: {"state", "elapsed", "partition", "gpus", "cpus", "mem_gb", "is_lumi_gpu"}}.
    Only the top-level job entry (no .batch / .extern / .N steps) is kept.
    """
    ids_str = ",".join(job_ids)
    cmd = [
        "sacct", "-j", ids_str,
        "--format=JobID,State,Elapsed,Partition,AllocTRES%200",
        "--noheader", "--parsable2",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
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
        # Parse GPU count from TRES string, e.g. "gres/gpu=32" or LUMI's
        # type-qualified "gres/gpu:mi250=32". On LUMI this count is GCDs,
        # not physical GPU modules (each MI250x module = 2 GCDs).
        gpus = 0
        is_lumi_gpu = False
        m = re.search(r"gres/gpu(?::(\w+))?=(\d+)", alloc_tres)
        if m:
            gpus = int(m.group(2))
            is_lumi_gpu = bool(m.group(1)) and "mi250" in m.group(1).lower()
        cpus = 0
        m = re.search(r"(?:^|,)cpu=(\d+)", alloc_tres)
        if m:
            cpus = int(m.group(1))
        mem_gb = 0.0
        m = re.search(r"(?:^|,)mem=([\d.]+)([KMGTP]?)", alloc_tres)
        if m:
            mem_gb = parse_mem_gb(m.group(1), m.group(2))
        info[job_id] = {
            "state": state.strip(),
            "elapsed": elapsed.strip(),
            "partition": partition.strip(),
            "gpus": gpus,
            "cpus": cpus,
            "mem_gb": mem_gb,
            "is_lumi_gpu": is_lumi_gpu,
        }
    return info


def main():
    parser = argparse.ArgumentParser(description="Calculate GPU hours for all experiments in a results directory.")
    parser.add_argument("results_dir", nargs="?", default=".",
                        help="Directory containing experiment subdirectories (default: current directory)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV file (default: <results_dir>/gpu_hours.csv)")
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
    col_exp   = max(len(e) for e in experiments) + 2
    col_job   = 12
    col_state = 20
    col_ela   = 14
    col_gpu   = 6
    col_gpuh  = 8

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
            gpu_h = gpu_hours_for_job(d, hours)
            exp_total += gpu_h
            csv_rows.append({
                "experiment": exp_name,
                "job_id": job_id,
                "state": d["state"],
                "elapsed": d["elapsed"],
                "gpus": d["gpus"],
                "gpu_hours": round(gpu_h, 1),
            })
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
            gpu_hours_for_job(sacct_info[jid], parse_elapsed(sacct_info[jid]["elapsed"]))
            for jid in job_ids if jid in sacct_info
        )
        print(f"  {exp_name:<{col_exp - 2}}  {exp_gpu_h:>8.1f} GPU-h")

    print(sep)
    print(f"  {'TOTAL':<{col_exp - 2}}  {grand_total:>8.1f} GPU-h")
    print(sep)

    # ---- save to CSV ----
    csv_rows.append({
        "experiment": "TOTAL",
        "job_id": "",
        "state": "",
        "elapsed": "",
        "gpus": "",
        "gpu_hours": round(grand_total, 1),
    })
    output_csv = args.output if args.output else os.path.join(results_dir, "gpu_hours.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "job_id", "state", "elapsed", "gpus", "gpu_hours"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nWrote {len(csv_rows) - 1} job rows (+1 total) to {output_csv}")


if __name__ == "__main__":
    main()
