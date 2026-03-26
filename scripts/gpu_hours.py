#!/usr/bin/env python3
"""
Calculate GPU hours for all experiments in a results directory.
Scans for slurm-<jobid>.log files, queries sacct, and reports per-job
and per-experiment GPU-hours.

Usage:
    python gpu_hours.py [results_dir]

Defaults to the current directory if no argument is given.
"""

import os
import re
import sys
import subprocess
from collections import defaultdict


def parse_elapsed(s):
    """Convert sacct elapsed string (D-HH:MM:SS or HH:MM:SS) to hours."""
    if "-" in s:
        days, rest = s.split("-", 1)
        h, m, sec = rest.split(":")
        return int(days) * 24 + int(h) + int(m) / 60 + int(sec) / 3600
    h, m, sec = s.split(":")
    return int(h) + int(m) / 60 + int(sec) / 3600


def collect_job_ids(results_dir):
    """Return {experiment_name: [job_id, ...]} by scanning slurm-*.log files."""
    experiments = defaultdict(list)
    for exp_name in sorted(os.listdir(results_dir)):
        exp_path = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        for fname in sorted(os.listdir(exp_path)):
            m = re.match(r"slurm-(\d+)\.log", fname)
            if m:
                experiments[exp_name].append(m.group(1))
    return experiments


def query_sacct(job_ids):
    """
    Run sacct for the given job IDs.
    Returns {job_id: {"state": ..., "elapsed": ..., "gpus": int}}.
    Only the top-level job entry (no .batch / .extern / .N steps) is kept.
    """
    ids_str = ",".join(job_ids)
    cmd = [
        "sacct", "-j", ids_str,
        "--format=JobID,State,Elapsed,AllocTRES%80",
        "--noheader", "--parsable2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sacct error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    info = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        job_id_field, state, elapsed, alloc_tres = parts[:4]
        # Skip sub-steps (.batch, .extern, .0, .1, ...)
        if "." in job_id_field:
            continue
        job_id = job_id_field.strip()
        # Parse GPU count from TRES string, e.g. "gres/gpu=32"
        gpus = 0
        m = re.search(r"gres/gpu=(\d+)", alloc_tres)
        if m:
            gpus = int(m.group(1))
        info[job_id] = {
            "state": state.strip(),
            "elapsed": elapsed.strip(),
            "gpus": gpus,
        }
    return info


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    results_dir = os.path.abspath(results_dir)

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
    col_state = 12
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

    for exp_name, job_ids in experiments.items():
        exp_total = 0.0
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
            gpu_h = hours * d["gpus"]
            exp_total += gpu_h
            print(
                f"{exp_name:<{col_exp}}  "
                f"{job_id:>{col_job}}  "
                f"{d['state']:>{col_state}}  "
                f"{d['elapsed']:>{col_ela}}  "
                f"{d['gpus']:>{col_gpu}}  "
                f"{gpu_h:>{col_gpuh}.1f}"
            )
            exp_name = ""  # only print name on first row of each experiment

        exp_totals[exp_name] = exp_total
        grand_total += exp_total

    # ---- per-experiment summary ----
    print(sep)
    print(f"\n{'Experiment summary':}")
    print(sep)
    for exp_name, job_ids in experiments.items():
        exp_gpu_h = sum(
            parse_elapsed(sacct_info[jid]["elapsed"]) * sacct_info[jid]["gpus"]
            for jid in job_ids if jid in sacct_info
        )
        print(f"  {exp_name:<{col_exp - 2}}  {exp_gpu_h:>8.1f} GPU-h")

    print(sep)
    print(f"  {'TOTAL':<{col_exp - 2}}  {grand_total:>8.1f} GPU-h")
    print(sep)


if __name__ == "__main__":
    main()
