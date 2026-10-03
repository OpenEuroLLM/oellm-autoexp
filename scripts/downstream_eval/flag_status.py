#!/usr/bin/env python3
# Usage:
#   flag_status.py STATE_FILE             per-task status of both halves; collect when complete
#   flag_status.py STATE_FILE --rerun     resubmit failed / never-run tasks of both halves
#   flag_status.py STATE_FILE --collect   collect now, even if incomplete (CSV is then partial)
#
# A task's state is that of its LATEST attempt (highest job id among its logs):
#   ok       log ends with "finished" and has no "[error] Evaluation failed"
#   failed   "[error] Evaluation failed" in the log, or the attempt left the queue unfinished
#   active   queued (pending or running) under the run's job or one of its reruns
#   missing  never started and not queued
# Reruns go to the same run directory, so results accumulate there and the collector keeps the
# newest result per eval. Environment: FLAG_ACCOUNT, COLLECT_RAW, FLAG_DATASETS_CSV, FLAG_WORK,
# DRY_RUN, FLAG_NO_QUEUE (1 = do not ask squeue; set by flag_stage.sh).
"""Status, reruns and collection of one FLAG-suite run (see flag_launch.sh)."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

LOG_RE = re.compile(r"^oellm-eval-(\d+)-(\d+)\.out$")
REASONS = [  # first match wins; what the failure means is in flag-evals' README section of this repo
    ("OfflineModeIsEnabled", "dataset missing from hf_home"),
    ("outer deadline", "HumanEval grader deadline (unpatched grader)"),
    ("failed to exit cleanly", "vLLM DP worker shutdown"),
    ("Cannot close a process", "vLLM DP worker shutdown"),
    ("killed after 1m0s timeout", "container start-up timeout"),
    ("context deadline exceeded", "container start-up timeout"),
    ("CUDA out of memory", "CUDA OOM"),
    (
        "CalledProcessError",
        "grading subprocess failed (HumanEval: see humaneval_artifacts/*/grading_*.log)",
    ),
    ("CANCELLED DUE TO TIME LIMIT", "time limit (hung?)"),
    ("CANCELLED AT", "cancelled"),
    ("Terminated", "killed (scancel or node failure; check whether it hung)"),
]


def read_state(path: Path) -> dict[str, str]:
    state = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            state[key] = value
    return state


def queued_tasks() -> set[tuple[str, int]]:
    """(job id, array index) of every queued task of this user, pending ranges
    expanded."""
    if os.environ.get("FLAG_NO_QUEUE") == "1":  # inside a flag_stage.sh allocation
        return set()
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-r", "-o", "%i"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"error: squeue failed ({e}); cannot tell running from failed tasks")
    tasks = set()
    for tok in out.split():
        job, _, idx = tok.partition("_")
        if idx.isdigit():
            tasks.add((job, int(idx)))
    return tasks


def task_names(run_dir: Path) -> list[str]:
    with open(run_dir / "jobs.csv") as f:
        return [f"{row['task_path']} ({row['n_shot']}-shot)" for row in csv.DictReader(f)]


def half_status(run_dir: Path, job_ids: set[str], queue: set[tuple[str, int]]):
    names = task_names(run_dir)
    latest: dict[int, str] = {}
    for p in (run_dir / "slurm_logs").glob("oellm-eval-*.out"):
        m = LOG_RE.match(p.name)
        if m:
            job, idx = m.group(1), int(m.group(2))
            if idx not in latest or int(job) > int(latest[idx]):
                latest[idx] = job
    queued_idx = {i for (j, i) in queue if j in job_ids}
    status, reason = {}, {}
    for i in range(len(names)):
        job = latest.get(i)
        if i in queued_idx:  # queued under the run's job or one of its reruns
            status[i] = "active"
            continue
        if job is None:
            status[i] = "missing"
            continue
        out = run_dir / "slurm_logs" / f"oellm-eval-{job}-{i}.out"
        err = out.with_suffix(".err")
        text = out.read_text(errors="replace") + (
            err.read_text(errors="replace") if err.exists() else ""
        )
        if "[error] Evaluation failed" in text:
            status[i] = "failed"
        elif re.search(r"finished\.?\s*$", out.read_text(errors="replace").strip()):
            status[i] = "ok"
        else:
            status[i] = "failed"
        if status[i] == "failed":
            reason[i] = next((why for pat, why in REASONS if pat in text), "unknown (see log)")
    return names, status, reason


def submit_rerun(sbatch_file: Path, indices: list[int]) -> str:
    args = [
        "sbatch",
        "--parsable",
        f"--account={os.environ['FLAG_ACCOUNT']}",
        f"--array={','.join(map(str, indices))}%10",
        str(sbatch_file),
    ]
    if os.environ.get("DRY_RUN") == "1":
        print("  " + " ".join(args))
        return "DRYRUN"
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def join_check(csv_path: Path, datasets_csv: Path) -> tuple[int, int, list[str]]:
    """How many of the suite's evals the collected CSV covers, on the
    collector's join key."""

    def key(task, n_shot, metric, filt):
        return (task, str(n_shot), metric, filt or "")

    with open(csv_path) as f:
        have = {key(r["task"], r["n_shot"], r["metric"], r["filter"]) for r in csv.DictReader(f)}
    with open(datasets_csv) as f:
        want = [
            key(r["task"], r["n_shot"], r["suite_metric"], r["suite_filter"])
            for r in csv.DictReader(f)
        ]
    missing = [f"{k[0]} ({k[1]}-shot, {k[2]})" for k in want if k not in have]
    return len(want) - len(missing), len(want), missing


def collect(state: dict[str, str]) -> None:
    out_dir = Path(os.environ["FLAG_WORK"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{state['NAME']}.flag-evals-438.tasks.csv"
    datasets = Path(os.environ["FLAG_DATASETS_CSV"])
    subprocess.run(
        [
            sys.executable,
            os.environ["COLLECT_RAW"],
            "--checkpoints",
            state["NAME"],
            "--runs",
            state["VLLM_RUN"],
            state["LIGHTEVAL_RUN"],
            "--chrf-patched",
            "--datasets",
            str(datasets),
            "-o",
            str(out),
        ],
        check=True,
    )
    joined, total, missing = join_check(out, datasets)
    print(
        f"collected: {out}\n  joined {joined}/{total} evals"
        + ("  -- COMPLETE" if joined == total else "")
    )
    for m in missing[:20]:
        print(f"    missing: {m}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("state", type=Path)
    ap.add_argument("--rerun", action="store_true", help="resubmit failed and never-run tasks")
    ap.add_argument("--collect", action="store_true", help="collect even if not complete")
    args = ap.parse_args()

    state = read_state(args.state)
    queue = queued_tasks()
    rerun_lines, complete = [], True
    print(f"{state['NAME']}  (eval rev {state.get('EVAL_REV', '?')[:9]})")
    for half in ("VLLM", "LIGHTEVAL"):
        run_dir = Path(state[f"{half}_RUN"])
        job_ids = (
            set(state[f"{half}_JOB"].split(","))
            | set(state.get(f"{half}_RERUN_JOBS", "").split(","))
        ) - {""}
        names, status, reason = half_status(run_dir, job_ids, queue)
        counts = {
            s: sum(1 for v in status.values() if v == s)
            for s in ("ok", "active", "failed", "missing")
        }
        print(
            f"  {half.lower():9s} {counts['ok']}/{len(names)} ok, {counts['active']} active, "
            f"{counts['failed']} failed, {counts['missing']} missing   {run_dir}"
        )
        todo = sorted(i for i, s in status.items() if s in ("failed", "missing"))
        for i in todo:
            print(
                f"    [{i}] {names[i]}: {status[i]}" + (f" -- {reason[i]}" if i in reason else "")
            )
        complete &= counts["ok"] == len(names)
        if todo and args.rerun:
            job = submit_rerun(run_dir / "submit_evals.sbatch", todo)
            rerun_lines.append(
                f"{half}_RERUN_JOBS={','.join(filter(None, [state.get(f'{half}_RERUN_JOBS', ''), job]))}"
            )
            print(f"    resubmitted {len(todo)} task(s) as job {job}")
        elif todo:
            print(f"    rerun: eval_checkpoints.sh flag-rerun {state['NAME']}")
    if rerun_lines and os.environ.get("DRY_RUN") != "1":
        with open(args.state, "a") as f:
            f.write("\n".join(rerun_lines) + "\n")
    if complete or args.collect:
        collect(state)


if __name__ == "__main__":
    main()
