#!/usr/bin/env python3
"""Convenience wrapper to plan, submit, and optionally monitor in one shot."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from uuid import uuid4

from compoconf import asdict

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.orchestrator import (
    build_execution_plan,
    ExecutionPlan,
    render_job_scripts,
    submit_jobs,
    chain_submit_jobs,
    run_loop,
    stop_path,
)
from oellm_autoexp.utils.logging_config import configure_logging


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="autoexp")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan and render without submitting jobs"
    )
    parser.add_argument("--no-monitor", action="store_true", help="Submit jobs but skip monitoring")
    parser.add_argument(
        "--submit-and-exit",
        action="store_true",
        help="Submit jobs to SLURM then exit immediately (no monitoring loop)",
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help=(
            "Submit all jobs to Slurm immediately as a dependency chain "
            "(job N+1 gets --dependency=afterok:jobN), so they are pre-queued "
            "and benefit from scheduling priority."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="With --chain: submit each job N times sequentially (useful for wall-time chaining).",
    )
    parser.add_argument(
        "--monitor-state-dir",
        default="./monitor_state",
        type=Path,
        help="Monitoring state directory",
    )
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--array-subset",
        type=str,
        help="Comma-separated sweep indices or ranges (e.g., '0,3-5') to rerun.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of submitting to SLURM (uses LocalCommandClient)",
    )
    parser.add_argument(
        "overrides", nargs="*", default=[], help="Hydra-style overrides (`key=value`)."
    )
    return parser.parse_args(argv)


def _default_manifest_path(base_output_dir: str | Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest_dir = Path(base_output_dir) / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex[:6]
    return manifest_dir / f"plan_{timestamp}_{suffix}.json"


def _collect_git_metadata(repo_root: Path) -> dict[str, str | bool]:
    def _run(cmd: Iterable[str]) -> str:
        try:
            result = subprocess.run(
                list(cmd),
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return ""
        return result.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"]) or "unknown"
    status = _run(["git", "status", "--porcelain"])
    dirty = bool(status)
    diff = _run(["git", "diff"]) if dirty else ""
    return {
        "commit": commit,
        "dirty": dirty,
        "status": status,
        "diff": diff,
    }


def _sanitize_env() -> dict[str, str]:
    pattern = re.compile(r"(KEY|SECRET)", re.IGNORECASE)
    return {key: value for key, value in os.environ.items() if not pattern.search(key)}


def _render_log_hint(log_template: str | Path, job_id: str) -> str:
    """Expand SLURM log templates (%j, %A, %a) using the submitted job id."""
    log_str = str(log_template)
    if "_" in job_id:
        base_id, array_idx = job_id.split("_", 1)
        log_str = log_str.replace("%A", base_id)
        log_str = log_str.replace("%a", array_idx)
    return log_str.replace("%j", job_id)


def _write_job_provenance(
    plan: ExecutionPlan,
    *,
    args: argparse.Namespace | None = None,
    subset_indices: set[int] | None = None,
    overrides: list[str] = (),
) -> None:
    git_meta = _collect_git_metadata(REPO_ROOT)
    sanitized_env = _sanitize_env()
    base_payload = {
        "git": git_meta,
        "command": {key: str(val) for key, val in vars(args).items()} or list(sys.argv),
        "overrides": overrides,
        "subset_indices": sorted(subset_indices),
        "plan": asdict(plan),
        "environment": sanitized_env,
    }

    manifest_path = _default_manifest_path(plan.config_setup.monitor_state_dir)
    with open(manifest_path, "w") as fp:
        json.dump(base_payload, fp)


def _parse_slurm_time(time_str: str) -> float:
    """Parse a SLURM time string (D-HH:MM:SS or HH:MM:SS) and return hours."""
    time_str = time_str.strip()
    days = 0
    if "-" in time_str:
        day_part, time_str = time_str.split("-", 1)
        days = int(day_part)
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        h, m, s = int(parts[0]), 0, 0
    return days * 24 + h + m / 60 + s / 3600


def _fmt_hm(hours: float) -> str:
    """Hours as a compact human duration: 0.14 -> '8m', 11.5 -> '11h30m'."""
    if hours <= 0:
        return "0m"
    total_min = int(round(hours * 60))
    h, m = divmod(total_min, 60)
    if h and m:
        return f"{h}h{m:02d}m"
    return f"{h}h" if h else f"{m}m"


def _derive_steps(job_config) -> int | None:
    """How many iterations one segment of this job will run, from the backend.

    Order matters and mirrors what actually stops a Megatron run first:
    `exit_interval` ends it after N iterations regardless of the schedule, so it
    wins; otherwise the schedule length bounds it (`train_iters`, or
    `train_samples / global_batch_size` for the sample-based configs this repo
    uses). Returns None when nothing says — a bash probe or a `skip_train` job
    has no steps, and the caller then falls back to the wall clock.
    """
    mg = getattr(getattr(job_config, "backend", None), "megatron", None)
    if mg is None:
        return None
    exit_interval = getattr(mg, "exit_interval", None)
    if exit_interval:
        return int(exit_interval)
    train_iters = getattr(mg, "train_iters", None)
    if train_iters:
        return int(train_iters)
    train_samples = getattr(mg, "train_samples", None)
    gbs = getattr(mg, "global_batch_size", None)
    if train_samples and gbs:
        return int(train_samples) // int(gbs)
    return None


def _full_run_hours(job) -> tuple[float, int] | None:
    """Total stepping hours for this job's WHOLE schedule, ignoring the wall.

    Returns (hours, steps), or None when no measured step time was given. This
    is the number that answers "how long until 15 T tokens are done", which the
    per-segment figure cannot: a production run is capped at its 12 h wall and
    resumes from a checkpoint, so the segment length says nothing about the
    finish date.
    """
    jc = job.config.job
    step_time = getattr(jc, "est_step_time_s", None)
    if not step_time:
        return None
    steps = getattr(jc, "est_steps", None) or _derive_steps(job.config)
    if not steps:
        return None
    startup_h = float(getattr(jc, "est_startup_min", 0.0) or 0.0) / 60.0
    return startup_h + int(steps) * float(step_time) / 3600.0, int(steps)


def _job_segment_hours(job) -> tuple[float, str]:
    """Wall-clock hours ONE allocation of this job will actually consume.

    Returns (hours, basis) where basis names whichever limit bound it,
    so the gate can say what it assumed instead of quoting a bare
    number.
    """
    sbatch = job.config.slurm.sbatch
    time_str = str(getattr(sbatch, "time", "1:00:00") or "1:00:00")
    try:
        wall = _parse_slurm_time(time_str)
    except (ValueError, IndexError):
        wall = 0.0

    limits: list[tuple[float, str]] = [(wall, "--time")]

    # Megatron's own self-imposed deadline, when it is tighter than the wall
    # (production sets 690 min against a 12 h limit to leave room for the final
    # checkpoint write).
    mg = getattr(getattr(job.config, "backend", None), "megatron", None)
    exit_mins = getattr(mg, "exit_duration_in_mins", None) if mg is not None else None
    if exit_mins:
        limits.append((float(exit_mins) / 60.0, "exit_duration_in_mins"))

    jc = job.config.job
    step_time = getattr(jc, "est_step_time_s", None)
    if step_time:
        steps = getattr(jc, "est_steps", None) or _derive_steps(job.config)
        if steps:
            startup_h = float(getattr(jc, "est_startup_min", 0.0) or 0.0) / 60.0
            limits.append((startup_h + steps * float(step_time) / 3600.0, "est_step_time_s"))

    return min(limits, key=lambda kv: kv[0])


def _job_segments(job, repeat: int = 1) -> tuple[int, str]:
    """How many allocations this job really takes, and why.

    A training run does not stop when the queued jobs run out — it exits at the
    wall, gets restarted (by the monitor, by the chain, or by hand) and resumes
    from its checkpoint until the SCHEDULE is done. So the number of allocations
    is driven by the schedule length, not by how many happen to be queued now.
    Pricing only what is queued understated the 512-node production run by 31x:
    3 segments quoted, ~93 actually needed.

    Deliberately rough. It ignores per-segment startup, checkpoint-reload cost
    and crash restarts, so it is a floor — the point is to be the right ORDER OF
    MAGNITUDE at the gate, not to predict a finish date.
    """
    effective_repeat = (
        repeat if repeat > 1 else int(getattr(job.config.job, "chain_repeat", 1) or 1)
    )
    segment_h, _ = _job_segment_hours(job)
    full = _full_run_hours(job)
    if full and segment_h > 0 and full[0] > segment_h:
        return max(effective_repeat, math.ceil(full[0] / segment_h)), "to finish the schedule"
    return effective_repeat, "queued"


def _compute_gpu_hours(plan: ExecutionPlan, repeat: int = 1) -> tuple[float, list[str]]:
    """Sum GPU-hours across all jobs in the plan, returning (total, notes).

    THREE THINGS THIS GETS RIGHT THAT THE OLD VERSION DID NOT:

      * THE RUN RESTARTS UNTIL THE SCHEDULE IS DONE. See _job_segments.
      * CHAIN REPEATS ARE REAL ALLOCATIONS. `plan.jobs` holds one entry per
        sweep point; `chain_repeat` is expanded later, in orchestrator.py's
        `effective_repeat` loop, into R submitted jobs. Pricing only the plan
        understated an 8-draw node_catch campaign by 8x (quoted 683 GPU-h,
        really up to 5,464).
      * A JOB DOES NOT ALWAYS RUN TO ITS WALL CLOCK. `exit_interval` /
        `exit_duration_in_mins` stop it earlier, and with a measured
        `job.est_step_time_s` that is now priced instead of the wall.
    """
    total = 0.0
    bases: set[str] = set()
    for job in plan.jobs:
        sbatch = job.config.slurm.sbatch
        nodes = int(getattr(sbatch, "nodes", 1) or 1)
        gpus_per_node = int(getattr(sbatch, "gpus_per_node", 1) or 1)
        hours, basis = _job_segment_hours(job)
        segments, why = _job_segments(job, repeat)
        bases.add(basis)
        if segments > 1:
            bases.add(f"x{segments} segments {why}")
        total += hours * nodes * gpus_per_node * segments
    return total, sorted(bases)


def _parse_subset(spec: str | None) -> set[int]:
    indices: set[int] = set()
    if not spec:
        return indices
    for token in spec.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"invalid range '{part}'")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def _resume_command(args, session_id: str) -> str:
    """Command that re-attaches a monitor to an already-submitted session.

    monitor_autoexp.py re-reads the per-job state files in the session
    directory, so resuming does NOT resubmit anything -- it picks the
    existing SLURM job ids back up and keeps applying the same log-event
    policy.
    """
    parts = ["python scripts/monitor_autoexp.py", f"--session {session_id}"]
    state_dir = str(getattr(args, "monitor_state_dir", "") or "")
    # only worth printing when it is not the default the script already assumes
    if state_dir and state_dir not in ("./monitor_state", "monitor_state"):
        parts.append(f"--monitor-state-dir {state_dir}")
    return " ".join(parts)


def _print_resume_hint(resume_cmd: str, res, *, reason: str) -> None:
    """Tell the user how to get monitoring back.

    Cheap to print, expensive to lose.
    """
    n = len(getattr(res, "submitted_job_ids", []) or [])
    bar = "=" * 78
    print(f"\n{bar}")
    print(f"  {n} job(s) submitted, session {res.session_id}  [{reason}]")
    print("  To continue monitoring these runs, run from the repo root:")
    print(f"      {resume_cmd}")
    print(f"{bar}\n", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(not args.no_verbose, args.debug)

    config_dir = Path(args.config_dir)

    overrides = list(args.overrides)
    if args.local:
        # Force single-node torchrun
        overrides = ["++slurm.sbatch.nodes=1"] + overrides
        # Auto-detect GPU count unless already overridden
        if not any("gpus_per_node" in o for o in overrides):
            try:
                import torch

                n_gpus = torch.cuda.device_count() or 1
            except Exception:
                n_gpus = 1
            overrides = [f"++slurm.sbatch.gpus_per_node={n_gpus}"] + overrides
        # Default output to ./output unless already overridden
        if not any("job.base_output_dir" in o for o in overrides):
            overrides = ["++job.base_output_dir=./output"] + overrides

    config_setup = ConfigSetup(
        pwd=os.path.abspath(os.curdir),
        config_name=args.config_name,
        config_dir=str(config_dir),
        overrides=overrides,
        monitor_state_dir=str(args.monitor_state_dir),
    )
    root = load_config_reference(config_setup=config_setup)

    try:
        subset_indices = _parse_subset(args.array_subset)
    except ValueError as exc:
        print(f"Invalid --array-subset argument: {exc}", file=sys.stderr)
        return

    plan = build_execution_plan(
        root,
        config_setup=config_setup,
        subset_indices=subset_indices or None,
    )

    # Cost estimate BEFORE the dry-run branch, so `--dry-run` shows it too.
    # It used to sit after, which made the estimate unreachable in exactly the
    # mode you would use to decide whether a campaign is affordable — and left
    # the `not args.dry_run` guard on the prompt below as dead code.
    gpu_hours, est_bases = _compute_gpu_hours(plan, repeat=getattr(args, "repeat", 1) or 1)
    n_jobs = len(plan.jobs)
    # Name the binding assumption. "assuming full slurm time" was a lie whenever
    # exit_interval or a measured step time was in play, and said nothing about
    # chain repeats at all.
    basis = ", ".join(est_bases) if est_bases else "--time"
    print(
        f"Plan: {n_jobs} job(s) — estimated {gpu_hours:,.0f} GPU-h total ({basis})"
        + (f" approx. ({gpu_hours / n_jobs:,.0f} GPU-h each)" if n_jobs > 1 else "")
    )

    # RUNTIME, not just cost. GPU-h answers "can I afford it"; this answers
    # "when do I get results", which is the number you actually want when
    # deciding whether to sit and wait for a dry-run-verified plan.
    cli_repeat = getattr(args, "repeat", 1) or 1
    for idx, job in enumerate(plan.jobs):
        seg_h, seg_basis = _job_segment_hours(job)
        reps = (
            cli_repeat if cli_repeat > 1 else int(getattr(job.config.job, "chain_repeat", 1) or 1)
        )
        label = getattr(job.config.job, "name", None) or f"job {idx}"
        chain = f" x{reps} chained = {_fmt_hm(seg_h * reps)}" if reps > 1 else ""
        print(f"  runtime {_fmt_hm(seg_h)}/segment{chain}  [{seg_basis}]  {label}")
        # When the schedule outlasts the allocation, the segment length is not
        # the answer to "when is it done" — say how many segments it takes.
        full = _full_run_hours(job)
        if full and full[0] > seg_h * reps * 1.001:
            total_h, steps = full
            segments = math.ceil(total_h / seg_h) if seg_h > 0 else 0
            print(
                f"    full schedule: {steps:,} steps = {total_h / 24:.1f} days of stepping"
                f" -> ~{segments} segments of {_fmt_hm(seg_h)}"
                f" ({reps} queued now, so ~{max(0, math.ceil(segments / reps))} resubmissions)"
            )
        if idx >= 7 and n_jobs > 9:
            print(f"  ... {n_jobs - idx - 1} more job(s)")
            break
    if n_jobs > 1:
        # Arms of a sweep run CONCURRENTLY when the scheduler has room, so the
        # sum is an upper bound on wall clock, not a prediction. Say so rather
        # than printing a total that reads like a delivery date.
        longest = max(_job_segment_hours(j)[0] for j in plan.jobs)
        print(
            f"  -> wall clock between {_fmt_hm(longest)} (all arms concurrent) and "
            f"{_fmt_hm(sum(_job_segment_hours(j)[0] for j in plan.jobs))} (fully serial), "
            "excluding queue time"
        )

    # Fire only when NO step time was configured — not merely when some other
    # limit happened to bind. Production supplies one and is still capped by
    # exit_duration_in_mins, which is not a reason to tell the user to supply one.
    if not any(getattr(j.config.job, "est_step_time_s", None) for j in plan.jobs):
        print(
            "  NB priced at the wall clock. If this job exits early, set "
            "job.est_step_time_s (+ job.est_startup_min) from a measured run."
        )

    if args.dry_run:
        script_paths = render_job_scripts(plan)
        if script_paths:
            print(f"Rendered {len(script_paths)} script(s) (not submitted):")
            for p in script_paths:
                print(f"  {p}")
        exit(0)

    if gpu_hours > 100:
        if n_jobs >= 5:
            jobs_comment = "Have you checked a single job with --array-subset to confirm it works?"
        else:
            jobs_comment = ""
        try:
            answer = (
                input(f"  This run will use ~{gpu_hours:.0f} GPU-h. {jobs_comment} Proceed? [y/N] ")
                .strip()
                .lower()
            )
            if gpu_hours > 5000:
                try:
                    answer = (
                        input(
                            f"  This run will use ~{gpu_hours:.0f} GPU-h !! {jobs_comment} Really proceed? [y/N] "
                        )
                        .strip()
                        .lower()
                    )
                except EOFError:
                    answer = ""
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    _write_job_provenance(
        plan,
        args=args,
        subset_indices=subset_indices,
        overrides=args.overrides,
    )

    use_chain = args.chain or any(
        getattr(job.config.job, "chain_repeat", 1) > 1 for job in plan.jobs
    )
    if use_chain:
        if args.local:
            print("--chain is not supported with --local mode", file=sys.stderr)
            return
        res = chain_submit_jobs(plan, no_error_catching=args.debug, repeat=args.repeat)
    else:
        res = submit_jobs(plan, no_error_catching=args.debug, local_mode=args.local)

    resume_cmd = _resume_command(args, res.session_id)

    if args.no_monitor or args.submit_and_exit:
        res.loop.observe_once()
        _print_resume_hint(resume_cmd, res, reason="not monitoring")
        exit(0)

    # Print the resume command BEFORE the loop starts, not only on the way out:
    # if the terminal dies, the session is killed by a timeout, or the ssh
    # connection drops, the exit path may never run and the session id would be
    # lost. It is also echoed on Ctrl-C below, which is the common case.
    _print_resume_hint(resume_cmd, res, reason="monitoring now")
    # Ctrl-C no longer raises: run_loop turns SIGINT/SIGTERM into a flag so the
    # poll in progress finishes (a signal landing inside an sbatch would
    # otherwise leave a submitted job whose id was never recorded) and records
    # the stop in <session_dir>/.monitor.stop, which is how a supervisor tells
    # "was stopped" from "died". KeyboardInterrupt is still caught for the
    # window before the loop installs its handlers.
    try:
        outcome = run_loop(res.loop)
    except KeyboardInterrupt:
        outcome = "signal"
    if outcome == "signal":
        print("\n[interrupted] Jobs were NOT cancelled -- they keep running in SLURM.")
        print(f"  Clear the stop request first:  rm {stop_path(res.state_store.root)}")
        _print_resume_hint(resume_cmd, res, reason="interrupted")
        exit(130)
    if outcome == "failed":
        _print_resume_hint(resume_cmd, res, reason="monitor gave up after repeated failures")
        exit(1)


if __name__ == "__main__":
    main()
