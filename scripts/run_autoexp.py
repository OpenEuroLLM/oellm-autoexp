#!/usr/bin/env python3
"""Convenience wrapper to plan, submit, and optionally monitor in one shot."""

from __future__ import annotations

import argparse
import json
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
        "--yes", "-y", action="store_true",
        help="Skip the GPU-h confirmation prompt(s). Required for detached / "
        "non-interactive launches (nohup/setsid) where stdin is closed.",
    )
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


def _job_runtime_hours(job) -> tuple[float, bool]:
    """Estimated runtime (hours) for a single job's GPU-hour accounting.

    Prefers a backend-agnostic ``aux.true_time`` if the config provides one, so
    the sweep-wide estimate reflects the *real* (possibly >QOS, un-capped)
    runtime rather than the ``slurm --time`` reservation — which is clamped to
    the partition's max wall-time and would badly under-count any job that
    wall-time-chains. ``aux.true_time`` may be a number of hours or a SLURM-style
    duration string ("D-HH:MM:SS"/"HH:MM:SS"). Falls back to ``slurm --time``.

    Returns (hours, used_true_time).
    """
    aux = getattr(job.config, "aux", None)
    true_time = None
    if aux is not None:
        try:
            true_time = aux.get("true_time") if hasattr(aux, "get") else aux["true_time"]
        except (KeyError, TypeError):
            true_time = None
    # ``aux`` is a free-form dict, so an interpolated ``true_time`` (e.g.
    # "${oc.eval:...}") is not resolved in place — resolve it lazily against the
    # job's own config. Accessing just this node resolves it and its dependency
    # subtree (train_iters, tok_s, nodes, ...) without touching unrelated
    # ``${sibling...}`` references elsewhere.
    if isinstance(true_time, str) and "${" in true_time:
        try:
            from omegaconf import OmegaConf

            full = OmegaConf.create(asdict(job.config))
            true_time = OmegaConf.select(
                full, "aux.true_time", throw_on_resolution_failure=True
            )
        except Exception:
            true_time = None
    if true_time is not None:
        if isinstance(true_time, (int, float)) and not isinstance(true_time, bool):
            return float(true_time), True
        try:
            return _parse_slurm_time(str(true_time)), True
        except (ValueError, IndexError):
            pass  # malformed -> fall through to the slurm reservation

    time_str = str(getattr(job.config.slurm.sbatch, "time", "1:00:00") or "1:00:00")
    try:
        return _parse_slurm_time(time_str), False
    except (ValueError, IndexError):
        return 0.0, False


def _format_hms(seconds: float) -> str:
    """Render seconds as a SLURM ``H:MM:SS`` duration.

    Floored at 30 min so short-runtime jobs (e.g. the first_cooldown "seed" that
    only loads a checkpoint + saves) still reserve enough for container start,
    kernel compile and checkpoint I/O rather than the throughput-derived ~0.
    """
    total = max(1800, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


TRUE_TIME_BUFFER = 1.5


def _apply_true_time_reservations(plan: ExecutionPlan, buffer: float = TRUE_TIME_BUFFER) -> int:
    """Right-size each job's ``slurm --time`` from the measured ``aux.true_time``.

    The reservation is ``min(configured --time, true_time * buffer)`` — the
    configured value acts as the cluster/QOS ceiling (e.g. 12h), so jobs whose
    true runtime exceeds it keep the full ceiling and wall-time-chain. Jobs
    without an ``aux.true_time`` are left untouched. The estimate in
    ``_compute_gpu_hours`` still reads the un-capped ``true_time``, so this only
    affects reservations, not the reported budget. Returns the count adjusted.

    The 50% buffer (``buffer=1.5``) absorbs the GPFS-dataloader throughput
    variance (iteration-time spikes) so a right-sized reservation still finishes
    within its wall-time window instead of timing out and restart-looping.
    """
    adjusted = 0
    for job in plan.jobs:
        hours, used_true = _job_runtime_hours(job)
        if not used_true:
            continue
        sbatch = job.config.slurm.sbatch
        cap_str = str(getattr(sbatch, "time", "") or "")
        try:
            cap_hours = _parse_slurm_time(cap_str) if cap_str else float("inf")
        except (ValueError, IndexError):
            cap_hours = float("inf")
        reserve_hours = min(cap_hours, hours * buffer)
        sbatch.time = _format_hms(reserve_hours * 3600)
        adjusted += 1
    return adjusted


def _compute_gpu_hours(plan: ExecutionPlan) -> tuple[float, int]:
    """Sum GPU-hours across all jobs. Returns (total_gpu_hours, n_from_true_time)."""
    total = 0.0
    n_true = 0
    for job in plan.jobs:
        sbatch = job.config.slurm.sbatch
        nodes = int(getattr(sbatch, "nodes", 1) or 1)
        gpus_per_node = int(getattr(sbatch, "gpus_per_node", 1) or 1)
        hours, used_true = _job_runtime_hours(job)
        if used_true:
            n_true += 1
        total += hours * nodes * gpus_per_node
    return total, n_true


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

    n_reserved = _apply_true_time_reservations(plan)
    if n_reserved:
        print(
            f"Right-sized slurm --time from aux.true_time (x{TRUE_TIME_BUFFER}, "
            f"capped at the configured ceiling) for "
            f"{n_reserved}/{len(plan.jobs)} job(s)."
        )

    gpu_hours, n_true = _compute_gpu_hours(plan)
    n_jobs = len(plan.jobs)
    if n_true == n_jobs and n_jobs > 0:
        basis = "aux.true_time (measured runtime, un-capped)"
    elif n_true > 0:
        basis = f"aux.true_time for {n_true}/{n_jobs} jobs, else slurm --time"
    else:
        basis = "full slurm --time (no aux.true_time set)"
    print(
        f"Plan: {n_jobs} job(s) — estimated {gpu_hours:.1f} GPU-h total - basis: {basis}; no restarts"
        + (f" approx. ({gpu_hours / n_jobs:.1f} GPU-h each)" if n_jobs > 1 else "")
    )

    if args.dry_run:
        script_paths = render_job_scripts(plan)
        if script_paths:
            print(f"Rendered {len(script_paths)} script(s) (not submitted):")
            for p in script_paths:
                print(f"  {p}")
        exit(0)
    if gpu_hours > 100 and not args.dry_run and not args.yes:
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

    if args.no_monitor or args.submit_and_exit:
        res.loop.observe_once()
        exit(0)

    run_loop(res.loop)


if __name__ == "__main__":
    main()
