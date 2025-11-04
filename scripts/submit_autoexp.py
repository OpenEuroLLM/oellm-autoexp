#!/usr/bin/env python3
"""Submit jobs described by a plan manifest and optionally monitor them."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from oellm_autoexp.workflow.host import (
    build_host_runtime,
    instantiate_controller,
    run_monitoring,
    submit_pending_jobs,
)
from oellm_autoexp.workflow.manifest import read_manifest


def _render_log_hint(log_template: str | Path, job_id: str) -> str:
    """Expand SLURM log templates (%j, %A, %a) using the current job id."""
    log_str = str(log_template)
    if "_" in job_id:
        base_id, array_idx = job_id.split("_", 1)
        log_str = log_str.replace("%A", base_id)
        log_str = log_str.replace("%a", array_idx)
    log_str = log_str.replace("%j", job_id)
    return log_str


def _configure_logging(verbose: bool = False, debug: bool = False) -> None:
    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--no-monitor", action="store_true", help="Submit jobs but skip monitoring")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without submitting")
    parser.add_argument("--use-fake-slurm", action="store_true", help="Use in-memory SLURM backend")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose, args.debug)

    manifest_path = Path(args.manifest).resolve()
    manifest = read_manifest(manifest_path)

    runtime = build_host_runtime(
        manifest,
        use_fake_slurm=args.use_fake_slurm,
        manifest_path=manifest_path,
    )

    controller = instantiate_controller(runtime, quiet=True)

    submitted_job_ids = submit_pending_jobs(runtime, controller, dry_run=args.dry_run)

    if submitted_job_ids:
        jobs_by_id = {state.job_id: state for state in controller.jobs()}
        for job_id in submitted_job_ids:
            state = jobs_by_id.get(job_id)
            job_name = state.name if state else "unknown"
            log_path = state.registration.log_path if state else "?"
            print(f"submitted {job_name} -> job {job_id} -> log: {log_path}", flush=True)
    else:
        print("No new jobs submitted; monitoring session already contains all jobs.", flush=True)

    print(f"Monitoring session: {runtime.state_store.session_id}", flush=True)

    if args.dry_run:
        return

    if args.no_monitor:
        cmd = f"{sys.executable} scripts/monitor_autoexp.py --manifest {manifest_path}"
        print("Skipping monitoring (--no-monitor).", flush=True)
        print(f"To monitor later run: {cmd}", flush=True)
        return

    monitor_cmd = f"{sys.executable} scripts/monitor_autoexp.py --manifest {manifest_path}"

    try:
        run_monitoring(runtime, controller)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user (Ctrl+C).", flush=True)
        active_jobs = list(controller.jobs())
        if len(active_jobs) == 1:
            job_state = active_jobs[0]
            log_hint = _render_log_hint(job_state.registration.log_path, job_state.job_id)
            print(
                f"To inspect the active log directly:\n  tail -n 1000 -f {log_hint}",
                flush=True,
            )
        elif len(active_jobs) > 1:
            print(
                "Multiple jobs are active; inspect their logs individually (see manifest).",
                flush=True,
            )
        print(f"To resume monitoring later run:\n  {monitor_cmd}", flush=True)


if __name__ == "__main__":
    main()
