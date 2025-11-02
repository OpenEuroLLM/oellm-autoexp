#!/usr/bin/env python3
"""Convenience wrapper around the oellm-autoexp CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from compoconf import asdict, parse_config

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import RootConfig
from oellm_autoexp.orchestrator import (
    build_execution_plan,
    execute_plan_sync,
    load_monitor_controller,
    render_scripts,
    submit_jobs,
)
from oellm_autoexp.persistence import MonitorStateStore
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoexp orchestration")
    parser.add_argument("--config-ref", default="autoexp")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-fake-slurm", action="store_true", default=False)
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Generate scripts but output sbatch command instead of submitting",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Submit jobs but don't monitor them (return immediately)",
    )
    parser.add_argument(
        "--monitor-session",
        type=str,
        help="Monitor a specific session by ID (reads from monitoring_state)",
    )
    parser.add_argument(
        "--monitor-all",
        action="store_true",
        help="Monitor all active sessions in monitoring_state directory",
    )
    parser.add_argument(
        "--monitoring-state-dir",
        type=Path,
        help="Path to monitoring_state directory (default: output/monitoring_state)",
    )
    parser.add_argument("--dump-config", type=Path, help="Dump resolved config to file and exit")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging (INFO level)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (DEBUG level)")
    parser.add_argument("override", nargs="*", default=[], help="Additional overrides.")
    return parser.parse_args()


def monitor_from_session_file(session_path: Path, use_fake_slurm: bool = False) -> None:
    """Monitor jobs from a session file."""
    session_data = MonitorStateStore.load_session(session_path)
    if not session_data:
        print(f"Error: Could not load session from {session_path}")
        return

    # Reconstruct config from session
    config_dict = session_data.get("config")
    if not config_dict:
        print(f"Error: No config found in session {session_path}")
        return

    root = parse_config(RootConfig, config_dict)
    plan = build_execution_plan(root)

    slurm_client = plan.runtime.slurm_client
    if use_fake_slurm:
        slurm_client = FakeSlurmClient(FakeSlurmClientConfig())
        slurm_client.configure(plan.config.slurm)

    session_id = session_data.get("session_id", session_path.stem)
    print(f"Monitoring session: {session_id} ({session_data.get('project_name', 'unknown')})")

    submission = load_monitor_controller(plan, slurm_client, session_id=session_id)
    execute_plan_sync(plan, submission.controller)


def main() -> None:
    args = parse_args()

    # Configure logging based on verbosity flags
    log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handle monitoring from session files
    if args.monitor_session or args.monitor_all:
        monitoring_state_dir = args.monitoring_state_dir or Path("output/monitoring_state")

        if args.monitor_all:
            sessions = MonitorStateStore.list_sessions(monitoring_state_dir)
            if not sessions:
                print(f"No monitoring sessions found in {monitoring_state_dir}")
                return

            print(f"Found {len(sessions)} session(s) to monitor:")
            for session in sessions:
                print(
                    f"  - {session['session_id']}: {session['project_name']} ({session['job_count']} jobs)"
                )

            # Monitor all sessions (this could be enhanced to run in parallel)
            for session in sessions:
                session_path = Path(session["session_path"])
                try:
                    monitor_from_session_file(session_path, args.use_fake_slurm)
                except Exception as e:
                    print(f"Error monitoring session {session['session_id']}: {e}")
                    continue
            return

        if args.monitor_session:
            session_path = monitoring_state_dir / f"{args.monitor_session}.json"
            if not session_path.exists():
                print(f"Error: Session file not found: {session_path}")
                print("Available sessions:")
                sessions = MonitorStateStore.list_sessions(monitoring_state_dir)
                for session in sessions:
                    print(f"  - {session['session_id']}: {session['project_name']}")
                return

            monitor_from_session_file(session_path, args.use_fake_slurm)
            return

    root = load_config_reference(args.config_ref, args.config_dir, args.override)

    # Dump config and exit if requested
    if args.dump_config:
        config_dict = asdict(root)
        args.dump_config.parent.mkdir(parents=True, exist_ok=True)
        args.dump_config.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
        print(f"Config dumped to: {args.dump_config}")
        return

    plan = build_execution_plan(root)
    rendered_artifacts = None

    if not args.no_submit:
        rendered_artifacts = render_scripts(plan)

    if args.dry_run:
        if rendered_artifacts is None:
            rendered_artifacts = render_scripts(plan)
        for script in rendered_artifacts.job_scripts:
            print(f"Generated: {script}")
        if rendered_artifacts.array_script:
            print(f"Generated array script: {rendered_artifacts.array_script}")
        if rendered_artifacts.sweep_json:
            print(f"Sweep JSON: {rendered_artifacts.sweep_json}")
        return

    # When --no-submit is set, output the sbatch command for the host to execute
    if args.no_submit:
        if rendered_artifacts is None:
            rendered_artifacts = render_scripts(plan)
        if rendered_artifacts.array_script:
            script_path = rendered_artifacts.array_script
        elif rendered_artifacts.job_scripts:
            script_path = rendered_artifacts.job_scripts[0]
        else:
            raise RuntimeError("No scripts generated to submit")

        # Output in format expected by run_autoexp_container.py
        print(f"Successful, to execute, run: sbatch {script_path}")
        return

    slurm_client = plan.runtime.slurm_client
    if args.use_fake_slurm:
        slurm_client = FakeSlurmClient(FakeSlurmClientConfig())
        slurm_client.configure(plan.config.slurm)

    if rendered_artifacts is None:
        rendered_artifacts = render_scripts(plan)

    submission = submit_jobs(plan, rendered_artifacts, slurm_client)

    jobs_by_id = {state.job_id: state for state in submission.controller.jobs()}
    if submission.submitted_job_ids:
        for job_id in submission.submitted_job_ids:
            state = jobs_by_id.get(job_id)
            job_name = state.name if state else "unknown"
            log_path = state.registration.log_path if state else "?"
            print(f"submitted {job_name} -> job {job_id} -> log: {log_path}")
    else:
        print("No new jobs submitted; monitoring session already contains all jobs.")

    print(f"Monitoring session: {submission.session_id}")

    if args.no_monitor:
        return

    execute_plan_sync(plan, submission.controller)


if __name__ == "__main__":
    main()
