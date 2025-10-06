#!/usr/bin/env python3
"""Convenience wrapper around the oellm-autoexp CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.orchestrator import (
    build_execution_plan,
    execute_plan_sync,
    render_scripts,
    submit_jobs,
)
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoexp orchestration")
    parser.add_argument("--config-name", default="autoexp")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-fake-slurm", action="store_true", default=False)
    parser.add_argument(
        "--no-submit", action="store_true", help="Generate scripts but output sbatch command instead of submitting"
    )
    parser.add_argument("override", nargs="*", default=[], help="Additional overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = load_config_reference(args.config_name, args.config_dir, args.override)
    plan = build_execution_plan(root)
    artifacts = render_scripts(plan)

    if args.dry_run:
        for script in artifacts.job_scripts:
            print(f"Generated: {script}")
        if artifacts.array_script:
            print(f"Generated array script: {artifacts.array_script}")
        if artifacts.sweep_json:
            print(f"Sweep JSON: {artifacts.sweep_json}")
        return

    # When --no-submit is set, output the sbatch command for the host to execute
    if args.no_submit:
        if artifacts.array_script:
            script_path = artifacts.array_script
        elif artifacts.job_scripts:
            script_path = artifacts.job_scripts[0]
        else:
            raise RuntimeError("No scripts generated to submit")

        # Output in format expected by run_megatron_container.py
        print(f"Successful, to execute, run: sbatch {script_path}")
        return

    slurm_client = plan.runtime.slurm_client
    if args.use_fake_slurm:
        slurm_client = FakeSlurmClient(FakeSlurmClientConfig())
        slurm_client.configure(plan.config.slurm)

    controller = submit_jobs(plan, artifacts, slurm_client)
    for state in controller.jobs():
        print(f"submitted {state.name} -> job {state.job_id} -> log: {state.registration.log_path}")

    execute_plan_sync(
        plan,
        slurm_client=slurm_client,
        artifacts=artifacts,
        controller=controller,
    )


if __name__ == "__main__":
    main()
