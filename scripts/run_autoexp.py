#!/usr/bin/env python3
"""Convenience wrapper around the oellm-autoexp CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts
from oellm_autoexp.slurm.fake_sbatch import FakeSlurm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoexp orchestration")
    parser.add_argument("config_ref", nargs="?", default="autoexp")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument("-o", "--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-fake-slurm", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = load_config_reference(args.config_ref, args.config_dir, args.override)
    plan = build_execution_plan(root)
    scripts = render_scripts(plan)

    if args.dry_run:
        for script in scripts:
            print(f"Generated: {script}")
        return

    if args.use_fake_slurm:
        slurm = FakeSlurm()
        for job, script in zip(plan.jobs, scripts):
            job_id = slurm.submit(job.name, script, job.log_path)
            print(f"submitted {job.name} -> job {job_id}")
        return

    raise SystemExit("Real SLURM submission not yet implemented")


if __name__ == "__main__":
    main()
