#!/usr/bin/env python3
"""Convenience wrapper to plan, submit, and optionally monitor in one shot."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts
from oellm_autoexp.workflow.host import (
    build_host_runtime,
    instantiate_controller,
    run_monitoring,
    submit_pending_jobs,
)
from oellm_autoexp.workflow.manifest import write_manifest
from oellm_autoexp.workflow.plan import create_manifest


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
    parser.add_argument("--config-ref", default="autoexp")
    parser.add_argument("-C", "--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--plan-id", type=str, help="Explicit plan identifier for the manifest")
    parser.add_argument(
        "--container-image", type=str, help="Override container image recorded in manifest"
    )
    parser.add_argument(
        "--container-runtime", type=str, help="Override container runtime recorded in manifest"
    )
    parser.add_argument("--use-fake-slurm", action="store_true", help="Use in-memory SLURM backend")
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan and render without submitting jobs"
    )
    parser.add_argument("--no-monitor", action="store_true", help="Submit jobs but skip monitoring")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "override", nargs="*", default=[], help="Hydra-style overrides (`key=value`)."
    )
    return parser.parse_args(argv)


def _default_manifest_path(base_output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest_dir = base_output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex[:6]
    return manifest_dir / f"plan_{timestamp}_{suffix}.json"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose, args.debug)

    config_dir = Path(args.config_dir)
    root = load_config_reference(args.config_ref, config_dir, args.override)

    plan = build_execution_plan(root)
    artifacts = render_scripts(plan)

    manifest = create_manifest(
        plan,
        artifacts,
        config_ref=args.config_ref,
        config_dir=config_dir,
        overrides=args.override,
        container_image=args.container_image,
        container_runtime=args.container_runtime,
        plan_id=args.plan_id,
    )

    if args.manifest is not None:
        manifest_path = Path(args.manifest)
    else:
        base_output = Path(plan.config.project.base_output_dir)
        manifest_path = _default_manifest_path(base_output)

    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    write_manifest(manifest, manifest_path)

    print(f"Plan manifest written to: {manifest_path}")
    print(f"Project: {manifest.project_name}")
    print(f"Jobs: {len(manifest.jobs)}")
    if manifest.rendered.array:
        print(
            f"Array script: {manifest.rendered.array.script_path} ({manifest.rendered.array.size} tasks)"
        )
    else:
        for script in manifest.rendered.job_scripts:
            print(f"Generated script: {script}")
    if manifest.rendered.sweep_json:
        print(f"Sweep manifest: {manifest.rendered.sweep_json}")

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
            print(f"submitted {job_name} -> job {job_id} -> log: {log_path}")
    else:
        print("No new jobs submitted; monitoring session already contains all jobs.")

    print(f"Monitoring session: {runtime.state_store.session_id}")

    if args.dry_run:
        return

    if args.no_monitor:
        cmd = f"{sys.executable} -u scripts/monitor_autoexp.py --manifest {manifest_path}"
        print("Skipping monitoring (--no-monitor).")
        print(f"To monitor later run: {cmd}")
        return

    run_monitoring(runtime, controller)


if __name__ == "__main__":
    main()
