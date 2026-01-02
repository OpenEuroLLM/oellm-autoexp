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
from omegaconf import OmegaConf

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts, ConfigSetup
from oellm_autoexp.workflow.host import (
    build_host_runtime,
    instantiate_controller,
    run_monitoring,
    submit_pending_jobs,
)
from oellm_autoexp.workflow.manifest import write_manifest
from oellm_autoexp.workflow.plan import create_manifest
from oellm_autoexp.utils.logging_config import configure_logging


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        "--array-subset",
        type=str,
        help="Comma-separated sweep indices or ranges (e.g., '0,3-5') to rerun.",
    )
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


def _write_job_provenance(
    plan,
    *,
    args: argparse.Namespace,
    subset_indices: set[int],
    overrides: list[str],
) -> None:
    resolved_config = asdict(plan.config)
    git_meta = _collect_git_metadata(REPO_ROOT)
    sanitized_env = _sanitize_env()
    config_reference = {
        "config_ref": args.config_ref,
        "config_dir": str(Path(args.config_dir).resolve()),
        "overrides": overrides,
    }
    base_payload = {
        "git": git_meta,
        "command": sys.argv,
        "overrides": overrides,
        "subset_indices": sorted(subset_indices),
        "config_reference": config_reference,
        "environment": sanitized_env,
    }

    # Re-load config as OmegaConf to preserve interpolations for unresolved YAML
    # We need to reload it because plan.config is already a dataclass
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from oellm_autoexp.config.resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir = Path(args.config_dir).resolve()

    # Load config using Hydra to get OmegaConf container
    # Check if Hydra is already initialized to avoid conflicts
    try:
        if GlobalHydra.instance().is_initialized():
            # Hydra already initialized, clear it first
            GlobalHydra.instance().clear()
    except Exception:
        pass  # GlobalHydra not initialized yet

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg_omega = compose(config_name=args.config_ref, overrides=overrides)
    except Exception as e:
        # If Hydra initialization fails, log error and skip YAML provenance files
        print(f"Warning: Could not generate unresolved config YAML: {e}", file=sys.stderr)
        cfg_omega = None

    for job in plan.jobs:
        output_dir = Path(job.output_dir)
        provenance_dir = output_dir / "provenance"
        print(f"WRITING TO PROVENANCE DIR: {provenance_dir}")
        provenance_dir.mkdir(parents=True, exist_ok=True)

        # Keep existing: resolved_config.json (for backwards compatibility)
        resolved_path = provenance_dir / "resolved_config.json"
        resolved_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")

        # NEW: resolved_config.yaml and unresolved_config.yaml (if Hydra load succeeded)
        if cfg_omega is not None:
            try:
                # Resolved YAML (fully resolved, easier to read/edit)
                resolved_yaml_path = provenance_dir / "resolved_config.yaml"
                OmegaConf.save(config=cfg_omega, f=str(resolved_yaml_path), resolve=True)

                # Unresolved YAML (keep ${...} interpolations for flexibility)
                unresolved_yaml_path = provenance_dir / "unresolved_config.yaml"
                OmegaConf.save(config=cfg_omega, f=str(unresolved_yaml_path), resolve=False)
            except Exception as e:
                print(f"Warning: Could not write YAML configs: {e}", file=sys.stderr)

        # Keep existing: config_reference.json (for Hydra reconstruction)
        reference_path = provenance_dir / "config_reference.json"
        reference_path.write_text(json.dumps(config_reference, indent=2), encoding="utf-8")

        # Keep existing: run_metadata.json
        job_payload = {
            **base_payload,
            "job": {
                "name": job.name,
                "parameters": job.parameters,
                "log_path": job.log_path,
                "output_dir": job.output_dir,
            },
        }
        metadata_path = provenance_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(job_payload, indent=2), encoding="utf-8")


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
    configure_logging(args.verbose, args.debug)

    config_dir = Path(args.config_dir)
    root = load_config_reference(args.config_ref, config_dir, args.override)
    try:
        subset_indices = _parse_subset(args.array_subset)
    except ValueError as exc:
        print(f"Invalid --array-subset argument: {exc}", file=sys.stderr)
        return

    try:
        plan = build_execution_plan(
            root,
            config_setup=ConfigSetup(
                pwd=os.path.abspath(os.curdir),
                config_ref=args.config_ref,
                config_dir=str(config_dir),
                override=args.override,
            ),
            subset_indices=subset_indices or None,
        )
    except ValueError as exc:
        print(f"Error while building execution plan: {exc}", file=sys.stderr)
        return

    _write_job_provenance(
        plan,
        args=args,
        subset_indices=subset_indices,
        overrides=args.override,
    )

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
