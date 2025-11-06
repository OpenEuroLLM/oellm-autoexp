#!/usr/bin/env python3
"""Render SBATCH scripts and produce a reusable plan manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts
from oellm_autoexp.workflow.manifest import write_manifest
from oellm_autoexp.workflow.plan import create_manifest


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
    parser.add_argument(
        "override", nargs="*", default=[], help="Hydra-style overrides (`key=value`)."
    )
    return parser.parse_args(argv)


def _default_manifest_path(monitoring_state_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest_dir = monitoring_state_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex[:6]
    return manifest_dir / f"plan_{timestamp}_{suffix}.json"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
        monitoring_state_dir = Path(plan.config.project.monitoring_state_dir)
        manifest_path = _default_manifest_path(monitoring_state_dir)

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


if __name__ == "__main__":
    main()
