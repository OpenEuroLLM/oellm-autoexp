#!/usr/bin/env python3
"""Render SBATCH scripts and produce a reusable plan manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import os

from compoconf import asdict
from omegaconf import OmegaConf

from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.orchestrator import build_execution_plan, ConfigSetup
from scripts.utils_plan import render_scripts, create_manifest, write_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="autoexp")
    parser.add_argument("--config-path", default=None)
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
        "overrides", nargs="*", default=[], help="Hydra-style overrides (`key=value`)."
    )
    return parser.parse_args(argv)


def _default_manifest_path(monitoring_state_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest_dir = monitoring_state_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    suffix = uuid4().hex[:6]
    return manifest_dir / f"plan_{timestamp}_{suffix}.json"


def _write_job_provenance(
    plan,
    *,
    config_name: str | None,
    config_path: str | None,
    config_dir: Path,
    overrides: list[str],
) -> None:
    """Write provenance files for each job in the plan."""
    resolved_config = asdict(plan.config)
    config_reference = {
        "config_name": config_name,
        "config_path": config_path,
        "config_dir": str(config_dir.resolve()),
        "overrides": overrides,
    }

    # Re-load config as OmegaConf to preserve interpolations for unresolved YAML
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from oellm_autoexp.config.resolvers import register_default_resolvers

    register_default_resolvers()
    config_dir_resolved = config_dir.resolve()

    # Check if Hydra is already initialized to avoid conflicts
    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
    except Exception:
        pass

    cfg_omega = None
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir_resolved)):
            cfg_omega = compose(config_name=config_name, overrides=overrides)
    except Exception as e:
        print(f"Warning: Could not generate unresolved config YAML: {e}", file=sys.stderr)

    for job in plan.jobs:
        output_dir = Path(job.output_dir)
        provenance_dir = output_dir / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)

        # Write resolved_config.json (for backwards compatibility)
        resolved_path = provenance_dir / "resolved_config.json"
        resolved_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")

        # Write YAML configs if Hydra load succeeded
        if cfg_omega is not None:
            try:
                # Resolved YAML (fully resolved)
                resolved_yaml_path = provenance_dir / "resolved_config.yaml"
                OmegaConf.save(config=cfg_omega, f=str(resolved_yaml_path), resolve=True)

                # Unresolved YAML (keep ${...} interpolations)
                unresolved_yaml_path = provenance_dir / "unresolved_config.yaml"
                OmegaConf.save(config=cfg_omega, f=str(unresolved_yaml_path), resolve=False)
            except Exception as e:
                print(f"Warning: Could not write YAML configs: {e}", file=sys.stderr)

        # Write config_reference.json (for Hydra reconstruction)
        reference_path = provenance_dir / "config_reference.json"
        reference_path.write_text(json.dumps(config_reference, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = Path(args.config_dir)

    config_setup = (
        ConfigSetup(
            pwd=os.path.abspath(os.curdir),
            config_name=args.config_name,
            config_path=args.config_path,
            config_dir=config_dir,
            overrides=args.overrides,
        ),
    )

    root = load_config_reference(
        config_setup=config_setup,
    )
    plan = build_execution_plan(
        root,
        config_setup=config_setup,
    )

    # Write provenance files for each job
    _write_job_provenance(
        plan,
        config_name=args.config_name,
        config_path=args.config_path,
        config_dir=config_dir,
        overrides=args.overrides,
    )

    artifacts = render_scripts(plan)

    manifest = create_manifest(
        plan,
        artifacts,
        config_name=args.config_name,
        config_path=args.config_path,
        config_dir=config_dir,
        overrides=args.overrides,
        container_image=args.container_image,
        container_runtime=args.container_runtime,
        plan_id=args.plan_id,
    )

    if args.manifest is not None:
        manifest_path = Path(args.manifest)
    else:
        monitoring_state_dir = Path(plan.config.job.monitoring_state_dir)
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
