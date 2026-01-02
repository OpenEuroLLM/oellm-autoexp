"""Helpers for building plan manifests from execution plans."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from collections.abc import Iterable

from compoconf import asdict

from oellm_autoexp.persistence.state_store import _serialize_for_json
from oellm_autoexp.workflow.manifest import (
    ArraySpec,
    ComponentSpec,
    PlanJobSpec,
    PlanManifest,
    RenderedArtifactsSpec,
)

LOGGER = logging.getLogger(__name__)


def create_manifest(
    plan,
    artifacts,
    *,
    config_ref: str,
    config_dir: Path,
    overrides: Iterable[str],
    container_image: str | None,
    container_runtime: str | None,
    plan_id: str | None = None,
) -> PlanManifest:
    """Convert an execution plan and rendered artifacts into a manifest."""

    manifest_id = plan_id or PlanManifest.new_plan_id()
    monitor = plan.runtime.monitor

    monitor_spec = plan.runtime.monitor.__class__
    monitor_config_cls = monitor.config.__class__

    slurm_client = plan.runtime.slurm_client
    slurm_client_cls = slurm_client.__class__
    slurm_client_config_cls = slurm_client.config.__class__

    slurm_config_obj = plan.config.slurm

    jobs: list[PlanJobSpec] = []

    script_paths = {job.name: path for job, path in zip(plan.jobs, artifacts.job_scripts)}
    default_array_script = (
        str(artifacts.array_script) if artifacts.array_script is not None else None
    )

    for job in plan.jobs:
        script_path = script_paths.get(job.name, default_array_script)
        if script_path is None:
            # Fallback: this should not happen, but keep manifest valid
            script_path = f"ARRAY::{plan.config.project.name}"
        jobs.append(
            PlanJobSpec(
                name=job.name,
                script_path=str(script_path),
                log_path=str(job.log_path),
                output_dir=str(job.output_dir),
                output_paths=[str(path) for path in job.output_paths],
                parameters=dict(job.parameters),
                start_condition_cmd=job.start_condition_cmd,
                start_condition_interval_seconds=job.start_condition_interval_seconds,
                termination_string=job.termination_string,
                termination_command=job.termination_command,
                inactivity_threshold_seconds=job.inactivity_threshold_seconds,
            )
        )

    rendered = RenderedArtifactsSpec(
        job_scripts=list(artifacts.job_scripts),
        sweep_json=artifacts.sweep_json,
        array=None
        if artifacts.array_script is None
        else ArraySpec(
            script_path=str(artifacts.array_script),
            job_name=str(artifacts.array_job_name or plan.config.project.name),
            size=len(plan.jobs),
        ),
    )

    container_cfg = plan.config.container
    image = container_image or (container_cfg.image if container_cfg else None)
    runtime = container_runtime or (container_cfg.runtime if container_cfg else None)

    manifest = PlanManifest(
        version=1,
        plan_id=manifest_id,
        created_at=time.time(),
        project_name=plan.config.project.name,
        config_ref=config_ref,
        config_dir=str(Path(config_dir)),
        overrides=list(overrides),
        base_output_dir=str(plan.config.project.base_output_dir),
        monitoring_state_dir=str(plan.config.project.monitoring_state_dir),
        container_image=image,
        container_runtime=runtime,
        config=_serialize_for_json(asdict(plan.config)),
        jobs=jobs,
        rendered=rendered,
        monitor=ComponentSpec(
            module=monitor_spec.__module__,
            class_name=monitor_spec.__name__,
            config_module=monitor_config_cls.__module__,
            config_class=monitor_config_cls.__name__,
            config=_serialize_for_json(asdict(monitor.config)),
        ),
        slurm_client=ComponentSpec(
            module=slurm_client_cls.__module__,
            class_name=slurm_client_cls.__name__,
            config_module=slurm_client_config_cls.__module__,
            config_class=slurm_client_config_cls.__name__,
            config=_serialize_for_json(asdict(slurm_client.config)),
        ),
        slurm_config_module=slurm_config_obj.__class__.__module__,
        slurm_config_class=slurm_config_obj.__class__.__name__,
        slurm_config=_serialize_for_json(asdict(slurm_config_obj)),
        action_queue_dir=str(Path(plan.config.project.monitoring_state_dir) / "actions"),
    )
    return manifest


__all__ = ["create_manifest"]
