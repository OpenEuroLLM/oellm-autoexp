"""Serialization helpers for container-generated execution plans."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, MISSING
from importlib import import_module
from pathlib import Path
from typing import Any


def _import_object(module: str, name: str) -> Any:
    """Load an object given its module and attribute name."""
    mod = import_module(module)
    return getattr(mod, name)


@dataclass(kw_only=True)
class ComponentSpec:
    """Description of a runtime component and its configuration."""

    module: str = field(default_factory=MISSING)
    class_name: str = field(default_factory=MISSING)
    config_module: str = field(default_factory=MISSING)
    config_class: str = field(default_factory=MISSING)
    config: dict[str, Any] = field(default_factory=MISSING)

    def instantiate(self) -> Any:
        """Instantiate the described component."""
        config_cls = _import_object(self.config_module, self.config_class)
        component_cls = _import_object(self.module, self.class_name)
        config_obj = config_cls(**self.config)
        return component_cls(config_obj)


@dataclass(kw_only=True)
class PolicySpec(ComponentSpec):
    mode: str = ""


@dataclass(kw_only=True)
class PlanJobSpec:
    """Metadata required to re-submit and monitor a job."""

    name: str = field(default_factory=MISSING)
    script_path: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    output_paths: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None
    termination_string: str | None = None
    termination_command: str | None = None
    inactivity_threshold_seconds: int | None = None


@dataclass(kw_only=True)
class ArraySpec:
    script_path: str = field(default_factory=MISSING)
    job_name: str = field(default_factory=MISSING)
    size: int = field(default_factory=MISSING)


@dataclass(kw_only=True)
class RenderedArtifactsSpec:
    job_scripts: list[str] = field(default_factory=list)
    sweep_json: str | None = None
    array: ArraySpec | None = None


@dataclass(kw_only=True)
class PlanManifest:
    """Serializable container planning result consumed on the host."""

    version: int = field(default_factory=MISSING)
    plan_id: str = field(default_factory=MISSING)
    created_at: float = field(default_factory=MISSING)
    project_name: str = field(default_factory=MISSING)
    config_ref: str = field(default_factory=MISSING)
    config_dir: str = field(default_factory=MISSING)
    overrides: list[str] = field(default_factory=list)
    base_output_dir: str = field(default_factory=MISSING)
    monitoring_state_dir: str = field(default_factory=MISSING)
    container_image: str | None = None
    container_runtime: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    jobs: list[PlanJobSpec] = field(default_factory=list)
    rendered: RenderedArtifactsSpec = field(default_factory=MISSING)
    monitor: ComponentSpec = field(default_factory=MISSING)
    restart_policies: list[PolicySpec] = field(default_factory=list)
    slurm_client: ComponentSpec = field(default_factory=MISSING)
    slurm_config_module: str = field(default_factory=MISSING)
    slurm_config_class: str = field(default_factory=MISSING)
    slurm_config: dict[str, Any] = field(default_factory=dict)
    action_queue_dir: str = field(default_factory=MISSING)

    @staticmethod
    def new_plan_id() -> str:
        return uuid.uuid4().hex[:8]

    def to_dict(self) -> dict[str, Any]:
        def serialize_component(spec: ComponentSpec) -> dict[str, Any]:
            payload = {
                "module": spec.module,
                "class_name": spec.class_name,
                "config_module": spec.config_module,
                "config_class": spec.config_class,
                "config": spec.config,
            }
            if isinstance(spec, PolicySpec):
                payload["mode"] = spec.mode
            return payload

        data = {
            "version": self.version,
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "project_name": self.project_name,
            "config_ref": self.config_ref,
            "config_dir": self.config_dir,
            "overrides": list(self.overrides),
            "base_output_dir": self.base_output_dir,
            "monitoring_state_dir": self.monitoring_state_dir,
            "container_image": self.container_image,
            "container_runtime": self.container_runtime,
            "config": self.config,
            "jobs": [
                {
                    "name": job.name,
                    "script_path": job.script_path,
                    "log_path": job.log_path,
                    "output_paths": list(job.output_paths),
                    "parameters": dict(job.parameters),
                    "start_condition_cmd": job.start_condition_cmd,
                    "start_condition_interval_seconds": job.start_condition_interval_seconds,
                    "termination_string": job.termination_string,
                    "termination_command": job.termination_command,
                    "inactivity_threshold_seconds": job.inactivity_threshold_seconds,
                }
                for job in self.jobs
            ],
            "rendered": {
                "job_scripts": list(self.rendered.job_scripts),
                "sweep_json": self.rendered.sweep_json,
                "array": None
                if self.rendered.array is None
                else {
                    "script_path": self.rendered.array.script_path,
                    "job_name": self.rendered.array.job_name,
                    "size": self.rendered.array.size,
                },
            },
            "monitor": serialize_component(self.monitor),
            "restart_policies": [serialize_component(spec) for spec in self.restart_policies],
            "slurm_client": serialize_component(self.slurm_client),
            "slurm_config_module": self.slurm_config_module,
            "slurm_config_class": self.slurm_config_class,
            "slurm_config": self.slurm_config,
            "action_queue_dir": self.action_queue_dir,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanManifest:
        def load_component(payload: dict[str, Any], *, is_policy: bool = False) -> ComponentSpec:
            spec_cls = PolicySpec if is_policy else ComponentSpec
            return spec_cls(
                module=payload["module"],
                class_name=payload["class_name"],
                config_module=payload["config_module"],
                config_class=payload["config_class"],
                config=payload.get("config", {}),
                **({"mode": payload.get("mode", "")} if is_policy else {}),
            )

        jobs = [
            PlanJobSpec(
                name=item["name"],
                script_path=item["script_path"],
                log_path=item["log_path"],
                output_paths=list(item.get("output_paths", [])),
                parameters=dict(item.get("parameters", {})),
                start_condition_cmd=item.get("start_condition_cmd"),
                start_condition_interval_seconds=item.get("start_condition_interval_seconds"),
                termination_string=item.get("termination_string"),
                termination_command=item.get("termination_command"),
                inactivity_threshold_seconds=item.get("inactivity_threshold_seconds"),
            )
            for item in data.get("jobs", [])
        ]

        rendered_payload = data.get("rendered", {})
        array_payload = rendered_payload.get("array")
        rendered = RenderedArtifactsSpec(
            job_scripts=list(rendered_payload.get("job_scripts", [])),
            sweep_json=rendered_payload.get("sweep_json"),
            array=None
            if not array_payload
            else ArraySpec(
                script_path=array_payload["script_path"],
                job_name=array_payload["job_name"],
                size=int(array_payload["size"]),
            ),
        )

        monitor_spec = load_component(data["monitor"])
        policy_specs = [
            load_component(item, is_policy=True) for item in data.get("restart_policies", [])
        ]
        slurm_spec = load_component(data["slurm_client"])

        return cls(
            version=int(data.get("version", 1)),
            plan_id=data["plan_id"],
            created_at=float(data.get("created_at", time.time())),
            project_name=data["project_name"],
            config_ref=data["config_ref"],
            config_dir=data["config_dir"],
            overrides=list(data.get("overrides", [])),
            base_output_dir=data["base_output_dir"],
            monitoring_state_dir=data["monitoring_state_dir"],
            container_image=data.get("container_image"),
            container_runtime=data.get("container_runtime"),
            config=dict(data.get("config", {})),
            jobs=jobs,
            rendered=rendered,
            monitor=monitor_spec,
            restart_policies=policy_specs,
            slurm_client=slurm_spec,
            slurm_config_module=data.get("slurm_config_module", "oellm_autoexp.config.schema"),
            slurm_config_class=data.get("slurm_config_class", "SlurmConfig"),
            slurm_config=dict(data.get("slurm_config", {})),
            action_queue_dir=data.get("action_queue_dir", ""),
        )


def write_manifest(manifest: PlanManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_manifest(path: Path) -> PlanManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlanManifest.from_dict(payload)


__all__ = [
    "ArraySpec",
    "ComponentSpec",
    "PlanJobSpec",
    "PlanManifest",
    "PolicySpec",
    "RenderedArtifactsSpec",
    "read_manifest",
    "write_manifest",
]
