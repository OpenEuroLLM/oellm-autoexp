"""Top-level config and runner helpers for monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from compoconf import parse_config

from oellm_autoexp.monitor.loop import JobFileStore, MonitorLoop
from oellm_autoexp.monitor.job_client_protocol import JobClientInterface


@dataclass(kw_only=True)
class JobConfig:
    job_id: str
    registration: dict[str, Any]


@dataclass(kw_only=True)
class RunConfig:
    max_cycles: int | None = None
    sleep_seconds: float | None = None


@dataclass(kw_only=True)
class MonitorAppConfig:
    monitor: Any
    jobs: list[JobConfig] = field(default_factory=list)
    slurm_client: Any = None
    local_client: Any = None
    state_store_dir: str | None = None
    run: RunConfig = field(default_factory=RunConfig)
    raw: dict[str, Any] | None = None


def load_app_config(path: str | Path) -> MonitorAppConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return parse_app_config(payload)


def parse_app_config(payload: dict[str, Any]) -> MonitorAppConfig:
    _import_registry()
    jobs_payload = payload.get("jobs", [])
    jobs: list[JobConfig] = []
    for job in jobs_payload:
        reg_payload = job.get("registration", {})
        jobs.append(JobConfig(job_id=str(job["job_id"]), registration=dict(reg_payload)))

    # Parse client configurations
    slurm_client_config = None
    local_client_config = None

    # Check for separate client configs
    if "slurm_client" in payload:
        slurm_client_config = parse_config(JobClientInterface.cfgtype, payload["slurm_client"])
    if "local_client" in payload:
        local_client_config = parse_config(JobClientInterface.cfgtype, payload["local_client"])

    # Fallback: if single "client" field exists, use it as local client
    if not local_client_config and not slurm_client_config:
        client_payload = payload.get("client", {})
        if not client_payload:
            client_payload = {"class_name": "LocalCommandClient"}
        local_client_config = parse_config(JobClientInterface.cfgtype, client_payload)

    run_payload = payload.get("run", {})
    return MonitorAppConfig(
        monitor=payload["monitor"],
        jobs=jobs,
        slurm_client=slurm_client_config,
        local_client=local_client_config,
        state_store_dir=payload.get("state_store_dir"),
        run=RunConfig(**run_payload) if run_payload else RunConfig(),
        raw=payload,
    )


def build_loop(config: MonitorAppConfig) -> MonitorLoop:
    _import_registry()

    # Extract poll interval from oellm_autoexp.monitor config
    monitor_cfg = config.monitor
    poll_interval = 60.0
    if isinstance(monitor_cfg, dict):
        poll_interval = monitor_cfg.get("poll_interval_seconds", 60.0)

    # Instantiate clients
    slurm_client = None
    if config.slurm_client:
        slurm_client = config.slurm_client.instantiate(JobClientInterface)

    local_client = None
    if config.local_client:
        local_client = config.local_client.instantiate(JobClientInterface)

    # Validate at least one client exists
    if not slurm_client and not local_client:
        raise ValueError("At least one client (slurm or local) must be configured")

    if not config.state_store_dir:
        raise ValueError("state_store_dir is required for monitor loop")

    store = JobFileStore(config.state_store_dir)
    return MonitorLoop(
        store,
        slurm_client=slurm_client,
        local_client=local_client,
        poll_interval_seconds=poll_interval,
    )


def _import_registry() -> None:
    import monitor.actions  # noqa: F401
    import monitor.conditions  # noqa: F401
    import monitor.local_client  # noqa: F401

    try:  # slurm_gen is optional
        import monitor.slurm_client  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional slurm_gen install
        if exc.name != "slurm_gen":
            raise
    import monitor.submission  # noqa: F401


__all__ = [
    "MonitorAppConfig",
    "load_app_config",
    "parse_app_config",
    "build_loop",
]
