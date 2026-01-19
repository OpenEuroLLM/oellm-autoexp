"""Host-side helpers to consume plan manifests."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from typing import Any

from monitor.loop import MonitorLoop, JobFileStore
from oellm_autoexp.monitor.adapter import SlurmClientAdapter
from oellm_autoexp.workflow.manifest import PlanManifest
from oellm_autoexp.slurm.client import BaseSlurmClient, FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.config.schema import SlurmConfig
from compoconf import parse_config

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class HostRuntime:
    manifest: PlanManifest = field(default_factory=MISSING)
    loop: MonitorLoop = field(default_factory=MISSING)
    slurm_client: BaseSlurmClient = field(default_factory=MISSING)
    state_store: JobFileStore = field(default_factory=MISSING)
    action_queue_dir: Path = field(default_factory=MISSING)


def build_host_runtime(
    manifest: PlanManifest,
    *,
    use_fake_slurm: bool = False,
    manifest_path: Path | None = None,
) -> HostRuntime:
    if use_fake_slurm:
        slurm_client: BaseSlurmClient = FakeSlurmClient(FakeSlurmClientConfig())
    else:
        slurm_client = manifest.slurm_client.instantiate()

    slurm_config_obj = parse_config(SlurmConfig, manifest.slurm_config)
    slurm_client.configure(slurm_config_obj)

    monitoring_state_dir = Path(manifest.monitoring_state_dir)
    session_id = manifest.plan_id
    session_dir = monitoring_state_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    store = JobFileStore(session_dir)
    adapter = SlurmClientAdapter(slurm_client)
    loop = MonitorLoop(store, adapter)

    action_queue_dir = Path(manifest.action_queue_dir or session_dir / "actions")

    return HostRuntime(
        manifest=manifest,
        loop=loop,
        slurm_client=slurm_client,
        state_store=store,
        action_queue_dir=action_queue_dir,
    )


def run_monitoring(runtime: HostRuntime, controller: Any = None) -> None:
    # controller arg is deprecated/unused
    import time

    while True:
        runtime.loop.observe_once()
        jobs = runtime.loop._store.list_paths()
        if not list(jobs):
            LOGGER.info("All jobs finished.")
            break
        time.sleep(runtime.loop.poll_interval_seconds)


def instantiate_controller(runtime: HostRuntime, *, quiet: bool = False) -> Any:
    return runtime.loop


def snapshot_runtime(runtime: HostRuntime, controller: Any) -> None:
    pass


__all__ = [
    "HostRuntime",
    "build_host_runtime",
    "instantiate_controller",
    "run_monitoring",
    "snapshot_runtime",
]
