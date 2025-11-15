"""Convert typed configuration into instantiated components."""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from pathlib import Path

from . import schema
from ..backends.base import BaseBackend
from ..monitor.watcher import BaseMonitor
from ..slurm.client import BaseSlurmClient, SlurmClientConfig


@dataclass(kw_only=True)
class RuntimeConfig:
    """Holds instantiated components alongside original config."""

    root: schema.RootConfig = field(default_factory=MISSING)
    backend: BaseBackend = field(default_factory=MISSING)
    monitor: BaseMonitor = field(default_factory=MISSING)
    slurm_client: BaseSlurmClient = field(default_factory=MISSING)

    @property
    def state_dir(self) -> Path:
        """Returns the monitoring state directory (stable, not run-specific).

        This is used for monitoring sessions and should NOT include
        timestamps so that --monitor-all can find sessions across runs.
        """
        return (
            self.root.project.monitoring_state_dir
            or self.root.project.state_dir
            or "./monitoring_state"
        )


def evaluate(root: schema.RootConfig) -> RuntimeConfig:
    """Instantiate components defined in ``RootConfig``."""

    backend = root.backend.instantiate(schema.BackendInterface)
    monitor = root.monitoring.instantiate(schema.MonitorInterface)

    client_cfg = root.slurm.client or SlurmClientConfig()
    slurm_client = client_cfg.instantiate(schema.SlurmClientInterface)
    slurm_client.configure(root.slurm)

    return RuntimeConfig(
        root=root,
        backend=backend,
        monitor=monitor,
        slurm_client=slurm_client,
    )


__all__ = ["RuntimeConfig", "evaluate"]
