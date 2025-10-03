"""Convert typed configuration into instantiated components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from . import schema
from ..backends.base import BaseBackend, BackendJobSpec
from ..monitor.policy import BaseRestartPolicy
from ..monitor.watcher import BaseMonitor


@dataclass
class RuntimeConfig:
    """Holds instantiated components alongside original config."""

    root: schema.RootConfig
    backend: BaseBackend
    monitor: BaseMonitor
    restart_policies: Dict[str, BaseRestartPolicy]

    @property
    def state_dir(self) -> Path:
        return Path(self.root.project.state_dir or self.root.project.base_output_dir / ".oellm-autoexp")


def evaluate(root: schema.RootConfig) -> RuntimeConfig:
    """Instantiate components defined in ``RootConfig``."""

    backend = root.backend.implementation.instantiate(schema.BackendInterface)
    monitor = root.monitoring.implementation.instantiate(schema.MonitorInterface)

    policies: Dict[str, BaseRestartPolicy] = {}
    for policy_cfg in root.restart_policies:
        policy = policy_cfg.implementation.instantiate(schema.RestartPolicyInterface)
        policies[policy_cfg.mode] = policy

    return RuntimeConfig(root=root, backend=backend, monitor=monitor, restart_policies=policies)


__all__ = ["RuntimeConfig", "evaluate"]

