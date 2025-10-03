"""Monitoring primitives for oellm_autoexp."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import MonitorInterface


@dataclass
class MonitorOutcome:
    """Snapshot emitted by a monitor iteration."""

    job_id: str
    status: str
    last_update_seconds: Optional[float]
    metadata: Dict[str, str] = field(default_factory=dict)


class BaseMonitor(MonitorInterface):
    """Base class monitors can inherit from to gain a common signature."""

    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    async def watch(self, job_ids: Iterable[str]) -> Dict[str, MonitorOutcome]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class NullMonitorConfig(ConfigInterface):
    """Monitor configuration that performs no observation."""

    class_name: str = "NullMonitor"


@register
class NullMonitor(BaseMonitor):
    config: NullMonitorConfig

    async def watch(self, job_ids: Iterable[str]) -> Dict[str, MonitorOutcome]:
        return {}


__all__ = ["BaseMonitor", "MonitorOutcome", "NullMonitorConfig"]
