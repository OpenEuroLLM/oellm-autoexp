"""Monitor actions triggered by monitor events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    asdict,
    register,
    register_interface,
)


@register_interface
class MonitorActionInterface(RegistrableConfigInterface):
    """Interface for monitor-triggered actions."""


class BaseMonitorAction(MonitorActionInterface):
    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def describe(self, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        payload = asdict(self.config)
        return {k: v for k, v in payload.items() if v is not None}

    @property
    def kind(self) -> str:  # pragma: no cover - override in subclasses
        return self.__class__.__name__


@dataclass
class ExecutionActionConfig(ConfigInterface):
    class_name: str = "ExecutionAction"
    command: list[str] = field(default_factory=list)


@register
class ExecutionAction(BaseMonitorAction):
    config: ExecutionActionConfig

    def describe(self, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        rendered = [segment.format(**metadata) for segment in self.config.command]
        return {"type": self.kind, "command": rendered}


@dataclass
class RestartActionConfig(ConfigInterface):
    class_name: str = "RestartAction"
    reason: str | None = None


@register
class RestartAction(BaseMonitorAction):
    config: RestartActionConfig

    def describe(self, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        payload = super().describe(job_id, metadata)
        payload.update({"type": self.kind, "reason": self.config.reason})
        return payload


@dataclass
class TerminationActionConfig(ConfigInterface):
    class_name: str = "TerminationAction"
    message: str | None = None


@register
class TerminationAction(BaseMonitorAction):
    config: TerminationActionConfig

    def describe(self, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        payload = super().describe(job_id, metadata)
        payload.update({"type": self.kind})
        return payload


@dataclass
class ErrorNoteActionConfig(ConfigInterface):
    class_name: str = "ErrorNoteAction"
    note: str = ""


@register
class ErrorNoteAction(BaseMonitorAction):
    config: ErrorNoteActionConfig

    def describe(self, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        payload = super().describe(job_id, metadata)
        payload.update({"type": self.kind, "note": self.config.note})
        return payload
