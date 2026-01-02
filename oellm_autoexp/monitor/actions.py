"""Monitor actions triggered by monitor events."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import re
from collections.abc import Mapping


from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    asdict,
    register,
    register_interface,
)

from oellm_autoexp.monitor.events import EventRecord, EventStatus

LOGGER = logging.getLogger(__name__)

ActionStatus = Literal["success", "retry", "failed"]

_PATTERN = re.compile(r"(?<!\$)\{([^\{\}\$:]+)\}")


def replace_braced_keys(s: str, values: Mapping[str, Any]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)  # keep as-is if missing

    return _PATTERN.sub(repl, s)


@dataclass(kw_only=True)
class ActionResult:
    status: ActionStatus
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ActionContext:
    event: EventRecord
    job_metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    workspace: Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def variables(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self.job_metadata)
        merged.update(self.event.metadata)
        merged.update(self.event.payload)
        merged.setdefault("event_id", self.event.event_id)
        merged.setdefault("event_name", self.event.name)
        return merged

    def render(self, template: str) -> str:
        try:
            return replace_braced_keys(template, self.variables)
        except KeyError:
            return template


@register_interface
class BaseMonitorAction(RegistrableConfigInterface):
    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def describe(self, job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        payload = asdict(self.config)
        payload.update({"type": self.kind})
        return payload

    def execute(self, context: ActionContext) -> ActionResult:  # pragma: no cover
        raise NotImplementedError

    def update_event(self, event: EventRecord, result: ActionResult) -> None:
        if result.status == "success":
            event.set_status(EventStatus.PROCESSED, note=result.message or self.kind)
        elif result.status == "failed":
            event.set_status(EventStatus.FAILED, note=result.message or self.kind)
        else:
            event.set_status(EventStatus.PENDING, note=result.message or self.kind)

    @property
    def kind(self) -> str:  # pragma: no cover - override in subclasses
        return self.__class__.__name__


def _run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None):
    print(f"RUNNING: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )


@dataclass
class RunCommandActionConfig(ConfigInterface):
    class_name: str = "RunCommandAction"
    command: list[str] = field(default_factory=list)
    cwd: str | None = None


@register
class RunCommandAction(BaseMonitorAction):
    config: RunCommandActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        if not self.config.command:
            return ActionResult(status="failed", message="command is empty")
        rendered = [context.render(segment) for segment in self.config.command]
        cwd = Path(self.config.cwd).expanduser() if self.config.cwd else context.workspace
        env = {**context.env} if context.env else None
        proc = _run_command(rendered, cwd=cwd, env=env)
        if proc.returncode == 0:
            return ActionResult(
                status="success",
                message="command completed",
                metadata={"stdout": proc.stdout.strip()},
            )
        return ActionResult(
            status="failed",
            message=f"command exited {proc.returncode}",
            metadata={"stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()},
        )


@dataclass
class RestartActionConfig(ConfigInterface):
    class_name: str = "RestartAction"
    reason: str = "Restart requested"


@register
class RestartAction(BaseMonitorAction):
    config: RestartActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(status="retry", message=self.config.reason)


@dataclass
class RunAutoexpActionConfig(ConfigInterface):
    class_name: str = "RunAutoexpAction"
    script: str = "scripts/run_autoexp.py"
    overrides: list[str] = field(default_factory=list)
    config_path: str | None = None
    no_monitor: bool = True  # Skip nested monitoring by default


@register
class RunAutoexpAction(BaseMonitorAction):
    config: RunAutoexpActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        cmd = [sys.executable, self.config.script]
        if self.config.config_path:
            cmd.extend(["--config-ref", context.render(self.config.config_path)])
        cmd.extend(context.render(arg) for arg in self.config.overrides)

        # Pass --no-monitor flag to prevent nested monitoring loop
        if self.config.no_monitor:
            cmd.append("--no-monitor")

        # Pass current session ID to reuse the same monitoring session
        session_id = context.job_metadata.get("session_id")
        if session_id:
            cmd.extend(["--plan-id", session_id])

        env = {**context.env} if context.env else None
        proc = _run_command(cmd, cwd=context.workspace, env=env)
        if proc.returncode == 0:
            return ActionResult(
                status="success",
                message="run_autoexp completed",
                metadata={"session_id": session_id},
            )
        return ActionResult(
            status="failed",
            message=f"run_autoexp exited {proc.returncode}",
            metadata={"stderr": proc.stderr.strip()},
        )


@dataclass
class LogActionConfig(ConfigInterface):
    class_name: str = "LogAction"
    message: str = ""


@register
class LogAction(BaseMonitorAction):
    config: LogActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(status="success", message=context.render(self.config.message))


@dataclass
class _LegacyLogMessageActionConfig(LogActionConfig):
    class_name: str = "LogMessageAction"


@register
class LogMessageAction(LogAction):
    config: _LegacyLogMessageActionConfig


# Backwards compatibility alias for configs referenced in existing configs/tests.
LogMessageActionConfig = _LegacyLogMessageActionConfig


@dataclass
class PublishEventActionConfig(ConfigInterface):
    class_name: str = "PublishEventAction"
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    event_name: str = ""


@register
class PublishEventAction(BaseMonitorAction):
    config: PublishEventActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            status="success",
            message="event published",
            metadata={
                "publish_event": {
                    "name": self.config.event_name,
                    "metadata": self.config.metadata,
                    "payload": self.config.payload,
                }
            },
        )


__all__ = [
    "BaseMonitorAction",
    "BaseMonitorAction",
    "ActionContext",
    "ActionResult",
    "RunCommandAction",
    "RestartAction",
    "RunAutoexpAction",
    "LogAction",
    "LogMessageAction",
    "PublishEventAction",
]
