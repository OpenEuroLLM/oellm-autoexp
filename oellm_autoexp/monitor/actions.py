"""Monitor actions triggered by monitor events.

This module now delegates to the 'monitor' library for base types but
defines actions locally for compatibility.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compoconf import register, ConfigInterface

from monitor.actions import (
    ActionContext,
    ActionResult,
    BaseMonitorAction,
    EventRecord,
    EventStatus,
    FinishAction,
    FinishActionConfig,
)

LOGGER = logging.getLogger(__name__)


def _run_command(command: list[str], *, cwd: Any = None, env: dict[str, str] | None = None):
    print(f"RUNNING: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
    )


# --- Standard Actions (Locally Defined with unique names to avoid library collision) ---


@dataclass
class LogActionConfig(ConfigInterface):
    class_name: str = "OellmLogAction"
    message: str = ""


@register
class OellmLogAction(BaseMonitorAction):
    config: LogActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(status="success", message=context.render(self.config.message))


@dataclass
class RestartActionConfig(ConfigInterface):
    class_name: str = "OellmRestartAction"
    reason: str = "Restart requested"


@register
class OellmRestartAction(BaseMonitorAction):
    config: RestartActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        # Return empty adjustments to trigger restart logic in MonitorLoop
        return ActionResult(
            status="retry", message=self.config.reason, metadata={"adjustments": {}}
        )


@dataclass
class RunCommandActionConfig(ConfigInterface):
    class_name: str = "OellmRunCommandAction"
    command: list[str] = field(default_factory=list)
    cwd: str | None = None


@register
class OellmRunCommandAction(BaseMonitorAction):
    config: RunCommandActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        if not self.config.command:
            return ActionResult(status="failed", message="command is empty")
        rendered = [context.render(segment) for segment in self.config.command]
        cwd = Path(self.config.cwd).expanduser() if self.config.cwd else context.workspace
        env = {**getattr(context, "env", {})} if getattr(context, "env", None) else None
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


# Aliases for backward compatibility in usage
LogAction = OellmLogAction
RestartAction = OellmRestartAction
RunCommandAction = OellmRunCommandAction

# --- Project Specific Actions ---


@dataclass
class LogMessageActionConfig(ConfigInterface):
    class_name: str = "LogMessageAction"
    message: str = ""


@register
class LogMessageAction(OellmLogAction):
    config: LogMessageActionConfig


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


@dataclass
class RunAutoexpActionConfig(ConfigInterface):
    class_name: str = "RunAutoexpAction"
    script: str = "scripts/run_autoexp.py"
    overrides: list[str] = field(default_factory=list)
    config_path: str | None = None
    no_monitor: bool = True


@register
class RunAutoexpAction(BaseMonitorAction):
    config: RunAutoexpActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        cmd = [sys.executable, self.config.script]
        if self.config.config_path:
            cmd.extend(["--config-ref", context.render(self.config.config_path)])
        cmd.extend(context.render(arg) for arg in self.config.overrides)

        if self.config.no_monitor:
            cmd.append("--no-monitor")

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


__all__ = [
    "ActionContext",
    "ActionResult",
    "BaseMonitorAction",
    "EventRecord",
    "EventStatus",
    "FinishAction",
    "FinishActionConfig",
    "LogAction",
    "LogActionConfig",
    "LogMessageAction",
    "LogMessageActionConfig",
    "OellmLogAction",
    "OellmRestartAction",
    "OellmRunCommandAction",
    "PublishEventAction",
    "PublishEventActionConfig",
    "RestartAction",
    "RestartActionConfig",
    "RunAutoexpAction",
    "RunAutoexpActionConfig",
    "RunCommandAction",
    "RunCommandActionConfig",
]
