"""Monitor actions triggered by monitor events.

This module now delegates to the 'monitor' library for base types but
defines actions locally for compatibility.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from compoconf import register, ConfigInterface

from oellm_autoexp.monitor.actions import (
    ActionContext,
    ActionResult,
    BaseMonitorAction,
    EventRecord,
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

        context_env = getattr(context, "env", None)
        env = {**context_env} if context_env else None
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
    "FinishAction",
    "FinishActionConfig",
    "RunAutoexpAction",
    "RunAutoexpActionConfig",
]
