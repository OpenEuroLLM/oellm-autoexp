"""Conditions used to gate monitor actions.

This module now delegates to the 'monitor' library.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from compoconf import register, ConfigInterface

from oellm_autoexp.monitor.conditions import (
    AlwaysTrueCondition,
    BaseCondition,
    CommandCondition,
    CompositeCondition,
    ConditionContext,
    ConditionResult,
    CooldownCondition,
    FileExistsCondition,
    GlobExistsCondition,
    MaxAttemptsCondition,
    MetadataCondition,
    MonitorConditionInterface,
)

LOGGER = logging.getLogger(__name__)

# Re-implement Condition Configs for accessibility


@dataclass
class ConditionConfigMixin:
    persistent_pass: bool = False
    persistent_fail: bool = False


@dataclass
class BlockingFileExistsConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "BlockingFileExistsCondition"
    path: str = ""
    blocking: bool = True
    timeout_seconds: float = 600.0
    poll_interval_seconds: float = 5.0


@register
class BlockingFileExistsCondition(BaseCondition):
    config: BlockingFileExistsConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        rendered = context.render(self.config.path)
        target = Path(rendered).expanduser()

        if target.exists():
            return ConditionResult(passed=True)
        return ConditionResult(passed=False, message=f"file {target} missing")


@dataclass
class CommandConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "CommandCondition"
    command: list[str] = field(default_factory=list)


@dataclass
class CompositeConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "CompositeCondition"
    mode: Literal["all", "any"] = "all"
    conditions: list[Any] = field(default_factory=list)


@dataclass
class MetadataConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "MetadataCondition"
    key: str = ""
    equals: Any | None = None
    within: list[Any] | None = None


# Re-implement SLURM-specific conditions


@dataclass
class SlurmStateConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "SlurmStateCondition"
    job_name: str = ""
    state: str = "FAILED"


@register
class SlurmStateCondition(BaseCondition):
    config: SlurmStateConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.job_name:
            return ConditionResult(passed=False, message="job_name not specified")

        rendered_name = context.render(self.config.job_name)
        target_state = self.config.state.upper()

        # job_states_by_name passed in context.extra
        job_states = context.extra.get("job_states_by_name", {})
        job_state = job_states.get(rendered_name)

        if job_state is None:
            # Job not found
            if target_state in {"FAILED", "CANCELLED", "TIMEOUT", "COMPLETED"}:
                # If checking for terminal state, missing job is ambiguous (could be old)
                # We treat it as not passed (waiting)
                return ConditionResult(
                    passed=False,
                    message=f"job '{rendered_name}' not found in SLURM queue",
                )
            # If checking for running state, missing means failed
            return ConditionResult(
                passed=False,
                message=f"job '{rendered_name}' not found",
            )

        if job_state == target_state:
            return ConditionResult(
                passed=True,
                message=f"job '{rendered_name}' is in state {target_state}",
            )

        return ConditionResult(
            passed=False,
            message=f"job '{rendered_name}' is in state {job_state}, not {target_state}",
        )


@dataclass
class LogPatternConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "LogPatternCondition"
    log_path: str = ""
    pattern: str = ""
    tail_lines: int = 1000


@register
class LogPatternCondition(BaseCondition):
    config: LogPatternConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.log_path:
            return ConditionResult(passed=False, message="log_path not specified")
        if not self.config.pattern:
            return ConditionResult(passed=False, message="pattern not specified")

        rendered_path = context.render(self.config.log_path)
        log_file = Path(rendered_path).expanduser()

        if not log_file.exists():
            return ConditionResult(
                passed=False,
                message=f"log file '{log_file}' does not exist",
            )

        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                # Efficiently read last N lines
                lines = f.readlines()
                tail = lines[-self.config.tail_lines :] if lines else []
                content = "".join(tail)
        except OSError as exc:
            return ConditionResult(
                passed=False,
                message=f"failed to read log file: {exc}",
            )

        pattern = re.compile(self.config.pattern)
        match = pattern.search(content)
        if match:
            return ConditionResult(
                passed=True,
                message=f"pattern '{self.config.pattern}' found in log",
            )

        return ConditionResult(
            passed=False,
            message=f"pattern '{self.config.pattern}' not found in log",
        )


__all__ = [
    "AlwaysTrueCondition",
    "BaseCondition",
    "BlockingFileExistsCondition",
    "BlockingFileExistsConditionConfig",
    "CommandCondition",
    "CommandConditionConfig",
    "CompositeCondition",
    "CompositeConditionConfig",
    "ConditionContext",
    "ConditionResult",
    "CooldownCondition",
    "FileExistsCondition",
    "GlobExistsCondition",
    "LogPatternCondition",
    "LogPatternConditionConfig",
    "MaxAttemptsCondition",
    "MetadataCondition",
    "MetadataConditionConfig",
    "MonitorConditionInterface",
    "SlurmStateCondition",
    "SlurmStateConditionConfig",
]
