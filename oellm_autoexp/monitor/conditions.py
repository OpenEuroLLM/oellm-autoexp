"""Conditions used to gate monitor actions."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import re
from collections.abc import Mapping


from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
)

from oellm_autoexp.monitor.events import EventRecord

LOGGER = logging.getLogger(__name__)

ConditionStatus = Literal["pass", "fail", "waiting"]


_PATTERN = re.compile(r"(?<!\$)\{([^\{\}\$:]+)\}")


def replace_braced_keys(s: str, values: Mapping[str, Any]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)  # keep as-is if missing

    return _PATTERN.sub(repl, s)


@register_interface
class MonitorConditionInterface(RegistrableConfigInterface):
    """Registrable interface for monitor conditions."""


@dataclass(kw_only=True)
class ConditionResult:
    status: ConditionStatus
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def waiting(self) -> bool:
        return self.status == "waiting"


@dataclass(kw_only=True)
class ConditionContext:
    event: EventRecord
    job_metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

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


class BaseCondition(MonitorConditionInterface):
    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def check(self, context: ConditionContext) -> ConditionResult:  # pragma: no cover
        raise NotImplementedError


@dataclass
class AlwaysTrueConditionConfig(ConfigInterface):
    class_name: str = "AlwaysTrueCondition"
    message: str = ""


@register
class AlwaysTrueCondition(BaseCondition):
    config: AlwaysTrueConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        return ConditionResult(status="pass", message=self.config.message)


@dataclass
class MaxAttemptsConditionConfig(ConfigInterface):
    class_name: str = "MaxAttemptsCondition"
    max_attempts: int = 1


@register
class MaxAttemptsCondition(BaseCondition):
    config: MaxAttemptsConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if context.attempts < self.config.max_attempts:
            return ConditionResult(status="pass")
        return ConditionResult(
            status="fail",
            message=f"attempts {context.attempts} >= limit {self.config.max_attempts}",
        )


@dataclass
class CooldownConditionConfig(ConfigInterface):
    class_name: str = "CooldownCondition"
    cooldown_seconds: float = 60.0
    note: str | None = None


@register
class CooldownCondition(BaseCondition):
    config: CooldownConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        last_ts = context.event.metadata.get("last_action_ts", context.event.last_seen_ts)
        elapsed = time.time() - float(last_ts)
        if elapsed >= self.config.cooldown_seconds:
            return ConditionResult(status="pass")
        remaining = self.config.cooldown_seconds - elapsed
        return ConditionResult(
            status="waiting",
            message=f"cooldown {remaining:.1f}s remaining",
        )


def _wait_for_predicate(
    predicate,
    *,
    blocking: bool,
    timeout_seconds: float,
    poll_interval_seconds: float,
    waiting_message: str,
) -> ConditionResult:
    start = time.time()
    while True:
        if predicate():
            return ConditionResult(status="pass")
        if not blocking:
            return ConditionResult(status="waiting", message=waiting_message)
        if timeout_seconds and (time.time() - start) >= timeout_seconds:
            return ConditionResult(status="fail", message=f"timeout waiting for {waiting_message}")
        time.sleep(max(poll_interval_seconds, 0.1))


@dataclass
class FileExistsConditionConfig(ConfigInterface):
    class_name: str = "FileExistsCondition"
    path: str = ""
    blocking: bool = False
    timeout_seconds: float = 600.0
    poll_interval_seconds: float = 5.0


@register
class FileExistsCondition(BaseCondition):
    config: FileExistsConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        rendered = context.render(self.config.path)
        target = Path(rendered).expanduser()

        def predicate() -> bool:
            return target.exists()

        return _wait_for_predicate(
            predicate,
            blocking=self.config.blocking,
            timeout_seconds=self.config.timeout_seconds,
            poll_interval_seconds=self.config.poll_interval_seconds,
            waiting_message=f"file {target} missing",
        )


@dataclass
class GlobExistsConditionConfig(ConfigInterface):
    class_name: str = "GlobExistsCondition"
    pattern: str = ""
    min_matches: int = 1
    blocking: bool = False
    timeout_seconds: float = 600.0
    poll_interval_seconds: float = 5.0


@register
class GlobExistsCondition(BaseCondition):
    config: GlobExistsConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        rendered = context.render(self.config.pattern)
        path = Path(rendered).expanduser()

        def predicate() -> bool:
            return len(list(path.parent.glob(path.name))) >= self.config.min_matches

        return _wait_for_predicate(
            predicate,
            blocking=self.config.blocking,
            timeout_seconds=self.config.timeout_seconds,
            poll_interval_seconds=self.config.poll_interval_seconds,
            waiting_message=f"glob {rendered} missing",
        )


@dataclass
class CommandConditionConfig(ConfigInterface):
    class_name: str = "CommandCondition"
    command: list[str] = field(default_factory=list)


@register
class CommandCondition(BaseCondition):
    config: CommandConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.command:
            return ConditionResult(status="fail", message="no command supplied")
        rendered = [context.render(segment) for segment in self.config.command]
        proc = subprocess.run(rendered, capture_output=True, text=True)
        if proc.returncode == 0:
            return ConditionResult(status="pass")
        return ConditionResult(
            status="fail",
            message=f"command exited with {proc.returncode}: {proc.stderr.strip()}",
        )


@dataclass
class CompositeConditionConfig(ConfigInterface):
    class_name: str = "CompositeCondition"
    mode: Literal["all", "any"] = "all"
    conditions: list[MonitorConditionInterface.cfgtype] = field(default_factory=list)


@register
class CompositeCondition(BaseCondition):
    config: CompositeConditionConfig

    def __init__(self, config: CompositeConditionConfig) -> None:
        super().__init__(config)
        self._children = [
            condition.instantiate(MonitorConditionInterface) for condition in config.conditions
        ]

    def check(self, context: ConditionContext) -> ConditionResult:
        results: list[ConditionResult] = [child.check(context) for child in self._children]
        if self.config.mode == "all":
            if all(result.passed for result in results):
                return ConditionResult(status="pass")
            waiting = next((r for r in results if r.waiting), None)
            if waiting:
                return waiting
            failed = next((r for r in results if not r.passed), None)
            return failed or ConditionResult(status="fail", message="unknown composite failure")

        # mode == "any"
        if any(result.passed for result in results):
            return ConditionResult(status="pass")
        waiting = next((r for r in results if r.waiting), None)
        if waiting:
            return waiting
        return ConditionResult(status="fail", message="all child conditions failed")


@dataclass
class MetadataConditionConfig(ConfigInterface):
    class_name: str = "MetadataCondition"
    key: str = ""
    equals: Any | None = None
    within: list[Any] | None = None


@register
class MetadataCondition(BaseCondition):
    config: MetadataConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.key:
            return ConditionResult(status="fail", message="metadata key missing")
        value = context.event.metadata.get(self.config.key)
        if value is None:
            return ConditionResult(
                status="fail", message=f"metadata key '{self.config.key}' not present"
            )
        if self.config.equals is not None and value != self.config.equals:
            return ConditionResult(
                status="fail",
                message=f"metadata key '{self.config.key}' != {self.config.equals!r}",
            )
        if self.config.within is not None and value not in self.config.within:
            return ConditionResult(
                status="fail",
                message=f"metadata key '{self.config.key}' not in {self.config.within!r}",
            )
        return ConditionResult(status="pass")


@dataclass
class SlurmStateConditionConfig(ConfigInterface):
    """Check if a job (by name) is in a specific SLURM state.

    Used for cancel_conditions to detect when a sibling job has failed.
    Requires 'slurm_snapshot' in context.extra to check job states.
    """

    class_name: str = "SlurmStateCondition"
    job_name: str = ""  # Name of the job to check (supports {var} interpolation)
    state: str = "FAILED"  # SLURM state to match (FAILED, CANCELLED, TIMEOUT, etc.)


@register
class SlurmStateCondition(BaseCondition):
    config: SlurmStateConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.job_name:
            return ConditionResult(status="fail", message="job_name not specified")

        rendered_name = context.render(self.config.job_name)
        target_state = self.config.state.upper()

        # Get SLURM snapshot from context.extra (must be provided by caller)
        job_states = context.extra.get("job_states_by_name", {})

        # Look up job state by name
        job_state = job_states.get(rendered_name)
        if job_state is None:
            # Job not found in snapshot - could be completed/removed
            # Check if we're looking for terminal states
            if target_state in {"FAILED", "CANCELLED", "TIMEOUT", "COMPLETED"}:
                # Job disappeared - might have failed, return waiting
                return ConditionResult(
                    status="waiting",
                    message=f"job '{rendered_name}' not found in SLURM queue",
                )
            return ConditionResult(
                status="fail",
                message=f"job '{rendered_name}' not found",
            )

        if job_state == target_state:
            return ConditionResult(
                status="pass",
                message=f"job '{rendered_name}' is in state {target_state}",
            )

        return ConditionResult(
            status="fail",
            message=f"job '{rendered_name}' is in state {job_state}, not {target_state}",
        )


@dataclass
class LogPatternConditionConfig(ConfigInterface):
    """Check if a log file contains a specific pattern (regex).

    Used for cancel_conditions to detect errors in sibling job logs.
    """

    class_name: str = "LogPatternCondition"
    log_path: str = ""  # Path to log file (supports {var} interpolation)
    pattern: str = ""  # Regex pattern to search for
    tail_lines: int = 1000  # Only check last N lines for efficiency


@register
class LogPatternCondition(BaseCondition):
    config: LogPatternConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.log_path:
            return ConditionResult(status="fail", message="log_path not specified")
        if not self.config.pattern:
            return ConditionResult(status="fail", message="pattern not specified")

        rendered_path = context.render(self.config.log_path)
        log_file = Path(rendered_path).expanduser()

        if not log_file.exists():
            return ConditionResult(
                status="fail",
                message=f"log file '{log_file}' does not exist",
            )

        try:
            # Read last N lines efficiently
            with open(log_file, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                tail = lines[-self.config.tail_lines :] if lines else []
                content = "".join(tail)
        except OSError as exc:
            return ConditionResult(
                status="fail",
                message=f"failed to read log file: {exc}",
            )

        pattern = re.compile(self.config.pattern)
        match = pattern.search(content)
        if match:
            return ConditionResult(
                status="pass",
                message=f"pattern '{self.config.pattern}' found in log: {match.group()[:100]}",
            )

        return ConditionResult(
            status="fail",
            message=f"pattern '{self.config.pattern}' not found in log",
        )


__all__ = [
    "MonitorConditionInterface",
    "ConditionContext",
    "ConditionResult",
    "ConditionStatus",
    "AlwaysTrueCondition",
    "MaxAttemptsCondition",
    "CooldownCondition",
    "FileExistsCondition",
    "GlobExistsCondition",
    "CommandCondition",
    "CompositeCondition",
    "MetadataCondition",
    "SlurmStateCondition",
    "LogPatternCondition",
]
