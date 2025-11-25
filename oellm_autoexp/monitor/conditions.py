"""Conditions used to gate monitor actions."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
)

from oellm_autoexp.monitor.events import EventRecord

ConditionStatus = Literal["pass", "fail", "waiting"]


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
            return template.format(**self.variables)
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
        return ConditionResult(status="pass")


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
]
