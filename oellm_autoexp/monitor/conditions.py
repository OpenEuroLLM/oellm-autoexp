"""Conditions used to gate monitor actions."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import re


from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oellm_autoexp.monitor.actions import EventRecord
from oellm_autoexp.monitor.utils.template import replace_braced_keys

LOGGER = logging.getLogger(__name__)


@dataclass
class ConditionResult:
    passed: bool
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.passed


@register_interface
class MonitorConditionInterface(RegistrableConfigInterface):
    """Registrable interface for monitor conditions."""


@dataclass(kw_only=True)
class ConditionContext:
    event: EventRecord | None = None
    job_metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    started_ts: float | None = None

    @property
    def variables(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self.job_metadata)
        if self.event:
            merged.update(self.event.metadata)  # pragma: no cover
            merged.update(self.event.payload)  # pragma: no cover
            merged.setdefault("event_id", self.event.event_id)  # pragma: no cover
            merged.setdefault("event_name", self.event.name)  # pragma: no cover
        return merged

    def render(self, template: str) -> str:
        try:
            return replace_braced_keys(template, self.variables)
        except KeyError:  # pragma: no cover
            return template


class BaseCondition(MonitorConditionInterface):
    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def check(self, context: ConditionContext) -> ConditionResult:  # pragma: no cover
        raise NotImplementedError


@dataclass
class ConditionConfigMixin:
    persistent_pass: bool = False
    persistent_fail: bool = False


@dataclass
class AlwaysTrueConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "AlwaysTrueCondition"
    message: str = ""


@register
class AlwaysTrueCondition(BaseCondition):
    config: AlwaysTrueConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        return ConditionResult(passed=True, message=self.config.message)


@dataclass
class MaxAttemptsConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "MaxAttemptsCondition"
    max_attempts: int = 1


@register
class MaxAttemptsCondition(BaseCondition):
    config: MaxAttemptsConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if context.attempts < self.config.max_attempts:
            return ConditionResult(passed=True)
        return ConditionResult(
            passed=False,
            message=f"attempts {context.attempts} >= limit {self.config.max_attempts}",
        )


@dataclass
class CooldownConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "CooldownCondition"
    cooldown_seconds: float = 60.0
    note: str | None = None


@register
class CooldownCondition(BaseCondition):
    config: CooldownConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if context.event is None:
            return ConditionResult(passed=True)  # No event means no cooldown needed
        last_ts = context.event.metadata.get("last_action_ts", context.event.last_seen_ts)
        elapsed = time.time() - float(last_ts)
        if elapsed >= self.config.cooldown_seconds:
            return ConditionResult(passed=True)
        remaining = self.config.cooldown_seconds - elapsed
        return ConditionResult(
            passed=False,
            message=f"cooldown {remaining:.1f}s remaining",
        )


@dataclass
class TimeoutConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "TimeoutCondition"
    timeout_seconds: float = 600.0
    message: str = "timeout waiting for condition"


@register
class TimeoutCondition(BaseCondition):
    config: TimeoutConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if context.started_ts is None:
            context.state["started_ts"] = time.time()
            context.started_ts = context.state["started_ts"]
        elapsed = time.time() - float(context.started_ts)
        if elapsed >= self.config.timeout_seconds:
            return ConditionResult(passed=False, message=self.config.message)
        remaining = self.config.timeout_seconds - elapsed
        return ConditionResult(
            passed=True, message=f"{self.config.message} ({remaining:.1f}s remaining)"
        )


@dataclass
class FileExistsConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "FileExistsCondition"
    path: str = ""


@register
class FileExistsCondition(BaseCondition):
    config: FileExistsConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        rendered = context.render(self.config.path)
        target = Path(rendered).expanduser()

        if target.exists():
            return ConditionResult(passed=True)
        return ConditionResult(passed=False, message=f"file {target} missing")


@dataclass
class GlobExistsConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "GlobExistsCondition"
    pattern: str = ""
    min_matches: int = 1


@register
class GlobExistsCondition(BaseCondition):
    config: GlobExistsConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        rendered = context.render(self.config.pattern)
        path = Path(rendered).expanduser()
        count = len(list(path.parent.glob(path.name)))
        if count >= self.config.min_matches:
            return ConditionResult(passed=True)
        return ConditionResult(passed=False, message=f"glob {rendered} missing")


@dataclass
class FileContentConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "FileContentCondition"
    path: str = ""
    pattern: str = ""
    mode: Literal["contains", "regex"] = "contains"


@register
class FileContentCondition(BaseCondition):
    config: FileContentConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        rendered_path = context.render(self.config.path)
        path = Path(rendered_path).expanduser()

        if not path.exists():
            return ConditionResult(passed=False, message=f"file {path} missing")
        try:
            content = path.read_text()
        except OSError:  # pragma: no cover
            return ConditionResult(passed=False, message=f"file {path} read failed")
        if self.config.mode == "contains":
            matched = self.config.pattern in content
        elif self.config.mode == "regex":
            matched = bool(re.search(self.config.pattern, content))
        else:
            matched = False  # pragma: no cover
        if matched:
            return ConditionResult(passed=True)
        return ConditionResult(passed=False, message=f"file {path} content match failed")


@dataclass
class CommandConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "CommandCondition"
    command: list[str] = field(default_factory=list)


@register
class CommandCondition(BaseCondition):
    config: CommandConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.command:
            return ConditionResult(passed=False, message="no command supplied")  # pragma: no cover
        rendered = [context.render(segment) for segment in self.config.command]
        proc = subprocess.run(rendered, capture_output=True, text=True)
        if proc.returncode == 0:
            return ConditionResult(passed=True)
        return ConditionResult(
            passed=False,
            message=f"command exited with {proc.returncode}: {proc.stderr.strip()}",
        )


@dataclass
class ShellCommandConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "ShellCommandCondition"
    command: str = ""


@register
class ShellCommandCondition(BaseCondition):
    config: ShellCommandConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        rendered = context.render(self.config.command)
        proc = subprocess.run(rendered, shell=True, capture_output=True, text=True)
        if proc.returncode == 0:
            return ConditionResult(passed=True)
        return ConditionResult(
            passed=False,
            message=f"shell command exited with {proc.returncode}",
        )


@dataclass
class CompositeConditionConfig(ConditionConfigMixin, ConfigInterface):
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
        child_states = context.state.setdefault("conditions", {})
        results: list[ConditionResult] = []
        for idx, child in enumerate(self._children):
            child_state = child_states.setdefault(str(idx), {})
            child_ctx = ConditionContext(
                event=context.event,
                job_metadata=context.job_metadata,
                attempts=context.attempts,
                extra=context.extra,
                state=child_state,
                started_ts=child_state.get("started_ts"),
            )
            results.append(child.check(child_ctx))
        if self.config.mode == "all":
            if all(result.passed for result in results):
                return ConditionResult(passed=True)
            failed = next((r for r in results if not r.passed), None)
            return failed or ConditionResult(
                passed=False, message="unknown composite failure"
            )  # pragma: no cover

        # mode == "any"
        if any(result.passed for result in results):
            return ConditionResult(passed=True)
        failed = next((r for r in results if not r.passed), None)
        return failed or ConditionResult(
            passed=False, message="all child conditions failed"
        )  # pragma: no cover


@dataclass
class AndConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "AndCondition"
    conditions: list[MonitorConditionInterface.cfgtype] = field(default_factory=list)


@register
class AndCondition(CompositeCondition):
    config: AndConditionConfig

    def __init__(self, config: AndConditionConfig) -> None:
        composite_config = CompositeConditionConfig(
            mode="all",
            conditions=config.conditions,
        )
        super().__init__(composite_config)


@dataclass
class OrConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "OrCondition"
    conditions: list[MonitorConditionInterface.cfgtype] = field(default_factory=list)


@register
class OrCondition(CompositeCondition):
    config: OrConditionConfig

    def __init__(self, config: OrConditionConfig) -> None:
        composite_config = CompositeConditionConfig(
            mode="any",
            conditions=config.conditions,
        )
        super().__init__(composite_config)


@dataclass
class NotConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "NotCondition"
    condition: MonitorConditionInterface.cfgtype = field(default_factory=dict)


@register
class NotCondition(BaseCondition):
    config: NotConditionConfig

    def __init__(self, config: NotConditionConfig) -> None:
        super().__init__(config)
        self._child = config.condition.instantiate(MonitorConditionInterface)

    def check(self, context: ConditionContext) -> ConditionResult:
        child_state = context.state.setdefault("condition", {})
        child_ctx = ConditionContext(
            event=context.event,
            job_metadata=context.job_metadata,
            attempts=context.attempts,
            extra=context.extra,
            state=child_state,
            started_ts=child_state.get("started_ts"),
        )
        result = self._child.check(child_ctx)
        if result.passed:
            return ConditionResult(
                passed=False, message=f"NOT condition failed: child {result.message}"
            )
        return ConditionResult(passed=True)


@dataclass
class MetadataConditionConfig(ConditionConfigMixin, ConfigInterface):
    class_name: str = "MetadataCondition"
    key: str = ""
    equals: Any | None = None
    within: list[Any] | None = None


@register
class MetadataCondition(BaseCondition):
    config: MetadataConditionConfig

    def check(self, context: ConditionContext) -> ConditionResult:
        if not self.config.key:
            return ConditionResult(passed=False, message="metadata key missing")  # pragma: no cover
        value = context.event.metadata.get(self.config.key)
        if value is None:
            return ConditionResult(
                passed=False, message=f"metadata key '{self.config.key}' not present"
            )
        if self.config.equals is not None and value != self.config.equals:
            return ConditionResult(
                passed=False,
                message=f"metadata key '{self.config.key}' != {self.config.equals!r}",
            )
        if self.config.within is not None and value not in self.config.within:
            return ConditionResult(
                passed=False,
                message=f"metadata key '{self.config.key}' not in {self.config.within!r}",
            )
        return ConditionResult(passed=True)


__all__ = [
    "MonitorConditionInterface",
    "ConditionContext",
    "ConditionResult",
    "AlwaysTrueCondition",
    "MaxAttemptsCondition",
    "CooldownCondition",
    "TimeoutCondition",
    "FileExistsCondition",
    "FileContentCondition",
    "GlobExistsCondition",
    "CommandCondition",
    "ShellCommandCondition",
    "CompositeCondition",
    "AndCondition",
    "OrCondition",
    "NotCondition",
    "MetadataCondition",
]
