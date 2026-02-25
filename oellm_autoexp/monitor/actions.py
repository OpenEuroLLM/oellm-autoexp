"""Actions and event definitions for monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import logging
import re
import time
from typing import Any, Literal

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
    MissingValue,
)

from oellm_autoexp.monitor.conditions import MonitorConditionInterface
from oellm_autoexp.monitor.utils.template import replace_braced_keys

LOGGER = logging.getLogger(__name__)


@register_interface
class JobInterface(RegistrableConfigInterface):
    """Registrable interface for job configurations."""


@dataclass(kw_only=True)
class ActionResult:
    """Outcome of an action execution."""

    special: Literal["cancel", "finish", "restart", "noop"] = "noop"
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "success"  # For tracking action execution status
    action_config: BaseMonitorAction.cfgtype | None = (
        None  # Reference to the action's config for typed access
    )


@dataclass(kw_only=True)
class EventRecord:
    """Persistent record of a detected event and its action history."""

    event_id: str
    name: str
    source: str
    count: int = 1
    first_seen_ts: float = field(default_factory=time.time)
    last_seen_ts: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def touch(self, *, payload: dict[str, Any] | None = None) -> None:
        """Increment the occurrence counter and update timestamps."""
        self.count += 1
        self.last_seen_ts = time.time()
        if payload:
            self.payload.update(payload)

    def set_status(self, *, note: str | None = None) -> None:
        """Move event into a new lifecycle state and append optional note."""
        self.last_seen_ts = time.time()
        if note:
            self.history.append({"ts": self.last_seen_ts, "note": note})  # pragma: no cover


def event_key(
    job_id: str, event_name: str, metadata: dict[str, Any] | None = None
) -> tuple[str, str]:
    h = hashlib.md5()
    h.update(json.dumps(metadata).encode("utf8"))
    h = str(h.digest())[:16]
    return (str(job_id), event_name, h)


def build_event_id(
    job_id: str,
    event_name: str,
    metadata: dict[str, Any] | None = None,
    *,
    now_ms: int | None = None,
) -> str:
    timestamp = int(time.time() * 1000) if now_ms is None else now_ms
    if metadata and "checkpoint_iteration" in metadata:
        return f"{job_id}:{event_name}:{metadata['checkpoint_iteration']}:{timestamp}"
    return f"{job_id}:{event_name}:{timestamp}"


@dataclass(kw_only=True)
class ActionContext:
    event: EventRecord
    job_metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    workspace: Path | None = None

    @property
    def variables(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self.job_metadata)
        merged.update(self.event.metadata)
        merged.update(self.event.payload)
        merged.setdefault("event_id", self.event.event_id)
        merged.setdefault("event_name", self.event.name)
        merged.setdefault("attempts", self.attempts)
        if self.workspace:
            merged.setdefault("workspace", str(self.workspace))
        return merged

    def render(self, template: str) -> str:
        try:
            return replace_braced_keys(template, self.variables)
        except KeyError:  # pragma: no cover
            return template


@register_interface
class BaseMonitorAction(RegistrableConfigInterface):
    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    def execute(self, context: ActionContext) -> ActionResult:  # pragma: no cover
        raise NotImplementedError


@dataclass
class LogActionConfig(ConfigInterface):
    class_name: str = "LogAction"
    message: str = "Event {event_name} triggered"
    level: str = "info"


@register
class LogAction(BaseMonitorAction):
    config: LogActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        msg = context.render(self.config.message)
        level = self.config.level.lower()
        if level == "debug":
            LOGGER.debug(msg)
        elif level == "warning":
            LOGGER.warning(msg)
        elif level == "error":
            LOGGER.error(msg)
        else:
            LOGGER.info(msg)
        return ActionResult(message=msg)


@dataclass
class NewJobActionConfig(ConfigInterface):
    class_name: str = "NewJobAction"
    job_config: JobInterface.cfgtype = field(default_factory=MissingValue)


@register
class NewJobAction(BaseMonitorAction):
    config: NewJobActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            status="success",
            message="Submitting local command job",
            action_config=self.config,
        )


@dataclass
class RestartActionConfig(ConfigInterface):
    class_name: str = "RestartAction"
    reason: str = "restarting job"


@register
class RestartAction(BaseMonitorAction):
    config: RestartActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            special="restart",
            status="success",
            message=self.config.reason,
        )


@dataclass
class FinishActionConfig(ConfigInterface):
    class_name: str = "FinishAction"
    reason: str = "finished"


@register
class FinishAction(BaseMonitorAction):
    config: FinishActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            special="finish",
            status="success",
            message=self.config.reason,
        )


@dataclass
class CancelActionConfig(ConfigInterface):
    class_name: str = "CancelAction"
    reason: str = "cancelled"


@register
class CancelAction(BaseMonitorAction):
    config: CancelActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            special="cancel",
            status="success",
            message=self.config.reason,
        )


@dataclass
class EventConfig:
    name: str = ""
    action: BaseMonitorAction.cfgtype | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    condition: MonitorConditionInterface.cfgtype | None = None


@dataclass
class LogEventConfig(EventConfig):
    """Configuration for a log-triggered event and action."""

    pattern: str = ""
    pattern_type: Literal["substring", "regex", "inactivity"] = "substring"
    extract_groups: dict[str, int | str] = field(default_factory=dict)
    match_once: bool = True


@dataclass
class StateEventConfig(EventConfig):
    """Configuration for a state-triggered event and action."""

    transition: tuple[str | None, str | None] = field(default_factory=MissingValue)

    def __post_init__(self):
        assert self.transition is not MissingValue


class LogEvent:
    """Handles log pattern matching and action execution."""

    config: LogEventConfig

    def __init__(self, config: LogEventConfig):
        self.config = config

    def check_triggers(self, log_text: str) -> list[dict[str, Any]]:
        """Check if event triggers in the given log text, return metadata for
        each match."""
        triggers = []
        if self.config.pattern_type == "inactivity":
            if log_text == "":
                metadata = dict(self.config.metadata)
                metadata["inactive"]
                triggers.append()
        else:
            for match in self._iter_matches(log_text):
                metadata = self._build_metadata(match, log_text)
                triggers.append(metadata)
        if self.config.match_once:
            triggers = triggers[:1]
        return triggers

    def _iter_matches(self, text: str) -> list:
        """Find all pattern matches in text."""
        if self.config.pattern_type == "regex":
            pattern = re.compile(self.config.pattern, flags=re.MULTILINE)
            return list(pattern.finditer(text))
        escaped = re.escape(self.config.pattern)
        pattern = re.compile(escaped, flags=re.MULTILINE)
        return list(pattern.finditer(text))

    def _build_metadata(self, match, text: str) -> dict[str, Any]:
        """Build metadata from a pattern match."""
        metadata = dict(self.config.metadata)
        metadata["match"] = match.group(0)
        metadata["line"] = match.string[match.start() : match.end()]

        # Extract groups based on configuration
        if self.config.extract_groups:
            for key, group in self.config.extract_groups.items():
                if isinstance(group, str) and group == "match":
                    metadata[key] = match.group(0)
                    continue
                try:
                    metadata[key] = match.group(group)
                except (IndexError, KeyError):
                    continue

        return metadata


class StateEvent:
    """Handles state transition matching and action execution."""

    config: StateEventConfig

    def __init__(self, config: StateEventConfig):
        self.config = config

    def check_trigger(self, old_status: str | None, new_status: str | None) -> bool:
        """Check if the state transition matches this event's transition."""
        expected_old, expected_new = self.config.transition
        old_status = old_status.lower() if old_status is not None else None
        new_status = new_status.lower() if new_status is not None else None
        expected_old = expected_old.lower() if expected_old is not None else None
        expected_new = expected_new.lower() if expected_new is not None else None
        # None in expected means "any state"
        old_matches = expected_old is None or old_status == expected_old
        new_matches = expected_new is None or new_status == expected_new
        return old_matches and new_matches

    def build_metadata(self, old_status: str | None, new_status: str | None) -> dict[str, Any]:
        """Build metadata for the triggered event."""
        metadata = dict(self.config.metadata)
        metadata["old_status"] = old_status
        metadata["new_status"] = new_status
        metadata["transition"] = f"{old_status} -> {new_status}"
        return metadata


__all__ = [
    "JobInterface",
    "EventRecord",
    "ActionResult",
    "event_key",
    "build_event_id",
    "ActionContext",
    "BaseMonitorAction",
    "LogAction",
    "NewJobAction",
    "NewJobActionConfig",
    "RestartActionConfig",
    "RestartAction",
    "FinishActionConfig",
    "FinishAction",
    "CancelActionConfig",
    "CancelAction",
    "LogEventConfig",
    "LogEvent",
    "StateEventConfig",
    "StateEvent",
]
