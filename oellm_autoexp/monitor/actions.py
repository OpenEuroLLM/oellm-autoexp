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
class AppendToFileActionConfig(ConfigInterface):
    """Configuration for an action that appends a line to a file.

    Designed to be chained from a ``LogEvent`` so a matched log line (e.g. a
    failing node name extracted via ``extract_groups``) can be persisted to an
    external list - for example the node-exclusion file consumed by the
    ``oc.exclude_nodes`` resolver.
    """

    class_name: str = "AppendToFileAction"
    # Target file path. Supports {var} templating against the action context.
    path: str = ""
    # Line to append. Supports {var} templating; defaults to the matched text.
    content: str = "{match}"
    # When True, skip appending if an identical (stripped) line already exists.
    dedup: bool = True
    # When True, create parent directories for ``path`` if they are missing.
    create_parents: bool = True


@register
class AppendToFileAction(BaseMonitorAction):
    config: AppendToFileActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        path = Path(context.render(self.config.path)).expanduser()
        line = context.render(self.config.content).strip()
        if not line:
            return ActionResult(status="failed", message="empty content, nothing appended")
        if self.config.create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.dedup and path.exists():
            existing = {entry.strip() for entry in path.read_text().splitlines()}
            if line in existing:
                return ActionResult(
                    status="success",
                    message=f"'{line}' already present in {path}",
                    metadata={"appended": False, "path": str(path), "line": line},
                )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return ActionResult(
            status="success",
            message=f"appended '{line}' to {path}",
            metadata={"appended": True, "path": str(path), "line": line},
        )


@dataclass
class UpdateChainExcludesActionConfig(ConfigInterface):
    """Propagate the current node-exclusion list to dependency-chained jobs.

    Designed to be chained from the same ``LogEvent`` that appends a failing node
    to ``exclude_file`` (list the AppendToFileAction event *before* this one so the
    file already contains the new node). The monitor loop reads ``exclude_file``
    and runs ``scontrol update JobId=.. ExcNodeList=..`` on every *pending* sibling
    job in the session, so the queued chain avoids the bad node without
    regenerating or resubmitting scripts. Running jobs cannot be edited live and
    are skipped (handle the failing job itself via a RestartAction).
    """

    class_name: str = "UpdateChainExcludesAction"
    # File holding the node-exclusion list (same one the AppendToFileAction writes
    # and the oc.exclude_nodes resolver reads). Supports {var} templating.
    exclude_file: str = ""


@register
class UpdateChainExcludesAction(BaseMonitorAction):
    config: UpdateChainExcludesActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        # The side effect (iterating the job store + calling scontrol) needs the
        # monitor's store and SLURM client, so signal the loop via action_config
        # and hand it the resolved exclude-file path (mirrors NewJobAction).
        exclude_file = context.render(self.config.exclude_file)
        return ActionResult(
            status="success",
            message="propagate excludes to pending chain jobs",
            action_config=self.config,
            metadata={"exclude_file": exclude_file},
        )


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
    pattern_type: Literal["substring", "regex", "inactivity", "progress"] = "substring"
    extract_groups: dict[str, int | str] = field(default_factory=dict)
    match_once: bool = True
    # For pattern_type == "inactivity": minimum number of consecutive polls
    # without new log output, AND minimum real elapsed seconds across that
    # streak, before the action fires. Both must hold (AND). Leaving a field at
    # its default disables that half of the check.
    inactivity_polls: int = 1
    inactivity_timeout_s: float = 0.0
    # For pattern_type == "progress": ``pattern`` is always treated as a REGEX
    # and must contain a capturing group holding a number that grows while the
    # job is healthy — Megatron's iteration counter, `iteration +([0-9]+)/`.
    # The streak counts consecutive polls in which that number did NOT move;
    # ``progress_polls`` / ``progress_timeout_s`` gate the action with the same
    # AND semantics as the inactivity pair above.
    #
    # WHY THIS IS NOT THE INACTIVITY CHECK: inactivity only asks whether the log
    # GREW. A job stuck in an ft_launcher restart loop keeps emitting fresh
    # setup banners, so it is never inactive — job 1375720 grew a 251 MB log
    # over 4 h while pinned at iteration 100 — but its counter does not move.
    progress_group: int | str = 1
    # increase: the streak resets whenever the newest value DIFFERS from the
    #   previous one. A resume from checkpoint jumps the counter BACKWARDS, and
    #   that still counts as movement, so this is safe with a short window. It
    #   detects "no iterations at all" and "iterations frozen".
    # max: the streak resets only when the running MAXIMUM advances. This also
    #   catches a loop that keeps re-running the same iterations forever, but
    #   the window must exceed the time needed to redo the work lost to the last
    #   rolling checkpoint, or a healthy restart trips it.
    progress_mode: Literal["increase", "max"] = "increase"
    progress_polls: int = 1
    progress_timeout_s: float = 0.0


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
        each match.

        Inactivity and progress events are streak based rather than
        match-once-and-fire, and are handled separately by the monitor loop (see
        ``inactivity_qualifies`` / ``progress_qualifies``); this method only
        matches substring/regex patterns.
        """
        if self.config.pattern_type in ("inactivity", "progress"):
            return []
        triggers = []
        for match in self._iter_matches(log_text):
            metadata = self._build_metadata(match, log_text)
            triggers.append(metadata)
        if self.config.match_once:
            triggers = triggers[:1]
        return triggers

    def inactivity_metadata(self) -> dict[str, Any]:
        """Stable metadata identifying this inactivity streak."""
        metadata = dict(self.config.metadata)
        metadata["inactive"] = True
        return metadata

    def inactivity_qualifies(self, *, count: int, elapsed_s: float) -> bool:
        """Return True when an inactivity streak has lasted long enough to act.

        ``count`` consecutive inactive polls AND ``elapsed_s`` real seconds must
        both meet their configured thresholds (each half disabled when its
        threshold is left at the default).
        """
        if count < self.config.inactivity_polls:
            return False
        if self.config.inactivity_timeout_s > 0 and elapsed_s < self.config.inactivity_timeout_s:
            return False
        return True

    def progress_metadata(self) -> dict[str, Any]:
        """Stable metadata identifying this progress streak.

        Must NOT contain the observed counter: ``event_key`` hashes this dict,
        so a value in here would mint a new streak every time the counter moved.
        The observed values live in the record's ``payload`` instead.
        """
        metadata = dict(self.config.metadata)
        metadata["progress"] = True
        return metadata

    def observe_progress(self, log_text: str) -> tuple[float | None, float | None, str | None]:
        """Extract the progress counter from *new* log text.

        Returns ``(last, max, raw)`` over every match in this poll's slice, or
        ``(None, None, None)`` when the text holds no usable match — which is
        itself the signal that nothing advanced. ``last`` drives
        ``progress_mode: increase`` (it is the job's current position, including
        after a backwards jump on resume) and ``max`` drives
        ``progress_mode: max``.
        """
        pattern = re.compile(self.config.pattern, flags=re.MULTILINE)
        values: list[float] = []
        raw: str | None = None
        for match in pattern.finditer(log_text):
            try:
                token = match.group(self.config.progress_group)
            except (IndexError, KeyError):
                continue
            if token is None:
                continue
            try:
                values.append(float(token))
            except (TypeError, ValueError):
                continue
            raw = token
        if not values:
            return None, None, None
        return values[-1], max(values), raw

    def progress_qualifies(self, *, count: int, elapsed_s: float) -> bool:
        """Return True when a no-progress streak has lasted long enough to act.

        ``count`` consecutive polls without movement AND ``elapsed_s`` real
        seconds must both meet their thresholds (each half disabled when its
        threshold is left at the default).
        """
        if count < self.config.progress_polls:
            return False
        if self.config.progress_timeout_s > 0 and elapsed_s < self.config.progress_timeout_s:
            return False
        return True

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
    "AppendToFileAction",
    "AppendToFileActionConfig",
    "UpdateChainExcludesAction",
    "UpdateChainExcludesActionConfig",
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
