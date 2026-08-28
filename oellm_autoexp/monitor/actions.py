"""Actions and event definitions for monitor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import logging
import re
import sys
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


def _ensure_megatron_importable() -> None:
    """Put the Megatron submodule on sys.path if it is not already importable.

    A torch_dist ``.metadata`` is a pickle referencing megatron.core classes, so
    ``read_metadata()`` raises ModuleNotFoundError unless megatron is importable.
    The monitor venv does NOT have it by default — verified on JUPITER, where the
    check only succeeded with ``PYTHONPATH=.:submodules/Megatron-LM``. Rather
    than require callers to set the environment, resolve it relative to this
    file so the check works however the monitor was launched.
    """
    try:
        import megatron.core  # noqa: F401

        return
    except Exception:
        pass
    candidate = Path(__file__).resolve().parents[2] / "submodules" / "Megatron-LM"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))


def _torch_dist_missing_files(path: Path) -> list[str] | None:
    """Return shard files that ``.metadata`` promises but that are absent.

    ``None`` means the metadata could not be read at all, which callers MUST
    treat as "unverifiable" rather than "fine" — see ``checkpoint_is_complete``.
    """
    try:  # imported lazily: the monitor must not hard-depend on torch
        _ensure_megatron_importable()
        from torch.distributed.checkpoint import FileSystemReader
    except Exception:  # pragma: no cover - torch not installed in this env
        return None
    try:
        metadata = FileSystemReader(path).read_metadata()
        expected = {info.relative_path for info in metadata.storage_data.values()}
    except Exception:  # pragma: no cover - truncated/corrupt metadata
        return None
    return sorted(name for name in expected if not (path / name).exists())


def checkpoint_status(path: Path) -> tuple[str, str]:
    """Classify a torch_dist checkpoint as complete / incomplete /
    unverifiable.

    THREE STATES, NOT TWO, and the third one is the point.

    `.metadata` alone is not sufficient: on 2026-08-19 ``iter_0003000`` carried a
    valid `.metadata` while 927 of its shards were absent and every rank died
    with FileNotFoundError. So the promised file list is checked against disk.

    But when the metadata cannot be READ we must not guess. An earlier version
    fell back to comparing shard counts against sibling ``iter_*`` directories,
    which is actively dangerous: a tree can legitimately hold checkpoints written
    at DIFFERENT WORLD SIZES (2026-08-19: 4096 files at world 2048 next to 8192
    files at world 4096), and the heuristic would call a perfectly complete
    checkpoint "short" and quarantine good state. Unverifiable now stops the
    machinery and asks for a human instead.
    """
    if not path.is_dir():
        return "incomplete", "not a directory"
    if not (path / ".metadata").is_file():
        # Written LAST by the coordinator, so its absence means the save never
        # finished. NB it is a dotfile: plain `ls` hides it.
        return "incomplete", "no .metadata (save never finalised)"
    # NB pathlib's "*" DOES match leading dots, and the in-flight shards are
    # written as ".__<rank>_<n>.distcp.tmp", so dedupe rather than concatenate.
    leftovers = set(path.glob("*.distcp.tmp")) | set(path.glob(".*.distcp.tmp"))
    if leftovers:
        return "incomplete", f"{len(leftovers)} unfinished .tmp shards"
    if not (path / "common.pt").is_file():
        return "incomplete", "no common.pt"

    missing = _torch_dist_missing_files(path)
    if missing is None:
        return "unverifiable", ".metadata present but unreadable (torch/megatron unavailable?)"
    if missing:
        return "incomplete", f"{len(missing)} shard(s) named in .metadata are absent"
    return "complete", f"verified against .metadata ({len(list(path.glob('*.distcp')))} shards)"


def checkpoint_is_complete(path: Path) -> tuple[bool, str]:
    """Backwards-compatible boolean wrapper.

    Unverifiable counts as NOT complete.
    """
    status, reason = checkpoint_status(path)
    return status == "complete", reason


@dataclass
class QuarantineCheckpointActionConfig(ConfigInterface):
    """Move a corrupt checkpoint aside and roll the tracker back to a valid
    one.

    Automates the manual recovery: rename ``iter_N`` to ``failed_iter_N`` and
    rewrite ``latest_checkpointed_iteration.txt`` to the newest checkpoint that
    actually loads, so a following ``RestartAction`` resumes instead of dying on
    the same file again.

    ONLY FOR GENUINELY INCOMPLETE CHECKPOINTS (missing shards / no ``.metadata``).
    Do NOT wire this to a world-size mismatch — ``rerun_state_machine_state/
    shard_<rank>_<world>`` missing means the checkpoint is FINE and the job's node
    count is wrong; quarantining would discard good state and roll back for
    nothing. That signature gets its own CancelAction.

    Chain it BEFORE the RestartAction for the same pattern, as
    config/job/autoexclude.yaml does for node exclusion.
    """

    class_name: str = "QuarantineCheckpointAction"
    # Checkpoint tree holding the iter_* directories and the tracker file.
    # Supports {var} templating.
    checkpoint_dir: str = ""
    # Iteration to quarantine; normally "{iteration}" captured via extract_groups.
    # If it does not resolve to an int, the tracker's current value is used.
    iteration: str = "{iteration}"
    quarantine_prefix: str = "failed_"
    tracker_name: str = "latest_checkpointed_iteration.txt"
    # Refuse to quarantine once this many checkpoints have already been moved
    # aside in this tree. Without a cap, a persistent fault walks backwards
    # through every checkpoint one restart at a time.
    max_rollbacks: int = 2
    # Report what would happen and change nothing.
    dry_run: bool = False


@register
class QuarantineCheckpointAction(BaseMonitorAction):
    config: QuarantineCheckpointActionConfig

    def execute(self, context: ActionContext) -> ActionResult:
        root = Path(context.render(self.config.checkpoint_dir)).expanduser()
        prefix = self.config.quarantine_prefix
        tracker = root / self.config.tracker_name

        if not root.is_dir():
            return ActionResult(status="failed", message=f"checkpoint dir not found: {root}")

        already = sorted(p.name for p in root.glob(f"{prefix}iter_*"))
        if len(already) >= self.config.max_rollbacks:
            return ActionResult(
                status="failed",
                message=(
                    f"refusing to quarantine: {len(already)} checkpoint(s) already moved aside "
                    f"in {root} (max_rollbacks={self.config.max_rollbacks}). Investigate manually."
                ),
                metadata={"quarantined": already},
            )

        # Which iteration is bad? Prefer the captured group, fall back to tracker.
        raw = context.render(self.config.iteration).strip()
        bad_iter: int | None = None
        if raw.isdigit():
            bad_iter = int(raw)
        elif tracker.is_file():
            current = tracker.read_text().strip()
            if current.isdigit():
                bad_iter = int(current)
        if bad_iter is None:
            return ActionResult(
                status="failed",
                message=f"could not determine the failing iteration (raw={raw!r}, tracker={tracker})",
            )

        bad_dir = root / f"iter_{bad_iter:07d}"

        # CHOOSE THE RESUME POINT BEFORE TOUCHING ANYTHING. If the rename came
        # first and the search then refused, the tracker would be left pointing
        # at a directory that no longer exists — a worse state than the one we
        # started in. Nothing is mutated until a verified target is in hand.
        candidates = sorted(
            (p for p in root.glob("iter_*") if p.is_dir() and p != bad_dir),
            key=lambda p: int(p.name.split("_")[-1]),
            reverse=True,
        )
        actions: list[str] = []
        rejected: list[str] = []
        for candidate in candidates:
            status, why = checkpoint_status(candidate)
            if status == "unverifiable":
                # STOP. Do not skip past it (it may be perfectly good, and
                # rolling further back would discard real training), and do not
                # resume from it (it may be broken). Neither guess is safe, so
                # hand it to a human with the reason.
                return ActionResult(
                    status="failed",
                    message=(
                        f"cannot verify {candidate.name} in {root}: {why}. "
                        f"Refusing to choose a resume point. Check it by hand with "
                        f'`python -c "from torch.distributed.checkpoint import FileSystemReader; '
                        f"print(FileSystemReader('{candidate}').read_metadata())\"` "
                        f"(needs megatron.core importable), then set "
                        f"{self.config.tracker_name} yourself."
                    ),
                    metadata={
                        "quarantined_iteration": bad_iter,
                        "unverifiable": candidate.name,
                        "rejected": rejected,
                    },
                )
            if status == "complete":
                iteration = int(candidate.name.split("_")[-1])
                # Verified target in hand — now, and only now, mutate the tree.
                if bad_dir.is_dir():
                    moved = root / f"{prefix}iter_{bad_iter:07d}"
                    if self.config.dry_run:
                        actions.append(f"would rename {bad_dir.name} -> {moved.name}")
                    else:
                        bad_dir.rename(moved)
                        actions.append(f"renamed {bad_dir.name} -> {moved.name}")
                else:
                    actions.append(f"{bad_dir.name} already absent")
                if self.config.dry_run:
                    actions.append(f"would set {tracker.name} -> {iteration} ({why})")
                else:
                    tracker.write_text(f"{iteration}\n")
                    actions.append(f"set {tracker.name} -> {iteration} ({why})")
                LOGGER.warning("checkpoint recovery in %s: %s", root, "; ".join(actions))
                return ActionResult(
                    status="success",
                    message="; ".join(actions),
                    metadata={
                        "checkpoint_dir": str(root),
                        "quarantined_iteration": bad_iter,
                        "resume_iteration": iteration,
                        "rejected": rejected,
                        "dry_run": self.config.dry_run,
                    },
                )
            rejected.append(f"{candidate.name}: {why}")

        return ActionResult(
            status="failed",
            message=(
                f"no loadable checkpoint left in {root} after quarantining iter_{bad_iter:07d}; "
                f"rejected: {rejected or 'none found'}"
            ),
            metadata={"quarantined_iteration": bad_iter, "rejected": rejected},
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
    # WHAT COUNTS AS PROGRESS, i.e. what resets the streak:
    #
    # any_change: ANY different value. A resume from checkpoint jumps the
    #   counter BACKWARDS and that still counts, so this cannot be tripped by a
    #   healthy restart and is safe with a short window. Answers "is the
    #   training loop emitting iterations at all?" — it catches a frozen counter
    #   and a job that never reaches iteration 1.
    #
    # furthest: only a value HIGHER than any seen before. Redoing iterations the
    #   run has already done is not progress, so this additionally catches a loop
    #   that keeps replaying the same range forever. The window must exceed the
    #   time needed to redo the work lost to the last rolling checkpoint (plus
    #   startup, if the streak spans a restart) or a healthy restart trips it.
    #
    # NB `any_change` was called `increase`, which was actively misleading: it
    #   accepts a DECREASE too, and must, for the resume case above. `furthest`
    #   was called `max`, which named the implementation rather than the
    #   question. Both legacy spellings are still accepted and normalised in
    #   __post_init__ so an older persisted job record still parses — without
    #   that, JobFileStore.load_all() swallows the ValueError and the job
    #   silently disappears from the monitor.
    progress_mode: Literal["any_change", "furthest", "increase", "max"] = "any_change"
    progress_polls: int = 1
    progress_timeout_s: float = 0.0

    def __post_init__(self):
        legacy = {"increase": "any_change", "max": "furthest"}
        if self.progress_mode in legacy:
            self.progress_mode = legacy[self.progress_mode]


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
        ``progress_mode: any_change`` (it is the job's current position, including
        after a backwards jump on resume) and ``max`` drives
        ``progress_mode: furthest``.
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
