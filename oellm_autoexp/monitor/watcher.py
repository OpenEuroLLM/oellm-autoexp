"""Monitoring primitives for oellm_autoexp."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, MISSING
import re
from typing import Any
from collections.abc import Iterable
from re import Pattern
from pathlib import Path

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import MonitorInterface
from oellm_autoexp.monitor.actions import (
    BaseMonitorAction,
    ErrorNoteActionConfig,
    MonitorActionInterface,
)
from oellm_autoexp.monitor.states import (
    BaseMonitorState,
    CrashStateConfig,
    MonitorStateInterface,
    SuccessState,
    SuccessStateConfig,
)
from oellm_autoexp.utils.run import run_with_tee


@dataclass(frozen=True)
class MonitoredJob:
    """Metadata describing an active job to be inspected."""

    job_id: str = field(default_factory=MISSING)
    name: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    check_interval_seconds: int = field(default_factory=MISSING)
    state: str = field(default_factory=MISSING)
    termination_string: str | None = None
    termination_command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class MonitorOutcome:
    """Snapshot emitted by a monitor iteration."""

    job_id: str = field(default_factory=MISSING)
    status: str = field(default_factory=MISSING)
    last_update_seconds: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[MonitorEvent] = field(default_factory=list)


@dataclass(kw_only=True)
class MonitorEvent:
    """Event detected by the monitor while inspecting a job."""

    job_id: str = field(default_factory=MISSING)
    name: str = field(default_factory=MISSING)
    state: BaseMonitorState | None = None
    actions: list[BaseMonitorAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMonitor(MonitorInterface):
    """Base class monitors can inherit from to gain a common signature."""

    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    async def watch(
        self, jobs: Iterable[MonitoredJob]
    ) -> dict[str, MonitorOutcome]:  # pragma: no cover
        raise NotImplementedError


@dataclass(kw_only=True)
class NullMonitorConfig(ConfigInterface):
    """Monitor configuration that performs no observation."""

    class_name: str = "NullMonitor"
    poll_interval_seconds: int = 600
    inactivity_threshold_seconds: int | None = 900
    termination_string: str | None = None
    termination_command: str | None = None
    log_path_template: str = "{output_dir}/slurm.out"
    output_paths: list[str] = field(default_factory=list)
    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None


@dataclass(kw_only=True)
class LogSignalConfig(ConfigInterface):
    """Configuration describing a log-derived signal."""

    class_name: str = "LogSignal"
    name: str
    pattern: str
    state: MonitorStateInterface.cfgtype | None = None
    actions: list[MonitorActionInterface.cfgtype] = field(default_factory=list)
    pattern_type: str = "regex"
    metadata: dict[str, Any] = field(default_factory=dict)
    extract_groups: dict[str, str | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state is None and not self.actions:
            raise ValueError(
                f"LogSignalConfig '{self.name}' must specify a state change or at least one action"
            )
        if self.pattern_type not in {"regex", "substring"}:
            raise ValueError(f"Unsupported pattern_type {self.pattern_type!r} for '{self.name}'")


@register
class NullMonitor(BaseMonitor):
    config: NullMonitorConfig

    async def watch(self, jobs: Iterable[MonitoredJob]) -> dict[str, MonitorOutcome]:
        return {}


@dataclass(kw_only=True)
class SlurmLogMonitorConfig(NullMonitorConfig):
    """Monitor that inspects SLURM logs for stalls or completion markers."""

    class_name: str = "SlurmLogMonitor"
    inactivity_threshold_seconds: int | None = 900
    check_interval_seconds: int = 300
    log_signals: list[LogSignalConfig] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    state_whitelist: list[str] = field(default_factory=lambda: ["pending", "running", "stall"])


@register
class SlurmLogMonitor(BaseMonitor):
    config: SlurmLogMonitorConfig

    def __init__(self, config: SlurmLogMonitorConfig) -> None:
        super().__init__(config)
        if not config.log_signals:
            config.log_signals = default_log_signals()
        self._snapshots: dict[str, _JobSnapshot] = {}
        self._state_whitelist = {state.lower() for state in config.state_whitelist}
        self._compiled_rules: list[_CompiledLogSignal] = [
            _CompiledLogSignal(
                rule=rule,
                pattern=_compile_pattern(rule),
                actions_config=tuple(rule.actions),
            )
            for rule in config.log_signals
        ]

    async def watch(self, jobs: Iterable[MonitoredJob]) -> dict[str, MonitorOutcome]:
        now = time.time()
        observations: dict[str, MonitorOutcome] = {}
        for job in jobs:
            outcome = self._evaluate_job(job, now)
            observations[job.job_id] = outcome
        return observations

    def _evaluate_job(self, job: MonitoredJob, now: float) -> MonitorOutcome:
        if self._state_whitelist and job.state.lower() not in self._state_whitelist:
            return MonitorOutcome(
                job_id=job.job_id,
                status=job.state,
                last_update_seconds=None,
                metadata={},
                events=[],
            )

        log_path = Path(job.log_path)
        status = "pending"
        last_update_seconds: float | None = None
        metadata: dict[str, Any] = {}
        events: list[MonitorEvent] = []
        snapshot = self._snapshots.get(job.job_id)
        updated = False

        termination_event = self._check_termination(job)
        if termination_event:
            status = "complete"
            metadata.update(termination_event.metadata)
            events.append(termination_event)
            snapshot = _JobSnapshot(log_content="", last_update=now)
            snapshot.output_contents = {}
            self._snapshots[job.job_id] = snapshot
            return MonitorOutcome(
                job_id=job.job_id,
                status=status,
                last_update_seconds=0.0,
                metadata=metadata,
                events=events,
            )

        log_previous = snapshot.log_content if snapshot else ""
        log_current = log_previous
        if log_path.exists():
            try:
                log_current = log_path.read_text(errors="ignore")
            except OSError:
                log_current = ""
            if log_current != log_previous:
                events.extend(
                    self._extract_events(
                        job,
                        log_current,
                        log_previous,
                        source="log",
                        source_path=log_path,
                    )
                )
                updated = True
        else:
            log_current = ""

        output_contents = dict(snapshot.output_contents) if snapshot else {}
        for output_path in job.output_paths:
            try:
                content = Path(output_path).read_text(errors="ignore")
            except OSError:
                content = ""
            previous = output_contents.get(output_path, "")
            if content != previous:
                events.extend(
                    self._extract_events(
                        job,
                        content,
                        previous,
                        source="output",
                        source_path=output_path,
                    )
                )
                output_contents[output_path] = content
                updated = True

        if snapshot is None:
            snapshot = _JobSnapshot(log_content=log_current, last_update=now)
        snapshot.log_content = log_current
        snapshot.output_contents = output_contents
        if updated:
            snapshot.last_update = now
            last_update_seconds = 0.0
            status = "active"
        else:
            if snapshot.log_content or snapshot.output_contents:
                last_update_seconds = now - snapshot.last_update
                threshold = self._effective_threshold(job)
                if threshold is not None and last_update_seconds >= threshold:
                    status = "stall"
                else:
                    status = "active"
            else:
                status = "pending"
                last_update_seconds = None

        self._snapshots[job.job_id] = snapshot

        return MonitorOutcome(
            job_id=job.job_id,
            status=status,
            last_update_seconds=last_update_seconds,
            metadata=metadata,
            events=events,
        )

    def _extract_events(
        self,
        job: MonitoredJob,
        content: str,
        previous: str,
        *,
        source: str,
        source_path: str | None = None,
    ) -> list[MonitorEvent]:
        events: list[MonitorEvent] = []
        if not self._compiled_rules:
            return events

        new_text = content
        if previous and content.startswith(previous):
            new_text = content[len(previous) :]  # noqa

        if not new_text:
            return events

        for compiled in self._compiled_rules:
            rule = compiled.rule
            for match in compiled.pattern.finditer(new_text):
                metadata = dict(rule.metadata)
                extracted = _extract_metadata(match, rule)
                metadata.update(extracted)
                metadata.setdefault("job_name", job.name)
                if source_path is not None:
                    metadata.setdefault("source_path", str(source_path))
                metadata.setdefault("source", source)
                state_instance = compiled.instantiate_state()
                events.append(
                    MonitorEvent(
                        job_id=job.job_id,
                        name=rule.name,
                        state=state_instance,
                        actions=[
                            action_cfg.instantiate(MonitorActionInterface)
                            for action_cfg in compiled.actions_config
                        ],
                        metadata=metadata,
                    )
                )
        return events

    def _effective_threshold(self, job: MonitoredJob) -> int | None:
        if job.metadata.get("inactivity_threshold_seconds") is not None:
            return job.metadata["inactivity_threshold_seconds"]
        return self.config.inactivity_threshold_seconds

    def _check_termination(self, job: MonitoredJob) -> MonitorEvent | None:
        log_path = Path(job.log_path)
        metadata: dict[str, Any] = {}
        termination_string = job.termination_string or self.config.termination_string
        if termination_string and log_path.exists():
            try:
                content = log_path.read_text(errors="ignore")
            except OSError:
                content = ""
            if termination_string in content:
                metadata.update(
                    {
                        "reason": "termination-string",
                        "log_path": str(log_path),
                    }
                )
                metadata.setdefault("job_name", job.name)
                return MonitorEvent(
                    job_id=job.job_id,
                    name="termination",
                    state=SuccessState(SuccessStateConfig()),
                    metadata=metadata,
                )

        command = job.termination_command or self.config.termination_command
        if command:
            proc = run_with_tee(command, shell=True, capture_output=True, check=False, text=True)
            try:
                exit_value = int(proc.stdout.strip() or proc.returncode)
            except ValueError:
                exit_value = proc.returncode
            if exit_value == 1:
                metadata.update(
                    {
                        "reason": "termination-command",
                        "command": command,
                    }
                )
                metadata.setdefault("job_name", job.name)
                return MonitorEvent(
                    job_id=job.job_id,
                    name="termination",
                    state=SuccessState(SuccessStateConfig()),
                    metadata=metadata,
                )

        return None


@dataclass(kw_only=True)
class _JobSnapshot:
    log_content: str
    last_update: float
    output_contents: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _CompiledLogSignal:
    rule: LogSignalConfig
    pattern: Pattern[str]
    actions_config: tuple[MonitorActionInterface.cfgtype, ...] = tuple()

    def instantiate_state(self) -> BaseMonitorState | None:
        if self.rule.state is None:
            return None
        return self.rule.state.instantiate(MonitorStateInterface)


def _compile_pattern(rule: LogSignalConfig) -> Pattern[str]:
    if rule.pattern_type == "regex":
        return re.compile(rule.pattern, flags=re.MULTILINE)
    escaped = re.escape(rule.pattern)
    return re.compile(escaped, flags=re.MULTILINE)


def _extract_metadata(match: re.Match[str], rule: LogSignalConfig) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    if not rule.extract_groups:
        return extracted
    for key, group in rule.extract_groups.items():
        if isinstance(group, str) and group == "match":
            extracted[key] = match.group(0)
            continue
        try:
            extracted[key] = match.group(group)
        except (IndexError, KeyError):
            continue
    return extracted


def default_log_signals() -> list[LogSignalConfig]:
    """Return generic monitoring signals covering errors and lifecycle
    events."""

    return [
        LogSignalConfig(
            name="error",
            pattern=r"(?P<error_type>ERROR|Error|error)[:\s]+(?P<message>.+)",
            pattern_type="regex",
            state=CrashStateConfig(),
            actions=[ErrorNoteActionConfig(note="log_error")],
            metadata={"severity": "error"},
            extract_groups={
                "error_message": "message",
                "error_kind": "error_type",
            },
        ),
        LogSignalConfig(
            name="checkpoint",
            pattern=r"Checkpoint (?:saved|written)[:\s]+(?P<checkpoint>\S+)",
            pattern_type="regex",
            actions=[ErrorNoteActionConfig(note="new_checkpoint")],
            metadata={"kind": "checkpoint"},
            extract_groups={
                "checkpoint_path": "checkpoint",
            },
        ),
        LogSignalConfig(
            name="training_complete",
            pattern=r"(Training complete|Training finished)",
            pattern_type="regex",
            state=SuccessStateConfig(),
            metadata={"kind": "completion"},
            extract_groups={"matched": "match"},
        ),
    ]


__all__ = [
    "BaseMonitor",
    "MonitoredJob",
    "MonitorOutcome",
    "MonitorEvent",
    "NullMonitorConfig",
    "LogSignalConfig",
    "SlurmLogMonitor",
    "SlurmLogMonitorConfig",
    "default_log_signals",
]
