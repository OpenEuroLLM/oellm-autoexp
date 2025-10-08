"""Monitoring primitives for oellm_autoexp."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

from compoconf import ConfigInterface, register

from oellm_autoexp.config.schema import MonitorInterface
import oellm_autoexp.utils.run


@dataclass(frozen=True)
class MonitoredJob:
    """Metadata describing an active job to be inspected."""

    job_id: str
    name: str
    log_path: Path
    check_interval_seconds: int
    termination_string: Optional[str] = None
    termination_command: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_paths: List[Path] = field(default_factory=list)


@dataclass
class MonitorOutcome:
    """Snapshot emitted by a monitor iteration."""

    job_id: str
    status: str
    last_update_seconds: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    signals: List["MonitorSignal"] = field(default_factory=list)


@dataclass(frozen=True)
class MonitorSignal:
    """Event detected by the monitor while inspecting a job."""

    job_id: str
    name: str
    action: Optional[str]
    mode: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMonitor(MonitorInterface):
    """Base class monitors can inherit from to gain a common signature."""

    config: ConfigInterface

    def __init__(self, config: ConfigInterface) -> None:
        self.config = config

    async def watch(self, jobs: Iterable[MonitoredJob]) -> Dict[int, MonitorOutcome]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class NullMonitorConfig(ConfigInterface):
    """Monitor configuration that performs no observation."""

    class_name: str = "NullMonitor"
    poll_interval_seconds: int = 600
    inactivity_threshold_seconds: Optional[int] = 900
    termination_string: Optional[str] = None
    termination_command: Optional[str] = None
    log_path_template: str = "{output_dir}/slurm.out"
    output_paths: List[str] = field(default_factory=list)
    start_condition_cmd: Optional[str] = None
    start_condition_interval_seconds: Optional[int] = None


@dataclass
class LogSignalConfig(ConfigInterface):
    """Configuration describing a log-derived signal."""

    name: str
    pattern: str
    action: Optional[str] = None
    mode: Optional[str] = None
    pattern_type: str = "regex"
    metadata: Dict[str, Any] = field(default_factory=dict)
    extract_groups: Dict[str, str | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action and not self.mode:
            raise ValueError(f"LogSignalConfig '{self.name}' must specify action or mode")
        if self.pattern_type not in {"regex", "substring"}:
            raise ValueError(f"Unsupported pattern_type {self.pattern_type!r} for '{self.name}'")


@register
class NullMonitor(BaseMonitor):
    config: NullMonitorConfig

    async def watch(self, jobs: Iterable[MonitoredJob]) -> Dict[int, MonitorOutcome]:
        return {}


@dataclass
class SlurmLogMonitorConfig(NullMonitorConfig):
    """Monitor that inspects SLURM logs for stalls or completion markers."""

    class_name: str = "SlurmLogMonitor"
    inactivity_threshold_seconds: Optional[int] = 900
    check_interval_seconds: int = 300
    log_signals: List[LogSignalConfig] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)


@register
class SlurmLogMonitor(BaseMonitor):
    config: SlurmLogMonitorConfig

    def __init__(self, config: SlurmLogMonitorConfig) -> None:
        super().__init__(config)
        if not config.log_signals:
            config.log_signals = default_log_signals()
        self._snapshots: Dict[int, _JobSnapshot] = {}
        self._compiled_rules: List[_CompiledLogSignal] = [
            _CompiledLogSignal(rule=rule, pattern=_compile_pattern(rule)) for rule in config.log_signals
        ]

    async def watch(self, jobs: Iterable[MonitoredJob]) -> Dict[int, MonitorOutcome]:
        now = time.time()
        observations: Dict[int, MonitorOutcome] = {}
        for job in jobs:
            outcome = self._evaluate_job(job, now)
            observations[job.job_id] = outcome
        return observations

    def _evaluate_job(self, job: MonitoredJob, now: float) -> MonitorOutcome:
        log_path = job.log_path
        status = "pending"
        last_update_seconds: Optional[float] = None
        metadata: Dict[str, Any] = {}
        signals: List[MonitorSignal] = []
        snapshot = self._snapshots.get(job.job_id)
        updated = False

        termination = self._check_termination(job)
        if termination:
            status = "complete"
            metadata.update(termination)
            signals.append(
                MonitorSignal(
                    job_id=job.job_id,
                    name="run_finished",
                    action="run_finished",
                    mode="success",
                    metadata={
                        **termination,
                        "job_name": job.name,
                        "log_path": str(job.log_path),
                    },
                )
            )
            snapshot = _JobSnapshot(log_content="", last_update=now)
            snapshot.output_contents = {}
            self._snapshots[job.job_id] = snapshot
            return MonitorOutcome(
                job_id=job.job_id,
                status=status,
                last_update_seconds=0.0,
                metadata=metadata,
                signals=signals,
            )

        log_previous = snapshot.log_content if snapshot else ""
        log_current = log_previous
        if log_path.exists():
            try:
                log_current = log_path.read_text(errors="ignore")
            except OSError:
                log_current = ""
            if log_current != log_previous:
                signals.extend(
                    self._extract_signals(
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
                content = output_path.read_text(errors="ignore")
            except OSError:
                content = ""
            previous = output_contents.get(output_path, "")
            if content != previous:
                signals.extend(
                    self._extract_signals(
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
            signals=signals,
        )

    def _extract_signals(
        self,
        job: MonitoredJob,
        content: str,
        previous: str,
        *,
        source: str,
        source_path: Optional[Path] = None,
    ) -> List[MonitorSignal]:
        signals: List[MonitorSignal] = []
        if not self._compiled_rules:
            return signals

        new_text = content
        if previous and content.startswith(previous):
            new_text = content[len(previous) :]

        if not new_text:
            return signals

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
                signals.append(
                    MonitorSignal(
                        job_id=job.job_id,
                        name=rule.name,
                        action=rule.action,
                        mode=rule.mode,
                        metadata=metadata,
                    )
                )
        return signals

    def _effective_threshold(self, job: MonitoredJob) -> Optional[int]:
        if job.metadata.get("inactivity_threshold_seconds") is not None:
            return job.metadata["inactivity_threshold_seconds"]
        return self.config.inactivity_threshold_seconds

    def _check_termination(self, job: MonitoredJob) -> Dict[str, Any] | None:
        log_path = job.log_path
        result: Dict[str, Any] = {}
        termination_string = job.termination_string or self.config.termination_string
        if termination_string and log_path.exists():
            try:
                content = log_path.read_text(errors="ignore")
            except OSError:
                content = ""
            if termination_string in content:
                result["reason"] = "termination-string"
                return result

        command = job.termination_command or self.config.termination_command
        if command:
            proc = oellm_autoexp.utils.run.run_with_tee(
                command, shell=True, capture_output=True, check=False, text=True
            )
            try:
                exit_value = int(proc.stdout.strip() or proc.returncode)
            except ValueError:
                exit_value = proc.returncode
            if exit_value == 1:
                result["reason"] = "termination-command"
                return result

        return None


@dataclass
class _JobSnapshot:
    log_content: str
    last_update: float
    output_contents: Dict[Path, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _CompiledLogSignal:
    rule: LogSignalConfig
    pattern: Pattern[str]


def _compile_pattern(rule: LogSignalConfig) -> Pattern[str]:
    if rule.pattern_type == "regex":
        return re.compile(rule.pattern, flags=re.MULTILINE)
    escaped = re.escape(rule.pattern)
    return re.compile(escaped, flags=re.MULTILINE)


def _extract_metadata(match: re.Match[str], rule: LogSignalConfig) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
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


def default_log_signals() -> List[LogSignalConfig]:
    """Return generic monitoring signals covering errors and lifecycle events."""

    return [
        LogSignalConfig(
            name="error",
            pattern=r"(?P<error_type>ERROR|Error|error)[:\s]+(?P<message>.+)",
            pattern_type="regex",
            action="error",
            mode="crash",
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
            action="new_checkpoint",
            metadata={"kind": "checkpoint"},
            extract_groups={
                "checkpoint_path": "checkpoint",
            },
        ),
        LogSignalConfig(
            name="training_complete",
            pattern=r"(Training complete|Training finished)",
            pattern_type="regex",
            action="run_finished",
            mode="success",
            metadata={"kind": "completion"},
            extract_groups={"matched": "match"},
        ),
    ]


__all__ = [
    "BaseMonitor",
    "MonitoredJob",
    "MonitorOutcome",
    "MonitorSignal",
    "NullMonitorConfig",
    "LogSignalConfig",
    "SlurmLogMonitor",
    "SlurmLogMonitorConfig",
    "default_log_signals",
]
