"""Simplified synchronous monitor loop."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from compoconf import ConfigInterface, parse_config, asdict

from oellm_autoexp.monitor.conditions import (
    ConditionContext,
    ConditionResult,
    MonitorConditionInterface,
)
from oellm_autoexp.monitor.actions import (
    EventRecord,
    build_event_id,
    LogEventConfig,
    LogEvent,
    StateEventConfig,
    StateEvent,
    ActionResult,
    BaseMonitorAction,
    NewJobActionConfig,
)
from oellm_autoexp.monitor.job_client_protocol import JobClientProtocol
from oellm_autoexp.monitor.submission import JobInterface, SlurmJobConfig, LocalJobConfig
from oellm_autoexp.monitor.utils.paths import resolve_log_path

LOGGER = logging.getLogger(__name__)


SCHEMA_VERSION = 1


@dataclass
class JobRuntimeConfig:
    class_name: str = "JobRuntime"
    submitted: bool = False
    attempts: int = 0
    runtime_job_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    log_cursor: int = 0
    condition_state: dict[str, Any] = field(default_factory=dict)
    action_state: dict[str, Any] = field(default_factory=dict)
    last_status: str | None = None
    final_state: str | None = None  # "finished", "cancelled", or None for active jobs


@dataclass(kw_only=True)
class JobRecordConfig:
    class_name: str = "JobRecord"
    job_id: str = ""
    definition: JobInterface.cfgtype = field(default_factory=MISSING)
    runtime: JobRuntimeConfig = field(default_factory=JobRuntimeConfig)
    schema_version: int = SCHEMA_VERSION
    array_idx: int | None = None


class JobFileStore:
    """Store job records as files in a state directory."""

    def __init__(self, state_dir: str | Path) -> None:
        self.root = Path(state_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def list_paths(self) -> Iterable[Path]:
        return self.root.glob("*.job.json")

    def load_all(self, *, include_finished: bool = False) -> list[JobRecordConfig]:
        """Load job records, optionally excluding finished/cancelled jobs."""
        jobs: list[JobRecordConfig] = []
        for path in self.list_paths():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                _import_registry()
                job = parse_config(JobRecordConfig, payload)
                # Parse nested event configs once
                _normalize_job_definition(job)
                # Skip jobs that are finished/cancelled unless requested
                if not include_finished and job.runtime.final_state is not None:
                    continue
                jobs.append(job)
            except (OSError, json.JSONDecodeError, ValueError, KeyError):
                continue
        return jobs

    def upsert(self, record: JobRecordConfig) -> None:
        path = self.path_for(record.job_id)
        payload = asdict(record)
        payload.setdefault("schema_version", SCHEMA_VERSION)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def mark_finished(self, job_id: str, final_state: str) -> None:
        """Mark a job as finished or cancelled without deleting it."""
        job = self.load(job_id)
        if job is None:
            return
        job.runtime.final_state = final_state
        job.runtime.end_ts = time.time()
        self.upsert(job)

    def remove(self, job_id: str) -> None:
        """Actually delete a job file (use with caution - prefer mark_finished)."""
        path = self.path_for(job_id)
        if path.exists():
            path.unlink()

    def load(self, job_id: str, include_finished=False) -> JobRecordConfig | None:
        path = self.path_for(job_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _import_registry()
            job = parse_config(JobRecordConfig, payload)
            # Parse nested event configs once
            _normalize_job_definition(job)
            if not include_finished and job.runtime.final_state is not None:
                return None
            return job
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            return None

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.job.json"


class MonitorLoop:
    """Synchronous monitor loop that evaluates jobs and actions inline."""

    def __init__(
        self,
        store: JobFileStore,
        *,
        slurm_client: JobClientProtocol | None = None,
        local_client: JobClientProtocol | None = None,
        poll_interval_seconds: float = 60.0,
        show_poll_state: bool = True,
    ) -> None:
        _import_registry()
        self._store = store
        self._slurm_client = slurm_client
        self._local_client = local_client
        self.poll_interval_seconds = poll_interval_seconds
        self.show_poll_state = show_poll_state

    def _get_client(self, job: JobRecordConfig) -> JobClientProtocol:
        """Get the appropriate client based on job configuration."""
        if job.definition is None:
            raise ValueError("Job definition is None")

        if isinstance(job.definition, SlurmJobConfig):
            return self._slurm_client
        elif isinstance(job.definition, LocalJobConfig):
            return self._local_client

    def observe_once(self) -> None:
        # Query both clients and merge statuses
        statuses: dict[str, str] = {}
        if self._slurm_client:
            statuses.update(self._slurm_client.squeue())
        if self._local_client:
            statuses.update(self._local_client.squeue())

        for job in self._store.load_all():
            if job.definition is None:
                continue
            runtime = job.runtime
            runtime_id = runtime.runtime_job_id
            new_status = statuses.get(runtime_id) if runtime_id else None
            self._status_action(job, runtime.last_status, new_status)
            runtime.last_status = new_status
            if not runtime.submitted:
                if self._check_cancel(job):
                    self._store.mark_finished(job.job_id, "cancelled")
                    continue
                if self._check_finish(job):
                    self._store.mark_finished(job.job_id, "finished")
                    continue
                if self._check_start(job):
                    self._start_jobs(job)

                    if job.runtime.runtime_job_id is None:
                        LOGGER.warning(f"Failed to start job: {job}")
                        self._store.mark_finished(job.job_id, "cancelled")
                else:
                    self._store.upsert(job)
                continue

            if self._check_cancel(job):
                if runtime_id:
                    client = self._get_client(job)
                    client.cancel(runtime_id)
                    client.remove(runtime_id)
                self._store.mark_finished(job.job_id, "cancelled")
                continue
            if self._check_finish(job):
                if runtime_id:
                    client = self._get_client(job)
                    client.remove(runtime_id)
                self._store.mark_finished(job.job_id, "finished")
                continue

            # Process log events
            effect = self._process_log_events(job)
            if effect == "finished":
                if runtime_id:
                    client = self._get_client(job)
                    client.remove(runtime_id)
                self._store.mark_finished(job.job_id, "finished")
                continue
            if effect == "cancelled":
                if runtime_id:
                    client = self._get_client(job)
                    client.cancel(runtime_id)
                    client.remove(runtime_id)
                self._store.mark_finished(job.job_id, "cancelled")
                continue
            if effect == "restart":
                self._restart_job(job)
                self._store.upsert(job)
                continue

            # Check if job completed naturally
            if runtime.last_status in {"COMPLETED", "FAILED", "CANCELLED"}:
                if runtime_id:
                    client = self._get_client(job)
                    client.remove(runtime_id)
                self._store.mark_finished(job.job_id, "finished")
                continue

            self._store.upsert(job)

        if self.show_poll_state:
            statuses: dict[str, str] = {}
            if self._slurm_client:
                statuses.update(self._slurm_client.squeue())
            if self._local_client:
                statuses.update(self._local_client.squeue())

            poll_state = {}
            for job in self._store.load_all():
                if self._check_finish(job):
                    continue
                runtime = job.runtime
                if not runtime.submitted:
                    poll_state[job.runtime.runtime_job_id] = {
                        "state": "pending",
                        "start_condition": job.definition.start_condition,
                        "runtime": job.runtime,
                    }
                else:
                    poll_state[job.runtime.runtime_job_id] = {
                        "state": statuses.get(runtime_id),
                        "cancel_condition": job.definition.cancel_condition,
                        "runtime": job.runtime,
                    }

            LOGGER.info(f"[{time.time():0.6f}]" + f"Monitor Polling: {poll_state}")

    def _check_start(self, job: JobRecordConfig) -> bool:
        condition = job.definition.start_condition
        if condition is None:
            return True
        return self._evaluate_condition(job, condition, label="start").passed

    def _check_cancel(self, job: JobRecordConfig) -> bool:
        condition = job.definition.cancel_condition
        if condition is None:
            return False
        return self._evaluate_condition(job, condition, label="cancel").passed

    def _check_finish(self, job: JobRecordConfig) -> bool:
        condition = job.definition.finish_condition
        if condition is None:
            return False
        return self._evaluate_condition(job, condition, label="finish").passed

    def _start_job(self, job: JobRecordConfig) -> None:
        runtime = job.runtime
        runtime.attempts += 1
        runtime.start_ts = time.time()
        client = self._get_client(job)
        if job.array_idx is not None:
            job_ids = client.submit_array(job.definition, indices=[job.array_idx])
            if not isinstance(job_ids, list):
                runtime_job_id = job_ids
            elif not job_ids:
                raise ValueError("submit_array returned no job ids")
            else:
                runtime_job_id = job_ids[0]
        else:
            try:
                runtime_job_id = client.submit(job.definition)
            except Exception as e:
                LOGGER.error(f"Unable to submit {job.definition.name}: {e}")
                runtime_job_id = None
        if runtime_job_id is not None:
            runtime.runtime_job_id = runtime_job_id
            runtime.submitted = True
            runtime.log_cursor = 0

    def _start_jobs(self, job: JobRecordConfig, indices: list[int] | None = None) -> None:
        definition = job.definition
        if definition is None:
            return
        array_len = int(getattr(definition, "array_len", 1) or 1)
        if array_len <= 1:
            self._start_job(job)
            self._store.upsert(job)
            return

        runtime = job.runtime
        runtime.attempts += 1
        runtime.start_ts = time.time()
        client = self._get_client(job)
        indices = indices or list(range(array_len))
        try:
            job_ids = client.submit_array(definition, indices)
        except Exception as e:
            LOGGER.error(f"Unable to submit {job.definition.name}: {e}")
            job_ids = []

        for idx, runtime_job_id in zip(indices, job_ids):
            task_runtime = JobRuntimeConfig(
                submitted=True,
                attempts=runtime.attempts,
                runtime_job_id=runtime_job_id,
                start_ts=runtime.start_ts,
                log_cursor=0,
            )
            task_record = JobRecordConfig(
                job_id=f"{job.job_id}_{idx}",
                definition=definition,
                runtime=task_runtime,
                array_idx=idx,
            )
            self._store.upsert(task_record)
        if job_ids:
            self._store.remove(job.job_id)

    def _process_log_events(self, job: JobRecordConfig) -> str:
        """Process log events by checking patterns in new log content.

        Returns: "continue", "finished", "cancelled", or "restart"
        """
        runtime = job.runtime
        definition = job.definition
        log_path = self._resolve_log_path(job)
        if not log_path.exists():
            return "continue"

        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(runtime.log_cursor)
                new_text = handle.read()
                runtime.log_cursor = handle.tell()
        except OSError:
            return "continue"

        if not new_text:
            return "continue"

        # Process each log event configuration (already parsed)
        log_events = getattr(definition, "log_events", None) or []
        for idx, event_cfg in enumerate(log_events):
            if event_cfg.action is None:
                continue

            # Create LogEvent instance
            log_event = LogEvent(event_cfg)

            # Check if event triggers
            triggers = log_event.check_triggers(new_text)

            for metadata in triggers:
                event_id = build_event_id(job.job_id, event_cfg.name, metadata)
                action_id = f"log:{event_cfg.name}:{idx}"
                action_state = runtime.action_state.get(action_id, {})

                event = EventRecord(
                    event_id=event_id,
                    name=event_cfg.name,
                    source="log",
                    payload=metadata,
                    metadata={
                        "job_id": job.job_id,
                        "job_name": definition.name,
                        "last_action_ts": float(action_state.get("last_action_ts", 0.0)),
                    },
                )

                # Instantiate and execute action
                action = event_cfg.action.instantiate(BaseMonitorAction)
                if self._evaluate_event_condition(job, event, event_cfg.condition, action_id):
                    result = action.execute(self._action_context(event, job))
                    self._update_action_state(runtime, action_id, result)
                    effect = self._handle_action_result(job, result)
                    if effect in ("finished", "cancelled", "restart"):
                        return effect

        return "continue"

    def _status_action(self, job: JobRecordConfig, old_status: str | None, new_status: str | None):
        """Process state transition events."""
        runtime = job.runtime
        definition = job.definition

        # Skip if no state change
        if old_status == new_status:
            return

        # Process each state event configuration (already parsed)
        state_events = getattr(definition, "state_events", None) or []
        for idx, event_cfg in enumerate(state_events):
            if event_cfg.action is None:
                continue

            # Create StateEvent instance
            state_event = StateEvent(event_cfg)

            # Check if event triggers
            if not state_event.check_trigger(old_status, new_status):
                continue

            # Build event metadata
            metadata = state_event.build_metadata(old_status, new_status)
            event_id = build_event_id(job.job_id, event_cfg.name, metadata)
            action_id = f"state:{event_cfg.name}:{idx}"
            action_state = runtime.action_state.get(action_id, {})

            event = EventRecord(
                event_id=event_id,
                name=event_cfg.name,
                source="state",
                payload=metadata,
                metadata={
                    "job_id": job.job_id,
                    "job_name": definition.name,
                    "last_action_ts": float(action_state.get("last_action_ts", 0.0)),
                },
            )

            # Instantiate and execute action
            action = event_cfg.action.instantiate(BaseMonitorAction)
            if self._evaluate_event_condition(job, event, event_cfg.condition, action_id):
                result = action.execute(self._action_context(event, job))
                self._update_action_state(runtime, action_id, result)
                # Note: we don't handle remove/restart here as state changes are informational
                # and shouldn't directly control job lifecycle

    def _action_context(self, event: EventRecord, job: JobRecordConfig):
        from oellm_autoexp.monitor.actions import ActionContext

        return ActionContext(
            event=event,
            job_metadata=self._build_job_metadata(job),
            attempts=job.runtime.attempts,
        )

    def _evaluate_event_condition(
        self,
        job: JobRecordConfig,
        event: EventRecord,
        condition_cfg: MonitorConditionInterface.cfgtype | None,
        action_id: str,
    ) -> bool:
        """Evaluate a single event condition."""
        if condition_cfg is None:
            return True

        action_state = job.runtime.action_state.setdefault(action_id, {})
        condition_state = action_state.setdefault("condition", {})

        if "started_ts" not in condition_state:
            condition_state["started_ts"] = time.time()

        condition = condition_cfg.instantiate(MonitorConditionInterface)
        ctx = ConditionContext(
            event=event,
            job_metadata=self._build_job_metadata(job),
            attempts=job.runtime.attempts,
            state=condition_state,
            started_ts=condition_state.get("started_ts"),
        )
        result = condition.check(ctx)
        result = _apply_persistence(condition_cfg, condition_state, result)
        return result.passed

    def _evaluate_condition(
        self,
        job: JobRecordConfig,
        condition_cfg: MonitorConditionInterface.cfgtype,
        *,
        label: str,
    ) -> ConditionResult:
        state = job.runtime.condition_state.setdefault(label, {})
        if "started_ts" not in state:
            state["started_ts"] = time.time()
        condition = condition_cfg.instantiate(MonitorConditionInterface)
        ctx = ConditionContext(
            job_metadata=self._build_job_metadata(job),
            attempts=job.runtime.attempts,
            state=state,
            started_ts=state.get("started_ts"),
        )
        result = condition.check(ctx)
        return _apply_persistence(condition_cfg, state, result)

    def _build_job_metadata(self, job: JobRecordConfig) -> dict[str, Any]:
        definition = job.definition
        metadata = dict(definition.metadata)
        metadata.setdefault("job_id", job.job_id)
        metadata.setdefault("job_name", definition.name)
        job_class = definition.class_name
        metadata.setdefault("job_class", job_class)
        return metadata

    def _resolve_log_path(self, job: JobRecordConfig) -> Path:
        definition = job.definition
        job_id = job.runtime.runtime_job_id or job.job_id
        runtime = job.runtime
        if "_" in job_id:
            array_index = int(job_id.split("_")[-1])
        else:
            array_index = 0
        if definition.log_path_current:
            log_path_cur = definition.log_path_current.replace("%a", str(array_index))
            return Path(log_path_cur)
        timestamp = int(runtime.start_ts or time.time())
        return resolve_log_path(
            definition.log_path,
            job_id=runtime.runtime_job_id or job.job_id,
            timestamp=timestamp,
        )

    def _update_action_state(self, runtime: JobRuntimeConfig, action_id: str, result) -> None:
        state = runtime.action_state.setdefault(action_id, {})
        state["last_action_ts"] = time.time()
        state["last_status"] = result.status

    def _handle_action_result(self, job: JobRecordConfig, result: ActionResult) -> str:
        """Handle the result of an action execution.

        Returns: "continue", "finished", "cancelled", or "restart"
        """
        # Handle special actions
        if result.special == "restart":
            return "restart"

        if result.special == "cancel":
            return "cancelled"

        if result.special == "finish":
            return "finished"

        # Handle new job submissions using typed config
        if result.action_config is not None:
            if isinstance(result.action_config, NewJobActionConfig) and isinstance(
                result.action_config.job_config, LocalJobConfig
            ):
                self._submit_local_job(result.action_config.job_config)
            elif isinstance(result.action_config, NewJobActionConfig) and isinstance(
                result.action_config.job_config, SlurmJobConfig
            ):
                self._submit_slurm_job(result.action_config.job_config)

        return "continue"

    def _restart_job(self, job: JobRecordConfig) -> None:
        """Restart job preserving condition_state, action_state, and
        attempts."""
        runtime = job.runtime
        client = self._get_client(job)

        # Cancel and remove existing job if it exists
        if runtime.runtime_job_id:
            client.cancel(runtime.runtime_job_id)
            client.remove(runtime.runtime_job_id)

        # Reset runtime fields but preserve state and attempts
        runtime.submitted = False
        runtime.runtime_job_id = None
        runtime.log_cursor = 0
        runtime.start_ts = None
        # Note: condition_state, action_state, and attempts are preserved

        # Restart the job
        self._start_job(job)
        if job.runtime.runtime_job_id is None:
            LOGGER.warning(f"Failed to re-start job: {job}")
            self._store.mark_finished(job.job_id, "cancelled")

    def _submit_local_job(self, job_config: ConfigInterface) -> None:
        """Submit a new local job (fire-and-forget)."""
        if not self._local_client:
            LOGGER.error("Cannot submit local job: no local client available")
            return

        try:
            job_instance = job_config.instantiate(JobInterface)
            job_id = self._local_client.submit(job_instance)
            LOGGER.info(f"Submitted local job {job_id}")
        except Exception as e:
            LOGGER.error(f"Failed to submit local job: {e}")

    def _submit_slurm_job(self, job_config: ConfigInterface) -> None:
        """Submit a new Slurm job (fire-and-forget)."""
        if not self._slurm_client:
            LOGGER.error("Cannot submit Slurm job: no Slurm client available")
            return

        try:
            job_instance = job_config.instantiate(JobInterface)
            job_id = self._slurm_client.submit(job_instance)
            LOGGER.info(f"Submitted Slurm job {job_id}")
        except Exception as e:
            LOGGER.error(f"Failed to submit Slurm job: {e}")


def _normalize_job_definition(job: JobRecordConfig) -> None:
    """Parse and normalize all nested event configs in the job definition."""
    if job.definition is None:
        return

    # Parse log events
    if hasattr(job.definition, "log_events") and job.definition.log_events:
        parsed_log_events = []
        for item in job.definition.log_events:
            if isinstance(item, LogEventConfig):
                parsed_log_events.append(item)
            elif isinstance(item, dict):
                parsed_log_events.append(parse_config(LogEventConfig, item))
        job.definition.log_events = parsed_log_events

    # Parse state events
    if hasattr(job.definition, "state_events") and job.definition.state_events:
        parsed_state_events = []
        for item in job.definition.state_events:
            if isinstance(item, StateEventConfig):
                parsed_state_events.append(item)
            elif isinstance(item, dict):
                parsed_state_events.append(parse_config(StateEventConfig, item))
        job.definition.state_events = parsed_state_events


def _apply_persistence(
    condition_cfg: MonitorConditionInterface.cfgtype,
    condition_state: dict[str, Any],
    result: ConditionResult,
) -> ConditionResult:
    if condition_state.get("latched_pass"):
        return ConditionResult(passed=True, message=result.message, metadata=result.metadata)
    if condition_state.get("latched_fail"):
        return ConditionResult(passed=False, message=result.message, metadata=result.metadata)
    persistent_pass = bool(getattr(condition_cfg, "persistent_pass", False))
    persistent_fail = bool(getattr(condition_cfg, "persistent_fail", False))
    if result.passed and persistent_pass:
        condition_state["latched_pass"] = True
    if (not result.passed) and persistent_fail:
        condition_state["latched_fail"] = True
    return result


def _import_registry() -> None:
    import oellm_autoexp.monitor.actions  # noqa: F401
    import oellm_autoexp.monitor.conditions  # noqa: F401
    import oellm_autoexp.monitor.submission  # noqa: F401
    import oellm_autoexp.config.conditions  # noqa: F401
