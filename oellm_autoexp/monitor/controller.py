"""Monitoring controller bridging SLURM state and event actions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, MISSING, asdict
import logging
from pathlib import Path
import time
from typing import Any, Literal
from collections.abc import Iterable

from oellm_autoexp.monitor.action_queue import ActionQueue
from oellm_autoexp.monitor.actions import ActionContext, ActionResult
from oellm_autoexp.monitor.event_bindings import EventActionBinding, instantiate_bindings
from oellm_autoexp.monitor.conditions import ConditionContext
from oellm_autoexp.monitor.events import EventRecord, EventStatus
from oellm_autoexp.monitor.states import (
    BaseMonitorState,
    CrashState,
    CrashStateConfig,
    MonitorStateInterface,
    StalledState,
    StalledStateConfig,
    TimeoutState,
    TimeoutStateConfig,
    PendingState,
    PendingStateConfig,
    UndefinedState,
    UndefinedStateConfig,
    StartedState,
    StartedStateConfig,
    SuccessState,
    SuccessStateConfig,
)
from oellm_autoexp.monitor.watcher import (
    BaseMonitor,
    MonitoredJob,
    MonitorEvent,
    MonitorOutcome,
    default_state_events,
)
from oellm_autoexp.persistence import MonitorStateStore, StoredJob
from oellm_autoexp.slurm.client import BaseSlurmClient
from oellm_autoexp.utils.start_condition import (
    resolve_start_condition_interval,
    wait_for_start_condition,
)


LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class JobRegistration:
    """Information required to resubmit and observe a job."""

    name: str = field(default_factory=MISSING)
    script_path: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    metadata: dict[str, Any] = field(default_factory=dict)
    termination_string: str | None = None
    termination_command: str | None = None
    inactivity_threshold_seconds: int | None = None
    output_paths: list[str] = field(default_factory=list)
    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None


@dataclass(kw_only=True)
class JobRuntimeState:
    job_id: str = field(default_factory=MISSING)
    registration: JobRegistration = field(default_factory=MISSING)
    attempts: int = 1
    last_outcome: MonitorOutcome | None = None
    last_slurm_state: str | None = None
    state: BaseMonitorState = field(default_factory=lambda: UndefinedState(UndefinedStateConfig()))

    @property
    def name(self) -> str:
        return self.registration.name


@dataclass(kw_only=True)
class MonitorRecord:
    """Recorded monitor event and optional action payload."""

    job_id: str = field(default_factory=MISSING)
    job_name: str = field(default_factory=MISSING)
    event: str = field(default_factory=MISSING)
    state: str | None = None
    action: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


ActionLiteral = Literal["restart", "stop", "success", "noop"]


@dataclass(kw_only=True)
class MonitorDecision:
    """Controller-level decision recorded for a job."""

    action: ActionLiteral = field(default_factory=MISSING)
    reason: str = field(default_factory=MISSING)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class MonitorCycleResult:
    """Aggregated outcome of a single monitoring iteration."""

    decisions: dict[str, MonitorDecision] = field(default_factory=dict)
    events: list[MonitorRecord] = field(default_factory=list)


class MonitorController:
    """Coordinate monitors, SLURM state, and event-driven actions."""

    def __init__(
        self,
        monitor: BaseMonitor,
        slurm: BaseSlurmClient,
        state_store: MonitorStateStore | None = None,
    ) -> None:
        self._monitor = monitor
        self._slurm = slurm
        self._jobs: dict[str, JobRuntimeState] = {}  # Job IDs are strings
        self._pending_records: list[MonitorRecord] = []
        self._state_store = state_store
        self._event_records: dict[str, EventRecord] = {}
        self._event_index: dict[tuple[str, str], str] = {}
        self._action_queue: ActionQueue | None = None
        if self._state_store is not None:
            self._event_records = self._state_store.load_events()
            for record in self._event_records.values():
                job_id = record.metadata.get("job_id")
                if job_id:
                    self._event_index[(str(job_id), record.name)] = record.event_id
            queue_path = self._state_store.session_path.with_suffix(".actions")
            self._action_queue = ActionQueue(queue_path)
        state_event_cfgs = getattr(self._monitor.config, "state_events", None)
        if not state_event_cfgs:
            state_event_cfgs = default_state_events()
        self._state_event_configs = {cfg.name: cfg for cfg in state_event_cfgs}

    def register_job(
        self,
        job_id: str,
        registration: JobRegistration,
        attempts: int = 1,
        state: BaseMonitorState = UndefinedState(UndefinedStateConfig()),
    ) -> None:
        job_key = str(job_id)
        state = JobRuntimeState(
            job_id=job_key,
            registration=registration,
            attempts=max(1, attempts),
            state=state,
        )
        self._jobs[job_key] = state
        LOGGER.info(
            f"[job {job_key}] registered for monitoring: name={registration.name}, "
            f"log_path={registration.log_path}, attempts={attempts}"
        )
        self._persist_job(state)

    def jobs(self) -> Iterable[JobRuntimeState]:
        return list(self._jobs.values())

    def observe_once_sync(self) -> MonitorCycleResult:
        interval = getattr(
            self._monitor.config,
            "check_interval_seconds",
            getattr(self._monitor.config, "poll_interval_seconds", 60),
        )
        monitored_jobs = [
            MonitoredJob(
                job_id=str(state.job_id),
                name=state.name,
                log_path=self._expand_log_path(state.job_id, state.registration.log_path),
                check_interval_seconds=interval,
                state=state.state.key,
                termination_string=state.registration.termination_string,
                termination_command=state.registration.termination_command,
                metadata=self._build_job_metadata(state),
                output_paths=[
                    str(self._expand_log_path(state.job_id, path))
                    for path in state.registration.output_paths
                ],
            )
            for state in self._jobs.values()
        ]
        outcomes = self._monitor.watch_sync(monitored_jobs)
        slurm_snapshot = self._slurm.squeue()

        cycle_result = MonitorCycleResult()
        for state in list(self._jobs.values()):
            outcome = outcomes.get(state.job_id)
            state.last_outcome = outcome
            slurm_state = slurm_snapshot.get(state.job_id)

            # Log monitor outcome for debugging
            if outcome:
                LOGGER.debug(
                    f"[job {state.job_id}] monitor outcome: status={outcome.status}, "
                    f"last_update={outcome.last_update_seconds}s, events={len(outcome.events)}"
                )

            # Log SLURM state for debugging
            LOGGER.debug(
                f"[job {state.job_id}] SLURM state: {slurm_state or 'NOT_FOUND'} "
                f"(previous: {state.last_slurm_state or 'NONE'})"
            )

            transition_records = self._capture_slurm_transitions(state, slurm_state)
            if transition_records:
                cycle_result.events.extend(transition_records)
            handled = False
            if outcome is not None:
                for event in outcome.events:
                    self._handle_monitor_event(state, event, cycle_result)
                    handled = handled or bool(event.state)
            if handled:
                continue
            classification = self._classify_mode(state, outcome, slurm_snapshot)
            if classification is None:
                continue
            mode, mode_metadata = classification
            LOGGER.info(
                f"[job {state.job_id}] classified as mode '{mode}' "
                f"(monitor_status={outcome.status if outcome else 'NONE'}, slurm_state={slurm_state})"
            )
            combined_metadata = dict(outcome.metadata if outcome else {})
            combined_metadata.update(mode_metadata)
            synthetic_event = self._build_state_event(
                state,
                mode,
                combined_metadata,
                default_state=_fallback_state_for(mode),
            )
            self._handle_monitor_event(state, synthetic_event, cycle_result)
        return cycle_result

    async def observe_once(self) -> MonitorCycleResult:
        return self.observe_once_sync()

    def handle_state_change(self, job_id: str, mode: str) -> MonitorDecision:
        state = self._jobs[job_id]
        event = self._build_state_event(
            state,
            mode,
            metadata={},
            default_state=_fallback_state_for(mode),
        )
        cycle = MonitorCycleResult()
        result = self._handle_monitor_event(state, event, cycle)
        if result:
            decision, _ = result
            return decision
        return MonitorDecision(action="noop", reason=f"no action for mode '{mode}'")

    def snapshot(self) -> dict[str, str]:
        return self._slurm.squeue()

    def drain_events(self) -> list[MonitorRecord]:
        records = self._pending_records
        self._pending_records = []
        return records

    def clear_state(self) -> None:
        if self._state_store:
            self._state_store.clear()

    def _persist_job(self, state: JobRuntimeState) -> None:
        if not self._state_store:
            return
        resolved_log_path = self._expand_log_path(state.job_id, state.registration.log_path)
        monitor_state = state.state.key if state.state else None
        stored = StoredJob.from_registration(
            state.job_id,
            state.attempts,
            state.registration,
            resolved_log_path=str(resolved_log_path),
            monitor_state=monitor_state,
            slurm_state=state.last_slurm_state,
        )
        self._state_store.upsert_job(stored)

    def _set_state(self, state: JobRuntimeState, key: str) -> None:
        normalized = (key or "").lower()
        if normalized in {"success", "completed"}:
            state.state = SuccessState(SuccessStateConfig())
        elif normalized in {"running", "started", "active"}:
            state.state = StartedState(StartedStateConfig())
        elif normalized in {"stall", "stalled"}:
            state.state = StalledState(StalledStateConfig())
        elif normalized in {"timeout"}:
            state.state = TimeoutState(TimeoutStateConfig())
        elif normalized in {"crash", "failed", "error", "cancelled"}:
            state.state = CrashState(CrashStateConfig())
        elif normalized in {"pending"}:
            state.state = PendingState(PendingStateConfig())

    def _finalize_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._slurm.remove(job_id)
        if self._state_store:
            self._state_store.remove_job(job_id)

    def _restart_job(
        self,
        state: JobRuntimeState,
        adjustments: dict[str, Any] | None = None,
    ) -> str:
        old_job_id = state.job_id
        if adjustments:
            script_override = adjustments.get("script_path")
            if script_override:
                LOGGER.info(
                    f"[job {old_job_id}] overriding script path: {state.registration.script_path} -> {script_override}"
                )
                state.registration.script_path = str(script_override)
            log_override = adjustments.get("log_path")
            if log_override:
                LOGGER.info(
                    f"[job {old_job_id}] overriding log path: {state.registration.log_path} -> {log_override}"
                )
                state.registration.log_path = str(log_override)
            metadata_patch = adjustments.get("metadata")
            if isinstance(metadata_patch, dict):
                state.registration.metadata.update(metadata_patch)

        if state.registration.start_condition_cmd:
            interval = resolve_start_condition_interval(
                state.registration.start_condition_interval_seconds,
                self._monitor.config,
            )
            wait_for_start_condition(
                state.registration.start_condition_cmd,
                interval_seconds=interval,
                logger=LOGGER,
            )

        self._slurm.cancel(old_job_id)
        self._slurm.remove(old_job_id)

        # Check if this is an array job task (job_id contains underscore)
        # Array jobs need to be resubmitted as single-element arrays to preserve $SLURM_ARRAY_TASK_ID
        if "_" in str(old_job_id) and hasattr(self._slurm, "submit_array"):
            # Extract the array task index from job_id (e.g., "12345_2" -> index 2)
            parts = str(old_job_id).split("_")
            if len(parts) == 2 and parts[1].isdigit():
                array_idx = int(parts[1])
                LOGGER.info(
                    f"[job {old_job_id}] restarting as single-element array job (task index {array_idx})"
                )
                # Submit as a single-element array with --array={array_idx}-{array_idx}
                # This ensures $SLURM_ARRAY_TASK_ID is set to the correct index
                job_ids = self._slurm.submit_array(
                    array_name=state.name,
                    script_path=state.registration.script_path,
                    log_paths=[state.registration.log_path],
                    task_names=[state.name],
                    start_index=array_idx,  # Critical: preserve the original array index
                )
                new_job_id = job_ids[0] if job_ids else None
                if new_job_id is None:
                    raise RuntimeError(f"Array job submission failed for {state.name}")
                LOGGER.info(
                    f"[job {old_job_id}] restarted as {new_job_id} with array index {array_idx}"
                )
            else:
                # Fallback to regular submission if we can't parse the array index
                LOGGER.warning(
                    f"[job {old_job_id}] has underscore but can't parse array index, using regular submit"
                )
                new_job_id = self._slurm.submit(
                    state.name, state.registration.script_path, state.registration.log_path
                )
        else:
            # Regular single job restart
            LOGGER.info(f"[job {old_job_id}] restarting as regular single job")
            new_job_id = self._slurm.submit(
                state.name, state.registration.script_path, state.registration.log_path
            )

        self._jobs.pop(old_job_id, None)
        state.job_id = new_job_id
        state.attempts += 1
        state.last_slurm_state = None
        state.state = PendingState(PendingStateConfig())
        self._jobs[new_job_id] = state
        if self._state_store:
            self._state_store.remove_job(old_job_id)
        self._persist_job(state)
        return new_job_id

    def _capture_slurm_transitions(
        self,
        state: JobRuntimeState,
        current_state: str | None,
    ) -> list[MonitorRecord]:
        previous = state.last_slurm_state
        records: list[MonitorRecord] = []
        if current_state != previous:
            LOGGER.info(
                (f"[job {state.job_id}] SLURM state transition: {previous or 'NONE'} ")
                + (f"-> {current_state or 'NOT_FOUND'}")
            )
            metadata = {
                "slurm_state": current_state or "NOT_FOUND",
                "previous_state": previous or "NONE",
            }
            if current_state == "RUNNING":
                self._set_state(state, "running")
                record = self._queue_event(
                    job_id=state.job_id,
                    job_name=state.name,
                    event_name="slurm_state_transition",
                    state_key=state.state.key,
                    metadata=metadata,
                    action_name="run_started",
                    payload={"type": "slurm", **metadata},
                )
                if record:
                    records.append(record)
            if current_state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}:
                self._set_state(state, current_state.lower())
                record = self._queue_event(
                    job_id=state.job_id,
                    job_name=state.name,
                    event_name="slurm_state_transition",
                    state_key=state.state.key,
                    metadata=metadata,
                    action_name="run_ended",
                    payload={"type": "slurm", **metadata},
                )
                if record:
                    records.append(record)
            if current_state is None and previous is not None:
                self._set_state(state, "timeout")
                record = self._queue_event(
                    job_id=state.job_id,
                    job_name=state.name,
                    event_name="slurm_state_transition",
                    state_key=state.state.key,
                    metadata=metadata,
                    action_name="run_ended",
                    payload={"type": "slurm", **metadata},
                )
                if record:
                    records.append(record)
        state.last_slurm_state = current_state
        self._persist_job(state)
        return records

    @staticmethod
    def _classify_mode(
        state: JobRuntimeState,
        outcome: MonitorOutcome | None,
        slurm_snapshot: dict[str, str],
    ) -> tuple[str, dict[str, Any]] | None:
        """Classify job state into a mode with associated metadata.

        Returns:
            Tuple of (mode, metadata) or None if no classification applies
        """
        if outcome and outcome.status == "complete":
            return "success", {"reason": "termination_condition_met"}

        slurm_state = slurm_snapshot.get(state.job_id)
        if outcome and outcome.status == "stall":
            return "stall", {"reason": "inactivity_timeout"}
        if slurm_state is None:
            return "timeout", {"reason": "job_not_in_queue", "error_type": "timeout"}
        if slurm_state == "CANCELLED":
            # CANCELLED could be manual intervention or external issue
            # Mark as potentially restartable by default
            return "crash", {
                "reason": "job_cancelled",
                "slurm_state": "CANCELLED",
                "error_type": "cancelled",
                "subsystem": "slurm",
            }
        if slurm_state == "FAILED":
            return "crash", {
                "reason": "job_failed",
                "slurm_state": "FAILED",
                "error_type": "slurm_failure",
            }
        if slurm_state == "COMPLETED":
            return "success", {"reason": "slurm_completed", "slurm_state": "COMPLETED"}
        if slurm_state == "TIMEOUT":
            return "timeout", {
                "reason": "slurm_timeout",
                "slurm_state": "TIMEOUT",
                "error_type": "timeout",
            }
        return None

    def _build_job_metadata(self, state: JobRuntimeState) -> dict[str, Any]:
        metadata = dict(state.registration.metadata)
        metadata.setdefault("job_name", state.name)
        metadata.setdefault("job_id", state.job_id)
        output_dir = metadata.get("output_dir")
        if not output_dir:
            output_dir = str(Path(state.registration.log_path).parent)
            metadata["output_dir"] = output_dir
        if state.registration.inactivity_threshold_seconds is not None:
            metadata.setdefault(
                "inactivity_threshold_seconds",
                state.registration.inactivity_threshold_seconds,
            )
        if state.registration.output_paths:
            metadata.setdefault(
                "output_paths",
                [str(path) for path in state.registration.output_paths],
            )
        return metadata

    def _expand_log_path(self, job_id: str, log_path: str) -> Path:
        """Expand SLURM log path templates (%j, %A, %a) to actual paths.

        Args:
            job_id: Job ID (synthetic for array jobs, real for single jobs)
            log_path: str with potential SLURM templates

        Returns:
            Path with templates expanded
        """
        log_str = str(log_path)

        if "_" in job_id:
            base_id, array_idx = job_id.split("_")
            log_str = log_str.replace("%A", str(base_id))
            log_str = log_str.replace("%a", str(array_idx))

        # Single job: expand %j to job_id
        log_str = log_str.replace("%j", str(job_id))
        if str(log_path) != log_str:
            LOGGER.debug(f"[job {job_id}] expanded single job log path: {log_path} -> {log_str}")

        return Path(log_str)

    def _event_key(self, job_id: str, event_name: str) -> tuple[str, str]:
        return (str(job_id), event_name)

    def _get_or_create_event_record(
        self,
        state: JobRuntimeState,
        monitor_event: MonitorEvent,
    ) -> EventRecord:
        key = self._event_key(state.job_id, monitor_event.name)
        event_id = self._event_index.get(key)
        if event_id and event_id in self._event_records:
            record = self._event_records[event_id]
            record.touch(payload=dict(monitor_event.metadata))
            return record
        event_id = f"{state.job_id}:{monitor_event.name}:{int(time.time() * 1000)}"
        metadata = {"job_id": state.job_id, "job_name": state.name}
        metadata.update(monitor_event.metadata)
        record = EventRecord(
            event_id=event_id,
            name=monitor_event.name,
            source="monitor",
            payload=dict(monitor_event.metadata),
            metadata=metadata,
        )
        self._event_index[key] = event_id
        self._event_records[event_id] = record
        return record

    def _persist_event(self, record: EventRecord) -> None:
        if self._state_store is not None:
            self._state_store.upsert_event(record)

    def _maybe_release_event(self, job_id: str, record: EventRecord) -> None:
        if record.status in {EventStatus.PROCESSED, EventStatus.FAILED}:
            key = self._event_key(job_id, record.name)
            self._event_index.pop(key, None)

    def _handle_monitor_event(
        self,
        state: JobRuntimeState,
        monitor_event: MonitorEvent,
        cycle_result: MonitorCycleResult,
    ) -> tuple[MonitorDecision, str] | None:
        event_metadata = dict(monitor_event.metadata)
        event_metadata.setdefault("event_name", monitor_event.name)
        event_metadata.setdefault("note", monitor_event.name)

        event_record = self._get_or_create_event_record(state, monitor_event)
        workspace = (
            Path(state.registration.script_path).parent if state.registration.script_path else None
        )
        action_outcome = self._execute_event_actions(state, monitor_event, event_record, workspace)

        self._persist_event(event_record)
        self._maybe_release_event(state.job_id, event_record)

        event_record_entry = self._queue_event(
            state.job_id,
            state.name,
            monitor_event.name,
            monitor_event.state.key if monitor_event.state else None,
            metadata=event_metadata,
            action_name="event",
            payload={"status": event_record.status.value, "count": event_record.count},
        )
        if event_record_entry:
            cycle_result.events.append(event_record_entry)

        if monitor_event.actions:
            summary_record = self._queue_event(
                state.job_id,
                state.name,
                monitor_event.name,
                monitor_event.state.key if monitor_event.state else None,
                metadata=event_metadata,
                action_name="actions",
                payload={
                    "restart": action_outcome["restart"],
                    "queued": action_outcome["queued"],
                    "results": [result.message for result in action_outcome["results"]],
                },
            )
            if summary_record:
                cycle_result.events.append(summary_record)

        if monitor_event.state:
            LOGGER.info(
                f"[job {state.job_id}] detected event '{monitor_event.name}' with state '{monitor_event.state.key}'"
            )
            state.state = monitor_event.state
        else:
            LOGGER.info(f"[job {state.job_id}] detected event '{monitor_event.name}'")

        decision_entry = self._finalize_event(state, monitor_event, action_outcome)
        if decision_entry:
            decision, job_key = decision_entry
            cycle_result.decisions[job_key] = decision
        return decision_entry

    def _queue_event(
        self,
        job_id: str,
        job_name: str,
        event_name: str,
        state_key: str | None,
        metadata: dict[str, Any],
        *,
        action_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> MonitorRecord | None:
        record = MonitorRecord(
            job_id=job_id,
            job_name=job_name,
            event=event_name,
            state=state_key,
            action=action_name,
            payload=payload or {},
            metadata=dict(metadata),
        )
        self._pending_records.append(record)
        return record

    def _execute_event_actions(
        self,
        state: JobRuntimeState,
        monitor_event: MonitorEvent,
        event_record: EventRecord,
        workspace: Path | None,
    ) -> dict[str, Any]:
        job_metadata = self._build_job_metadata(state)
        condition_context = ConditionContext(
            event=event_record,
            job_metadata=job_metadata,
            attempts=state.attempts,
        )
        action_context = ActionContext(
            event=event_record,
            job_metadata=job_metadata,
            attempts=state.attempts,
            workspace=workspace,
        )

        restart_requested = False
        queued_ids: list[str] = []
        inline_results: list[ActionResult] = []

        if not monitor_event.actions:
            event_record.set_status(EventStatus.PROCESSED, note="no actions configured")
            return {"restart": False, "queued": [], "results": []}

        for binding in monitor_event.actions:
            status = self._evaluate_action_conditions(binding, condition_context, event_record)
            if status == "waiting":
                continue
            if status == "fail":
                continue

            if binding.mode == "queue":
                if self._action_queue is None:
                    raise RuntimeError("Action queue not configured but queue mode requested")
                config_payload = asdict(binding.action.config)
                record = self._action_queue.enqueue(
                    binding.action.config.class_name,
                    config_payload,
                    event_id=event_record.event_id,
                    metadata={"job": job_metadata, "event": event_record.event_id},
                )
                queued_ids.append(record.queue_id)
                event_record.set_status(EventStatus.PENDING, note="action queued")
                continue

            result = binding.action.execute(action_context)
            binding.action.update_event(event_record, result)
            inline_results.append(result)
            if result.status == "retry":
                restart_requested = True
            event_record.metadata["last_action_ts"] = time.time()

        return {
            "restart": restart_requested,
            "queued": queued_ids,
            "results": inline_results,
        }

    def _evaluate_action_conditions(
        self,
        binding: EventActionBinding,
        context: ConditionContext,
        event_record: EventRecord,
    ) -> Literal["pass", "waiting", "fail"]:
        if not binding.conditions:
            return "pass"
        for condition in binding.conditions:
            result = condition.check(context)
            if result.status == "waiting":
                event_record.set_status(
                    EventStatus.PENDING, note=result.message or "condition waiting"
                )
                return "waiting"
            if result.status == "fail":
                event_record.set_status(
                    EventStatus.FAILED, note=result.message or "condition failed"
                )
                return "fail"
        return "pass"

    def _finalize_event(
        self,
        state: JobRuntimeState,
        monitor_event: MonitorEvent,
        action_outcome: dict[str, Any],
    ) -> tuple[MonitorDecision, str] | None:
        if action_outcome["restart"]:
            LOGGER.info(
                f"[job {state.job_id}] restarting job due to event '{monitor_event.name}' "
                f"(attempt {state.attempts} -> {state.attempts + 1})"
            )
            new_job_id = self._restart_job(state, adjustments=None)
            decision = MonitorDecision(
                action="restart",
                reason=f"{monitor_event.name} requested restart",
                metadata={"event": monitor_event.name, "new_job_id": new_job_id},
            )
            return decision, new_job_id

        state_key = monitor_event.state.key if monitor_event.state else None
        if state_key == "success":
            LOGGER.info(
                f"[job {state.job_id}] job completed successfully via '{monitor_event.name}'"
            )
            self._finalize_job(state.job_id)
            decision = MonitorDecision(
                action="success",
                reason="job completed",
                metadata={"event": monitor_event.name},
            )
            return decision, state.job_id

        if state_key in {"crash", "stall", "timeout"}:
            reason = monitor_event.metadata.get("reason") or f"{state_key} detected"
            LOGGER.info(f"[job {state.job_id}] stopping after '{monitor_event.name}': {reason}")
            self._finalize_job(state.job_id)
            decision = MonitorDecision(
                action="stop",
                reason=reason,
                metadata={"event": monitor_event.name, **monitor_event.metadata},
            )
            return decision, state.job_id

        return None

    def _build_state_event(
        self,
        state: JobRuntimeState,
        event_name: str,
        metadata: dict[str, Any],
        default_state: BaseMonitorState | None = None,
    ) -> MonitorEvent:
        cfg = self._state_event_configs.get(event_name)
        merged_metadata = dict(cfg.metadata if cfg else {})
        merged_metadata.setdefault("job_name", state.name)
        merged_metadata.update(metadata)
        if cfg and cfg.state is not None:
            event_state = cfg.state.instantiate(MonitorStateInterface)
        else:
            event_state = default_state or _fallback_state_for(event_name)
        actions = instantiate_bindings(cfg.actions) if cfg else []
        return MonitorEvent(
            job_id=state.job_id,
            name=event_name,
            state=event_state,
            metadata=merged_metadata,
            actions=actions,
        )


def _fallback_state_for(name: str) -> BaseMonitorState | None:
    key = name.lower()
    if key == "stall":
        return StalledState(StalledStateConfig())
    if key == "timeout":
        return TimeoutState(TimeoutStateConfig())
    if key == "crash":
        return CrashState(CrashStateConfig())
    if key == "success":
        return SuccessState(SuccessStateConfig())
    return None


__all__ = [
    "MonitorController",
    "JobRuntimeState",
    "JobRegistration",
    "MonitorRecord",
    "MonitorCycleResult",
]
