"""Monitoring controller bridging SLURM state and restart policies."""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
import logging
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from oellm_autoexp.monitor.actions import BaseMonitorAction, RestartAction
from oellm_autoexp.monitor.policy import BaseRestartPolicy, RestartDecision, RestartEvent
from oellm_autoexp.monitor.states import (
    BaseMonitorState,
    CrashState,
    CrashStateConfig,
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
from oellm_autoexp.monitor.watcher import BaseMonitor, MonitoredJob, MonitorOutcome
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


@dataclass(kw_only=True)
class MonitorCycleResult:
    """Aggregated outcome of a single monitoring iteration."""

    decisions: dict[str, RestartDecision] = field(default_factory=dict)
    events: list[MonitorRecord] = field(default_factory=list)


class MonitorController:
    """Coordinate monitors, SLURM state, and restart policies."""

    def __init__(
        self,
        monitor: BaseMonitor,
        slurm: BaseSlurmClient,
        policies: dict[str, BaseRestartPolicy],
        state_store: MonitorStateStore | None = None,
    ) -> None:
        self._monitor = monitor
        self._slurm = slurm
        self._policies = policies
        self._jobs: dict[str, JobRuntimeState] = {}  # Job IDs are strings
        self._pending_records: list[MonitorRecord] = []
        self._state_store = state_store

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
                    event_metadata = dict(event.metadata)
                    event_metadata.setdefault("event_name", event.name)
                    event_metadata.setdefault("signal_name", event.name)
                    if event.state:
                        state.state = event.state
                        event_metadata.setdefault("state_key", event.state.key)
                    base_record = self._queue_event(
                        job_id=state.job_id,
                        job_name=state.name,
                        event_name=event.name,
                        state_key=event.state.key if event.state else None,
                        action=None,
                        metadata=event_metadata,
                    )
                    if base_record:
                        cycle_result.events.append(base_record)
                    if event.state:
                        LOGGER.info(
                            f"[job {state.job_id}] detected event '{event.name}' with state '{event.state.key}'"
                        )
                    if event.actions:
                        LOGGER.info(
                            (f"[job {state.job_id}] detected event '{event.name}' ")
                            + (f"with actions {', '.join(action.kind for action in event.actions)}")
                        )
                    else:
                        LOGGER.info(f"[job {state.job_id}] detected event '{event.name}'")
                    restart_actions = [
                        action for action in event.actions if isinstance(action, RestartAction)
                    ]
                    for action in event.actions:
                        record = self._queue_event(
                            job_id=state.job_id,
                            job_name=state.name,
                            event_name=event.name,
                            state_key=event.state.key if event.state else None,
                            action=action,
                            metadata=event_metadata,
                        )
                        if record:
                            cycle_result.events.append(record)
                            LOGGER.info(
                                f"[job {state.job_id}] action '{action.kind}' triggered by event '{event.name}'"
                            )
                    for action in restart_actions:
                        mode_override = getattr(action.config, "mode", None)
                        mode = mode_override or (
                            event.state.key if event.state is not None else None
                        )
                        if not mode:
                            LOGGER.warning(
                                f"[job {state.job_id}] restart action '{action.kind}' missing mode; skipping"
                            )
                            continue
                        restart_metadata = dict(event_metadata)
                        if action.config.reason:
                            restart_metadata.setdefault("restart_reason", action.config.reason)
                        result = self._apply_policy(state, mode, restart_metadata)
                        if result is not None:
                            decision, job_key = result
                            cycle_result.decisions[job_key] = decision
                            handled = True
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
            # Merge outcome metadata with mode classification metadata
            combined_metadata = dict(outcome.metadata if outcome else {})
            combined_metadata.update(mode_metadata)
            result = self._apply_policy(state, mode, combined_metadata)
            if result is None:
                LOGGER.warning(f"[job {state.job_id}] no policy configured for mode '{mode}'")
                continue
            decision, job_key = result
            cycle_result.decisions[job_key] = decision
        return cycle_result

    async def observe_once(self) -> MonitorCycleResult:
        return self.observe_once_sync()

    def handle_state_change(self, job_id: str, mode: str) -> RestartDecision:
        state = self._jobs[job_id]
        result = self._apply_policy(state, mode, {})
        if result is None:
            return RestartDecision(action="stop", reason="no policy configured")
        decision, _ = result
        return decision

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
        stored = StoredJob.from_registration(state.job_id, state.attempts, state.registration)
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

    def _restart_job(self, state: JobRuntimeState) -> str:
        old_job_id = state.job_id
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

    def _apply_policy(
        self,
        state: JobRuntimeState,
        mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[RestartDecision, str] | None:
        policy = self._policies.get(mode)
        if policy is None:
            return None
        event = RestartEvent(mode=mode, attempt=state.attempts, metadata=metadata or {})
        decision = policy.decide(event)
        self._set_state(state, mode)
        LOGGER.info(
            f"[job {state.job_id}] policy decision for mode '{mode}': "
            f"action={decision.action}, reason={decision.reason}, attempt={state.attempts}"
        )
        if decision.action == "restart":
            LOGGER.info(
                f"[job {state.job_id}] restarting job (attempt {state.attempts} -> {state.attempts + 1})"
            )
            new_job_id = self._restart_job(state)
            LOGGER.info(f"[job {state.job_id}] restarted as job {new_job_id}")
            augmented = RestartDecision(
                action="restart",
                reason=decision.reason,
                adjustments={**(decision.adjustments or {}), "new_job_id": new_job_id},
            )
            return augmented, new_job_id
        if mode == "success":
            LOGGER.info(f"[job {state.job_id}] job completed successfully, finalizing")
        else:
            LOGGER.info(f"[job {state.job_id}] job stopped (no restart), reason: {decision.reason}")
        self._finalize_job(state.job_id)
        return decision, state.job_id

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
                    action=None,
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
                    action=None,
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
                    action=None,
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

    def _queue_event(
        self,
        job_id: str,
        job_name: str,
        event_name: str,
        state_key: str | None,
        action: BaseMonitorAction | None,
        metadata: dict[str, Any],
        *,
        action_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> MonitorRecord | None:
        derived_action = action_name
        derived_payload = payload
        if action is not None:
            derived_action = action.kind
            derived_payload = action.describe(job_id, metadata)
        if derived_payload is None:
            derived_payload = {}
        record = MonitorRecord(
            job_id=job_id,
            job_name=job_name,
            event=event_name,
            state=state_key,
            action=derived_action,
            payload=derived_payload,
            metadata=dict(metadata),
        )
        self._pending_records.append(record)
        return record


__all__ = [
    "MonitorController",
    "JobRuntimeState",
    "JobRegistration",
    "MonitorRecord",
    "MonitorCycleResult",
]
