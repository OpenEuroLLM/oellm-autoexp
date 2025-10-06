"""Monitoring controller bridging SLURM state and restart policies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from oellm_autoexp.monitor.policy import BaseRestartPolicy, RestartDecision, RestartEvent
from oellm_autoexp.monitor.watcher import BaseMonitor, MonitoredJob, MonitorOutcome
from oellm_autoexp.persistence import MonitorStateStore, StoredJob
from oellm_autoexp.slurm.client import BaseSlurmClient
from oellm_autoexp.utils.start_condition import (
    resolve_start_condition_interval,
    wait_for_start_condition,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class JobRegistration:
    """Information required to resubmit and observe a job."""

    name: str
    script_path: Path
    log_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    termination_string: Optional[str] = None
    termination_command: Optional[str] = None
    inactivity_threshold_seconds: Optional[int] = None
    output_paths: List[Path] = field(default_factory=list)
    start_condition_cmd: Optional[str] = None
    start_condition_interval_seconds: Optional[int] = None


@dataclass
class JobRuntimeState:
    job_id: int
    registration: JobRegistration
    attempts: int = 1
    last_outcome: Optional[MonitorOutcome] = None
    last_slurm_state: Optional[str] = None

    @property
    def name(self) -> str:
        return self.registration.name


@dataclass
class MonitorAction:
    """Action triggered by monitor signal processing."""

    job_id: int
    job_name: str
    action: str
    signal: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorCycleResult:
    """Aggregated outcome of a single monitoring iteration."""

    decisions: Dict[int, RestartDecision] = field(default_factory=dict)
    actions: List[MonitorAction] = field(default_factory=list)


class MonitorController:
    """Coordinate monitors, SLURM state, and restart policies."""

    def __init__(
        self,
        monitor: BaseMonitor,
        slurm: BaseSlurmClient,
        policies: Dict[str, BaseRestartPolicy],
        state_store: Optional[MonitorStateStore] = None,
    ) -> None:
        self._monitor = monitor
        self._slurm = slurm
        self._policies = policies
        self._jobs: Dict[int, JobRuntimeState] = {}
        self._pending_actions: List[MonitorAction] = []
        self._state_store = state_store

    def register_job(self, job_id: int, registration: JobRegistration, attempts: int = 1) -> None:
        state = JobRuntimeState(job_id=job_id, registration=registration, attempts=max(1, attempts))
        self._jobs[job_id] = state
        self._persist_job(state)

    def jobs(self) -> Iterable[JobRuntimeState]:
        return list(self._jobs.values())

    async def observe_once(self) -> "MonitorCycleResult":
        interval = getattr(
            self._monitor.config,
            "check_interval_seconds",
            getattr(self._monitor.config, "poll_interval_seconds", 60),
        )
        monitored_jobs = [
            MonitoredJob(
                job_id=state.job_id,
                name=state.name,
                log_path=state.registration.log_path,
                check_interval_seconds=interval,
                termination_string=state.registration.termination_string,
                termination_command=state.registration.termination_command,
                metadata=self._build_job_metadata(state),
                output_paths=list(state.registration.output_paths),
            )
            for state in self._jobs.values()
        ]
        outcomes = await self._monitor.watch(monitored_jobs)
        slurm_snapshot = self._slurm.squeue()

        cycle_result = MonitorCycleResult()
        for state in list(self._jobs.values()):
            outcome = outcomes.get(state.job_id)
            state.last_outcome = outcome
            slurm_state = slurm_snapshot.get(state.job_id)
            self._capture_slurm_transitions(state, slurm_state, cycle_result)
            handled = False
            if outcome is not None:
                for signal in outcome.signals:
                    if signal.mode:
                        result = self._apply_policy(state, signal.mode, signal.metadata)
                        if result is not None:
                            decision, job_key = result
                            cycle_result.decisions[job_key] = decision
                            handled = True
                    if signal.action:
                        cycle_result.actions.append(
                            self._record_action(
                                job_id=state.job_id,
                                job_name=state.name,
                                action=signal.action,
                                signal=signal.name,
                                metadata=dict(signal.metadata),
                            )
                        )
            if handled:
                continue
            mode = self._classify_mode(state, outcome, slurm_snapshot)
            if mode is None:
                continue
            result = self._apply_policy(state, mode, outcome.metadata if outcome else {})
            if result is None:
                continue
            decision, job_key = result
            cycle_result.decisions[job_key] = decision
        return cycle_result

    def observe_once_sync(self) -> "MonitorCycleResult":
        return asyncio.run(self.observe_once())

    def handle_state_change(self, job_id: int, mode: str) -> RestartDecision:
        state = self._jobs[job_id]
        result = self._apply_policy(state, mode, {})
        if result is None:
            return RestartDecision(action="stop", reason="no policy configured")
        decision, _ = result
        return decision

    def snapshot(self) -> Dict[int, str]:
        return self._slurm.squeue()

    def drain_actions(self) -> List[MonitorAction]:
        actions = self._pending_actions
        self._pending_actions = []
        return actions

    def clear_state(self) -> None:
        if self._state_store:
            self._state_store.clear()

    def _persist_job(self, state: JobRuntimeState) -> None:
        if not self._state_store:
            return
        stored = StoredJob.from_registration(state.job_id, state.attempts, state.registration)
        self._state_store.upsert_job(stored)

    def _finalize_job(self, job_id: int) -> None:
        self._jobs.pop(job_id, None)
        self._slurm.remove(job_id)
        if self._state_store:
            self._state_store.remove_job(job_id)

    def _restart_job(self, state: JobRuntimeState) -> int:
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
        new_job_id = self._slurm.submit(state.name, state.registration.script_path, state.registration.log_path)
        self._jobs.pop(old_job_id, None)
        state.job_id = new_job_id
        state.attempts += 1
        state.last_slurm_state = None
        self._jobs[new_job_id] = state
        if self._state_store:
            self._state_store.remove_job(old_job_id)
        self._persist_job(state)
        return new_job_id

    def _apply_policy(
        self,
        state: JobRuntimeState,
        mode: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[RestartDecision, int]]:
        policy = self._policies.get(mode)
        if policy is None:
            return None
        event = RestartEvent(mode=mode, attempt=state.attempts, metadata=metadata or {})
        decision = policy.decide(event)
        if decision.action == "restart":
            new_job_id = self._restart_job(state)
            augmented = RestartDecision(
                action="restart",
                reason=decision.reason,
                adjustments={**(decision.adjustments or {}), "new_job_id": new_job_id},
            )
            return augmented, new_job_id
        if mode == "success":
            self._finalize_job(state.job_id)
        else:
            self._persist_job(state)
        return decision, state.job_id

    def _capture_slurm_transitions(
        self,
        state: JobRuntimeState,
        current_state: Optional[str],
        cycle_result: "MonitorCycleResult",
    ) -> None:
        previous = state.last_slurm_state
        if current_state != previous:
            if current_state == "RUNNING":
                cycle_result.actions.append(
                    self._record_action(
                        job_id=state.job_id,
                        job_name=state.name,
                        action="run_started",
                        signal="slurm-run-started",
                        metadata={"slurm_state": current_state or "UNKNOWN"},
                    )
                )
            if current_state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}:
                cycle_result.actions.append(
                    self._record_action(
                        job_id=state.job_id,
                        job_name=state.name,
                        action="run_ended",
                        signal="slurm-run-ended",
                        metadata={"slurm_state": current_state},
                    )
                )
            if current_state is None and previous is not None:
                cycle_result.actions.append(
                    self._record_action(
                        job_id=state.job_id,
                        job_name=state.name,
                        action="run_ended",
                        signal="slurm-run-ended",
                        metadata={"slurm_state": "UNKNOWN"},
                    )
                )
        state.last_slurm_state = current_state
        self._persist_job(state)

    @staticmethod
    def _classify_mode(
        state: JobRuntimeState,
        outcome: Optional[MonitorOutcome],
        slurm_snapshot: Dict[int, str],
    ) -> Optional[str]:
        if outcome and outcome.status == "complete":
            return "success"

        slurm_state = slurm_snapshot.get(state.job_id)
        if outcome and outcome.status == "stall":
            return "stall"
        if slurm_state is None:
            return "timeout"
        if slurm_state in {"FAILED", "CANCELLED"}:
            return "crash"
        if slurm_state == "COMPLETED":
            return "success"
        if slurm_state == "TIMEOUT":
            return "timeout"
        return None

    def _build_job_metadata(self, state: JobRuntimeState) -> Dict[str, Any]:
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

    def _record_action(
        self,
        job_id: int,
        job_name: str,
        action: str,
        signal: str,
        metadata: Dict[str, Any],
    ) -> MonitorAction:
        item = MonitorAction(job_id=job_id, job_name=job_name, action=action, signal=signal, metadata=metadata)
        self._pending_actions.append(item)
        return item


__all__ = [
    "MonitorController",
    "JobRuntimeState",
    "JobRegistration",
    "MonitorAction",
    "MonitorCycleResult",
]
