"""Monitoring controller bridging SLURM state and restart policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from oellm_autoexp.monitor.policy import BaseRestartPolicy, RestartDecision, RestartEvent
from oellm_autoexp.monitor.watcher import BaseMonitor
from oellm_autoexp.slurm.fake_sbatch import FakeSlurm


@dataclass
class JobRuntimeState:
    job_id: int
    name: str
    attempts: int = 1


class MonitorController:
    """Minimal monitoring controller suitable for integration tests."""

    def __init__(self, monitor: BaseMonitor, slurm: FakeSlurm, policies: Dict[str, BaseRestartPolicy]) -> None:
        self._monitor = monitor
        self._slurm = slurm
        self._policies = policies
        self._jobs: Dict[int, JobRuntimeState] = {}

    def register_job(self, job_id: int, name: str) -> None:
        self._jobs[job_id] = JobRuntimeState(job_id=job_id, name=name)

    def handle_state_change(self, job_id: int, mode: str) -> RestartDecision:
        event = RestartEvent(mode=mode, attempt=self._jobs[job_id].attempts)
        policy = self._policies.get(mode)
        if policy is None:
            return RestartDecision(action="stop", reason="no policy configured")
        decision = policy.decide(event)
        if decision.action == "restart":
            self._jobs[job_id].attempts += 1
        return decision

    def snapshot(self) -> Dict[int, str]:
        return self._slurm.squeue()


__all__ = ["MonitorController", "JobRuntimeState"]

