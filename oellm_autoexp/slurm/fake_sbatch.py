"""In-memory SLURM stand-in used for integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class FakeJob:
    job_id: int
    name: str
    script_path: Path
    log_path: Path
    state: str = "PENDING"
    return_code: Optional[int] = None
    last_update: float = 0.0


class FakeSlurm:
    """Minimal SLURM controller for tests."""

    def __init__(self) -> None:
        self._jobs: Dict[int, FakeJob] = {}
        self._next_id = 1

    def submit(self, name: str, script_path: Path, log_path: Path) -> int:
        job_id = self._next_id
        self._next_id += 1
        self._jobs[job_id] = FakeJob(job_id=job_id, name=name, script_path=script_path, log_path=log_path)
        return job_id

    def set_state(self, job_id: int, state: str, return_code: Optional[int] = None) -> None:
        job = self._jobs[job_id]
        job.state = state
        if return_code is not None:
            job.return_code = return_code

    def squeue(self) -> Dict[int, str]:
        return {job_id: job.state for job_id, job in self._jobs.items()}

    def get_job(self, job_id: int) -> FakeJob:
        return self._jobs[job_id]


__all__ = ["FakeSlurm", "FakeJob"]

