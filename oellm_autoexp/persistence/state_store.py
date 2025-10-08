"""Simple JSON persistence for monitor state."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class StoredJob:
    job_id: str
    name: str
    script_path: str
    log_path: str
    attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)
    termination_string: Optional[str] = None
    termination_command: Optional[str] = None
    inactivity_threshold_seconds: Optional[int] = None
    start_condition_cmd: Optional[str] = None
    start_condition_interval_seconds: Optional[int] = None

    @staticmethod
    def from_registration(job_id: str, attempts: int, registration: Any) -> "StoredJob":
        return StoredJob(
            job_id=job_id,
            name=registration.name,
            script_path=str(registration.script_path),
            log_path=str(registration.log_path),
            attempts=attempts,
            metadata=dict(registration.metadata),
            output_paths=[str(path) for path in registration.output_paths],
            termination_string=registration.termination_string,
            termination_command=registration.termination_command,
            inactivity_threshold_seconds=registration.inactivity_threshold_seconds,
            start_condition_cmd=registration.start_condition_cmd,
            start_condition_interval_seconds=registration.start_condition_interval_seconds,
        )


class MonitorStateStore:
    """Persist monitor state to disk for crash-resilient restarts."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._path = self._root / "monitor" / "state.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[int, StoredJob]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        jobs: Dict[int, StoredJob] = {}
        for raw in payload.get("jobs", []):
            try:
                job = StoredJob(
                    job_id=str(raw["job_id"]),
                    name=raw.get("name", ""),
                    script_path=raw.get("script_path", ""),
                    log_path=raw.get("log_path", ""),
                    attempts=int(raw.get("attempts", 1)),
                    metadata=dict(raw.get("metadata", {})),
                    output_paths=list(raw.get("output_paths", [])),
                    termination_string=raw.get("termination_string"),
                    termination_command=raw.get("termination_command"),
                    inactivity_threshold_seconds=raw.get("inactivity_threshold_seconds"),
                    start_condition_cmd=raw.get("start_condition_cmd"),
                    start_condition_interval_seconds=raw.get("start_condition_interval_seconds"),
                )
            except (TypeError, ValueError):
                continue
            jobs[job.job_id] = job
        return jobs

    def save_jobs(self, jobs: Iterable[StoredJob]) -> None:
        payload = {
            "timestamp": time.time(),
            "jobs": [asdict(job) for job in jobs],
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert_job(self, job: StoredJob) -> None:
        records = self.load()
        records[job.job_id] = job
        self.save_jobs(records.values())

    def remove_job(self, job_id: str) -> None:
        records = self.load()
        if job_id in records:
            records.pop(job_id)
            if records:
                self.save_jobs(records.values())
            else:
                self.clear()

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()


__all__ = ["MonitorStateStore", "StoredJob"]
