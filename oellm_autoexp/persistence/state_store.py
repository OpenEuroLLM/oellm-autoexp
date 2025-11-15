"""Simple JSON persistence for monitor state."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, MISSING
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from compoconf import asdict

from oellm_autoexp.monitor.events import EventRecord


def _serialize_for_json(obj: Any) -> Any:
    """Recursively convert Path objects and other non-serializable types to
    JSON-compatible types."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except Exception:
            return None


@dataclass(kw_only=True)
class StoredJob:
    job_id: str = field(default_factory=MISSING)
    name: str = field(default_factory=MISSING)
    script_path: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)
    termination_string: str | None = None
    termination_command: str | None = None
    inactivity_threshold_seconds: int | None = None
    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None
    resolved_log_path: str | None = None
    last_monitor_state: str | None = None
    last_slurm_state: str | None = None
    last_updated: float = field(default_factory=time.time)

    @staticmethod
    def from_registration(
        job_id: str,
        attempts: int,
        registration: Any,
        *,
        resolved_log_path: str | None = None,
        monitor_state: str | None = None,
        slurm_state: str | None = None,
    ) -> StoredJob:
        if resolved_log_path is None:
            resolved_log_path = str(registration.log_path)
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
            resolved_log_path=resolved_log_path,
            last_monitor_state=monitor_state,
            last_slurm_state=slurm_state,
            last_updated=time.time(),
        )


class MonitorStateStore:
    """Persist monitor state to disk for crash-resilient restarts.

    Each monitoring session gets its own file in the monitoring_state
    directory. Session files contain the full config and job state for
    that session.
    """

    def __init__(self, root: str, session_id: str | None = None) -> None:
        self._root = Path(root)
        self._session_id = session_id or str(uuid.uuid4())[:8]
        # Canonical session file
        self._session_path = self._root / f"{self._session_id}.json"
        self._session_path.parent.mkdir(parents=True, exist_ok=True)

        # Legacy layout expected by older tooling/tests: <root>/monitor/state.json
        self._legacy_path = self._root / "monitor" / "state.json"
        self._legacy_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_path(self) -> Path:
        return self._session_path

    @property
    def descriptor_path(self) -> Path:
        """Alias for backward compatibility."""
        return self._session_path

    def _load_payload(self) -> dict[str, Any]:
        """Load JSON payload from session or legacy location."""
        for path in (self._session_path, self._legacy_path):
            if not path.exists():
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
        return {}

    def _write_payload(self, payload: dict[str, Any]) -> None:
        """Persist payload to session file and legacy compatibility path."""
        text = json.dumps(payload, indent=2)
        self._session_path.write_text(text, encoding="utf-8")
        try:
            self._legacy_path.write_text(text, encoding="utf-8")
        except OSError:
            # Legacy path may live on read-only storage; ignore if we cannot mirror.
            pass

    def load(self) -> dict[str, StoredJob]:
        """Load jobs from session file."""
        payload = self._load_payload()
        if not payload:
            return {}
        jobs: dict[str, StoredJob] = {}
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
                    resolved_log_path=raw.get("resolved_log_path") or raw.get("expanded_log_path"),
                    last_monitor_state=raw.get("last_monitor_state"),
                    last_slurm_state=raw.get("last_slurm_state"),
                    last_updated=float(raw.get("last_updated", time.time())),
                )
            except (TypeError, ValueError):
                continue
            jobs[job.job_id] = job
        return jobs

    def save_jobs(self, jobs: Iterable[StoredJob]) -> None:
        """Save jobs to session file (merges with existing config if
        present)."""
        # Try to load existing session data to preserve config
        existing_data = self._load_payload()

        # Update with new job data
        payload = {
            **existing_data,
            "session_id": self._session_id,
            "timestamp": time.time(),
            "jobs": [asdict(job) for job in jobs],
        }
        self._write_payload(payload)

    def load_events(self) -> dict[str, EventRecord]:
        payload = self._load_payload()
        events: dict[str, EventRecord] = {}
        for raw in payload.get("events", []):
            try:
                record = EventRecord.from_dict(raw)
            except (KeyError, ValueError):
                continue
            events[record.event_id] = record
        return events

    def save_events(self, events: Iterable[EventRecord]) -> None:
        existing_data = self._load_payload()
        payload = {
            **existing_data,
            "session_id": self._session_id,
            "timestamp": time.time(),
            "events": [event.to_dict() for event in events],
        }
        self._write_payload(payload)

    def upsert_event(self, event: EventRecord) -> None:
        records = self.load_events()
        records[event.event_id] = event
        self.save_events(records.values())

    def remove_event(self, event_id: str) -> None:
        records = self.load_events()
        if event_id in records:
            records.pop(event_id)
            self.save_events(records.values())

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
        """Remove session file."""
        if self._session_path.exists():
            self._session_path.unlink()
        if self._legacy_path.exists():
            self._legacy_path.unlink()

    def save_session(self, config_dict: dict[str, Any], project_name: str) -> None:
        """Save monitoring session with full config for resumability.

        Serializes Path objects and other non-JSON-compatible types to
        strings.
        """
        # Serialize config_dict to convert Path objects to strings
        serialized_config = _serialize_for_json(config_dict)

        session_data = {
            "session_id": self._session_id,
            "project_name": project_name,
            "created_at": time.time(),
            "config": serialized_config,
            "jobs": [],  # Will be populated by save_jobs
        }
        self._write_payload(session_data)

    @staticmethod
    def list_sessions(monitoring_state_dir: str | Path) -> list[dict[str, Any]]:
        """List all monitoring sessions in the monitoring_state directory."""
        monitoring_state_dir = Path(monitoring_state_dir)
        if not monitoring_state_dir.exists():
            return []

        sessions = []
        for session_file in monitoring_state_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text(encoding="utf-8"))
                sessions.append(
                    {
                        "session_id": data.get("session_id", session_file.stem),
                        "project_name": data.get("project_name", "unknown"),
                        "created_at": data.get("created_at", 0),
                        "session_path": str(session_file),
                        "job_count": len(data.get("jobs", [])),
                        "manifest_path": data.get("manifest_path"),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)

    @staticmethod
    def load_session(session_path: str | Path) -> dict[str, Any] | None:
        """Load a monitoring session file."""
        session_path = Path(session_path)
        if not session_path.exists():
            return None
        try:
            return json.loads(session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


__all__ = ["MonitorStateStore", "StoredJob"]
