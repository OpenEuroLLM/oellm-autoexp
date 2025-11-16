"""Lightweight JSON-backed queue for asynchronous monitor actions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ActionQueueStatus = Literal["pending", "running", "done", "failed"]


@dataclass(kw_only=True)
class QueuedAction:
    queue_id: str
    action_class: str
    config: dict[str, Any]
    event_id: str
    status: ActionQueueStatus = "pending"
    enqueued_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    metadata: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_id": self.queue_id,
            "action_class": self.action_class,
            "config": self.config,
            "event_id": self.event_id,
            "status": self.status,
            "enqueued_at": self.enqueued_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "result": self.result,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> QueuedAction:
        return QueuedAction(
            queue_id=payload["queue_id"],
            action_class=payload["action_class"],
            config=dict(payload.get("config", {})),
            event_id=payload["event_id"],
            status=payload.get("status", "pending"),
            enqueued_at=payload.get("enqueued_at", time.time()),
            updated_at=payload.get("updated_at", time.time()),
            metadata=dict(payload.get("metadata", {})),
            result=dict(payload.get("result", {})),
        )


class ActionQueue:
    """Simple queue storing one file per pending action under event-named
    directories."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _record_path(self, event_id: str, queue_id: str) -> Path:
        event_dir = self._root / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir / f"{queue_id}.json"

    def _find_record_path(self, queue_id: str) -> Path | None:
        for event_dir in self._root.iterdir():
            if not event_dir.is_dir():
                continue
            candidate = event_dir / f"{queue_id}.json"
            if candidate.exists():
                return candidate
        return None

    def _load_path(self, path: Path) -> QueuedAction | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return QueuedAction.from_dict(payload)
        except (KeyError, ValueError):
            return None

    def _write(self, record: QueuedAction) -> None:
        path = self._record_path(record.event_id, record.queue_id)
        path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

    def list(self) -> list[QueuedAction]:
        records: list[QueuedAction] = []
        for event_dir in sorted(self._root.iterdir()):
            if not event_dir.is_dir():
                continue
            for path in sorted(event_dir.glob("*.json")):
                record = self._load_path(path)
                if record is not None:
                    records.append(record)
        records.sort(key=lambda record: record.enqueued_at)
        return records

    def enqueue(
        self,
        action_class: str,
        config: dict[str, Any],
        *,
        event_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> QueuedAction:
        record = QueuedAction(
            queue_id=str(uuid.uuid4()),
            action_class=action_class,
            config=config,
            event_id=event_id,
            metadata=dict(metadata or {}),
        )
        self._write(record)
        return record

    def claim_next(self) -> QueuedAction | None:
        records = self.list()
        for record in records:
            if record.status != "pending":
                continue
            record.status = "running"
            record.updated_at = time.time()
            self._write(record)
            return record
        return None

    def mark_done(
        self,
        queue_id: str,
        *,
        status: ActionQueueStatus,
        result: dict[str, Any] | None = None,
    ) -> None:
        path = self._find_record_path(queue_id)
        if path is None:
            return
        record = self._load_path(path)
        if record is None:
            return
        record.status = status
        record.updated_at = time.time()
        if result is not None:
            record.result = result
        if status in {"done", "failed"}:
            try:
                path.unlink()
                # Clean up empty event directories to keep queue tidy.
                if path.parent != self._root and not any(path.parent.iterdir()):
                    path.parent.rmdir()
            except OSError:
                pass
        else:
            self._write(record)

    def load(self, queue_id: str) -> QueuedAction | None:
        """Return the queued action without mutating state."""
        path = self._find_record_path(queue_id)
        if path is None:
            return None
        return self._load_path(path)

    def retry(self, queue_id: str) -> bool:
        """Reset a running/pending queue entry to pending again."""
        record = self.load(queue_id)
        if record is None:
            return False
        record.status = "pending"
        record.updated_at = time.time()
        self._write(record)
        return True


__all__ = ["ActionQueue", "QueuedAction"]
