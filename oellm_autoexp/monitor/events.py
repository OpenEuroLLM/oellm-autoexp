"""Definitions for monitor events tracked across restarts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class EventStatus(str, Enum):
    """Lifecycle state for an event emitted by the watcher."""

    TRIGGERED = "triggered"
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass(kw_only=True)
class EventRecord:
    """Stateful representation of an emitted event."""

    event_id: str
    name: str
    source: str = "monitor"
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    count: int = 1
    status: EventStatus = EventStatus.TRIGGERED
    first_seen_ts: float = field(default_factory=lambda: time.time())
    last_seen_ts: float = field(default_factory=lambda: time.time())
    history: list[dict[str, Any]] = field(default_factory=list)

    def touch(self, *, payload: dict[str, Any] | None = None) -> None:
        """Increment the occurrence counter and update timestamps."""
        self.count += 1
        self.last_seen_ts = time.time()
        if payload:
            self.payload.update(payload)

    def set_status(self, status: EventStatus, *, note: str | None = None) -> None:
        """Move event into a new lifecycle state and append optional note."""
        self.status = status
        self.last_seen_ts = time.time()
        if note:
            self.history.append({"ts": self.last_seen_ts, "status": status, "note": note})

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "source": self.source,
            "payload": self.payload,
            "metadata": self.metadata,
            "count": self.count,
            "status": self.status.value,
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
            "history": list(self.history),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EventRecord:
        return EventRecord(
            event_id=str(data["event_id"]),
            name=str(data.get("name", "")),
            source=str(data.get("source", "monitor")),
            payload=dict(data.get("payload", {})),
            metadata=dict(data.get("metadata", {})),
            count=int(data.get("count", 1)),
            status=EventStatus(data.get("status", EventStatus.TRIGGERED.value)),
            first_seen_ts=float(data.get("first_seen_ts", time.time())),
            last_seen_ts=float(data.get("last_seen_ts", time.time())),
            history=list(data.get("history", [])),
        )


__all__ = ["EventStatus", "EventRecord"]
