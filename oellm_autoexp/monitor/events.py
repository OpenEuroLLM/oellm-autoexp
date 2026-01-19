"""Definitions for monitor events tracked across restarts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any

from monitor.actions import EventRecord as BaseEventRecord, EventStatus

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EventRecord(BaseEventRecord):
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if hasattr(self.status, "value"):
            d["status"] = self.status.value
        return d

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EventRecord:
        status_val = data.get("status", EventStatus.PENDING.value)
        # Handle legacy "triggered" status
        if status_val == "triggered":
            status_val = EventStatus.PENDING.value

        try:
            status = EventStatus(status_val)
        except ValueError:
            status = EventStatus.PENDING

        return EventRecord(
            event_id=str(data["event_id"]),
            name=str(data.get("name", "")),
            source=str(data.get("source", "monitor")),
            payload=dict(data.get("payload", {})),
            metadata=dict(data.get("metadata", {})),
            count=int(data.get("count", 1)),
            status=status,
            first_seen_ts=float(data.get("first_seen_ts", time.time())),
            last_seen_ts=float(data.get("last_seen_ts", time.time())),
            history=list(data.get("history", [])),
        )


__all__ = ["EventStatus", "EventRecord"]
