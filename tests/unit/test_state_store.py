from __future__ import annotations

from oellm_autoexp.monitor.events import EventRecord, EventStatus
from oellm_autoexp.persistence.state_store import MonitorStateStore


def test_event_roundtrip(tmp_path) -> None:
    store = MonitorStateStore(tmp_path / "state")
    event = EventRecord(event_id="ev1", name="stall")

    store.upsert_event(event)
    loaded = store.load_events()
    assert "ev1" in loaded
    assert loaded["ev1"].status == EventStatus.TRIGGERED

    event.set_status(EventStatus.PROCESSED, note="handled")
    store.upsert_event(event)

    reloaded = store.load_events()
    assert reloaded["ev1"].status == EventStatus.PROCESSED
    assert reloaded["ev1"].history


def test_remove_event(tmp_path) -> None:
    store = MonitorStateStore(tmp_path / "state")
    event = EventRecord(event_id="cleanup", name="done")
    store.upsert_event(event)
    assert store.load_events()

    store.remove_event("cleanup")
    assert not store.load_events()
