from oellm_autoexp.persistence.state_store import MonitorStateStore
from oellm_autoexp.monitor.events import EventRecord, EventStatus


def test_event_roundtrip(tmp_path) -> None:
    store = MonitorStateStore(tmp_path / "state")
    event = EventRecord(event_id="ev1", name="stall", source="test")
    store.upsert_event(event)

    loaded = store.load_events()
    assert len(loaded) == 1
    assert loaded["ev1"].name == "stall"
    assert loaded["ev1"].status == EventStatus.PENDING


def test_remove_event(tmp_path) -> None:
    store = MonitorStateStore(tmp_path / "state")
    event = EventRecord(event_id="cleanup", name="done", source="test")
    store.upsert_event(event)
    store.remove_event("cleanup")
    assert not store.load_events()
