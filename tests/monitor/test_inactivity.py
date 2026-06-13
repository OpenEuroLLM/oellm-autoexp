"""Tests for inactivity-triggered restarts in the monitor loop.

These exercise the ``pattern_type: "inactivity"`` log event end-to-end using
real *local* (non-SLURM) jobs: a ``sleep`` process produces no output, so its
log stays inactive and the streak accumulates one count per poll. The
time-threshold cases inject a controllable clock so the "N minutes" branch is
deterministic without real waiting.

See ``LogEvent.inactivity_qualifies`` / ``MonitorLoop._process_inactivity_event``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compoconf import asdict

from oellm_autoexp.monitor import loop as loop_mod
from oellm_autoexp.monitor.actions import (
    EventRecord,
    LogEvent,
    LogEventConfig,
    RestartActionConfig,
)
from oellm_autoexp.monitor.loop import (
    JobFileStore,
    JobRecord,
    JobRuntime,
    MonitorLoop,
)
from oellm_autoexp.monitor.local_client import LocalCommandClient
from oellm_autoexp.monitor.submission import LocalJobConfig


# --------------------------------------------------------------------------- #
# Pure-logic unit tests (no processes, no clock)
# --------------------------------------------------------------------------- #


def test_check_triggers_is_noop_for_inactivity():
    """Inactivity is time/poll based, not text based: check_triggers must not
    match against log content."""
    cfg = LogEventConfig(name="inactive", pattern_type="inactivity")
    event = LogEvent(cfg)
    assert event.check_triggers("anything at all\nmore text") == []
    assert event.check_triggers("") == []


def test_inactivity_metadata_is_stable_and_marked():
    cfg = LogEventConfig(name="inactive", pattern_type="inactivity", metadata={"k": "v"})
    event = LogEvent(cfg)
    md = event.inactivity_metadata()
    assert md["inactive"] is True
    assert md["k"] == "v"
    # Stable across calls -> stable event_key across polls.
    assert event.inactivity_metadata() == md


@pytest.mark.parametrize(
    "polls, timeout_s, count, elapsed_s, expected",
    [
        # polls-only (timeout disabled)
        (5, 0.0, 4, 999.0, False),
        (5, 0.0, 5, 0.0, True),
        # time-only (polls floor at default 1)
        (1, 300.0, 10, 299.0, False),
        (1, 300.0, 2, 300.0, True),
        # AND: both must hold
        (5, 300.0, 5, 299.0, False),  # count ok, time short
        (5, 300.0, 4, 600.0, False),  # time ok, count short
        (5, 300.0, 5, 300.0, True),  # both satisfied
    ],
)
def test_inactivity_qualifies_truth_table(polls, timeout_s, count, elapsed_s, expected):
    cfg = LogEventConfig(
        name="inactive",
        pattern_type="inactivity",
        inactivity_polls=polls,
        inactivity_timeout_s=timeout_s,
    )
    event = LogEvent(cfg)
    assert event.inactivity_qualifies(count=count, elapsed_s=elapsed_s) is expected


def test_event_record_roundtrips_through_asdict():
    """The streak is persisted as asdict(EventRecord) and rebuilt with
    EventRecord(**stored); count/timestamps must survive that round-trip."""
    rec = EventRecord(
        event_id="e",
        name="inactive",
        source="log",
        count=1,
        first_seen_ts=100.0,
        last_seen_ts=100.0,
        payload={"inactive": True},
    )
    stored = asdict(rec)
    rebuilt = EventRecord(**stored)
    assert rebuilt.count == 1
    assert rebuilt.first_seen_ts == 100.0
    rebuilt.touch()
    assert rebuilt.count == 2


# --------------------------------------------------------------------------- #
# End-to-end tests through MonitorLoop with real local jobs
# --------------------------------------------------------------------------- #


class FakeClock:
    """Minimal stand-in for the ``time`` module used by the monitor loop.

    Only ``.time()`` is consumed by ``loop.py``; advancing it lets us simulate
    elapsed wall-clock time for inactivity timeouts deterministically.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def time(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


@pytest.fixture
def client():
    c = LocalCommandClient()
    try:
        yield c
    finally:
        c.cleanup()


def _make_job(
    tmp_path: Path,
    *,
    inactivity_polls: int = 1,
    inactivity_timeout_s: float = 0.0,
) -> tuple[JobRecord, Path]:
    """Build a local sleep job whose only log event restarts on inactivity."""
    log_path = tmp_path / "train.log"
    definition = LocalJobConfig(
        name="sleeper",
        command=["sleep", "120"],
        log_path=str(log_path),  # no %j/%a/%t -> loop and client resolve identically
        log_events=[
            LogEventConfig(
                name="inactive",
                pattern_type="inactivity",
                inactivity_polls=inactivity_polls,
                inactivity_timeout_s=inactivity_timeout_s,
                action=RestartActionConfig(reason="inactive log"),
            )
        ],
    )
    record = JobRecord(job_id="job", definition=definition, runtime=JobRuntime())
    return record, log_path


def _reload(store: JobFileStore) -> JobRecord:
    jobs = store.load_all()
    assert len(jobs) == 1
    return jobs[0]


def _streak_count(job: JobRecord) -> int:
    if not job.runtime.events:
        return 0
    (entry,) = job.runtime.events.values()
    return entry["count"]


def test_restart_after_n_consecutive_inactive_polls(tmp_path, client):
    """Polls=3: streak accumulates one per poll and restarts on the 3rd."""
    store = JobFileStore(tmp_path / "state")
    record, _ = _make_job(tmp_path, inactivity_polls=3)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    # Poll 1: starts the sleep job (no log processing on the submit poll).
    monitor.observe_once()
    job = _reload(store)
    assert job.runtime.attempts == 1
    first_job_id = job.runtime.runtime_job_id
    assert first_job_id is not None
    assert _streak_count(job) == 0

    # Polls 2 & 3: two inactive polls -> count 1, then 2, no restart yet.
    monitor.observe_once()
    assert _streak_count(_reload(store)) == 1
    monitor.observe_once()
    job = _reload(store)
    assert _streak_count(job) == 2
    assert job.runtime.attempts == 1  # still the original run

    # Poll 4: count reaches 3 -> RestartAction fires.
    monitor.observe_once()
    job = _reload(store)
    assert job.runtime.attempts == 2  # restarted
    assert job.runtime.runtime_job_id != first_job_id  # resubmitted as a new job
    assert job.runtime.events == {}  # streak cleared on restart
    # Old job was cancelled and removed from the client.
    assert first_job_id not in client.squeue()


def test_activity_resets_the_inactivity_streak(tmp_path, client):
    """New log output mid-streak clears the count, delaying the restart."""
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, inactivity_polls=3)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # poll 1: submit
    monitor.observe_once()  # poll 2: count 1
    monitor.observe_once()  # poll 3: count 2
    assert _streak_count(_reload(store)) == 2

    # Emit log output before poll 4 -> activity resets the streak.
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("still training...\n")
    monitor.observe_once()  # poll 4: activity -> streak cleared
    job = _reload(store)
    assert job.runtime.events == {}
    assert job.runtime.attempts == 1  # no restart

    # Streak must re-accumulate from scratch: counts 1, 2, then 3 -> restart.
    monitor.observe_once()  # count 1
    monitor.observe_once()  # count 2
    assert _reload(store).runtime.attempts == 1
    monitor.observe_once()  # count 3 -> restart
    assert _reload(store).runtime.attempts == 2


def test_restart_after_inactivity_timeout(tmp_path, client, monkeypatch):
    """Timeout=300s with a 1-poll floor fires once real elapsed time crosses
    the threshold, regardless of how many polls that took."""
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)

    store = JobFileStore(tmp_path / "state")
    record, _ = _make_job(tmp_path, inactivity_polls=1, inactivity_timeout_s=300.0)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # poll 1: submit (anchors nothing yet)
    monitor.observe_once()  # poll 2: streak starts, elapsed 0
    assert _reload(store).runtime.attempts == 1

    clock.advance(150.0)
    monitor.observe_once()  # elapsed 150 < 300
    assert _reload(store).runtime.attempts == 1

    clock.advance(150.0)
    monitor.observe_once()  # elapsed 300 >= 300 -> restart
    assert _reload(store).runtime.attempts == 2


def test_and_semantics_time_floor_gates_poll_count(tmp_path, client, monkeypatch):
    """Polls=5 AND timeout=300s: hitting 5 polls is not enough while the real
    elapsed time is still under 300s."""
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)

    store = JobFileStore(tmp_path / "state")
    record, _ = _make_job(tmp_path, inactivity_polls=5, inactivity_timeout_s=300.0)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # poll 1: submit
    # Six inactive polls, advancing 60s each -> count reaches 5 at the 6th,
    # but elapsed is only 240s (< 300) so it must NOT restart yet.
    for _ in range(5):
        monitor.observe_once()
        clock.advance(60.0)
    job = _reload(store)
    assert _streak_count(job) == 5
    assert job.runtime.attempts == 1  # count satisfied, time floor not yet

    # One more inactive poll: elapsed crosses 300s -> restart.
    monitor.observe_once()
    assert _reload(store).runtime.attempts == 2
