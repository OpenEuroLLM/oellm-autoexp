"""Tests for progress-stall detection in the monitor loop.

These exercise the ``pattern_type: "progress"`` log event, which asks whether a
counter in the log ADVANCED rather than whether the log grew. The distinction is
the point: a job stuck in an ft_launcher restart loop keeps writing fresh setup
banners, so ``pattern_type: "inactivity"`` never fires on it (job 1375720 grew a
251 MB log over 4 h while pinned at iteration 100).

The end-to-end cases use real *local* jobs and write the log by hand so the
counter is fully controlled.

See ``LogEvent.observe_progress`` / ``LogEvent.progress_qualifies`` /
``MonitorLoop._process_progress_event`` / ``loop._progress_advanced``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
    _progress_advanced,
)
from oellm_autoexp.monitor.local_client import LocalCommandClient
from oellm_autoexp.monitor.submission import LocalJobConfig


ITER_PATTERN = r"iteration +([0-9]+)/"


def _iter_line(n: int) -> str:
    """A trimmed Megatron iteration line, the real thing this parses."""
    return (
        f"[default3]: [2026-08-18 05:19:34] iteration {n:>8}/  894000 | "
        f"consumed samples: 12292096 | elapsed time per iteration (ms): 4425.1 | "
        f"lm loss: 2.135887E+00 |\n"
    )


# --------------------------------------------------------------------------- #
# Pure-logic unit tests
# --------------------------------------------------------------------------- #


def test_check_triggers_is_noop_for_progress():
    """Progress is streak based, not match-once-and-fire: check_triggers must
    not turn a matching line into an immediate action."""
    cfg = LogEventConfig(name="stalled", pattern_type="progress", pattern=ITER_PATTERN)
    event = LogEvent(cfg)
    assert event.check_triggers(_iter_line(42)) == []
    assert event.check_triggers("") == []


def test_progress_metadata_is_stable_and_marked():
    """The event key hashes this dict, so it must not carry the counter."""
    cfg = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=ITER_PATTERN, metadata={"k": "v"}
    )
    event = LogEvent(cfg)
    md = event.progress_metadata()
    assert md["progress"] is True
    assert md["k"] == "v"
    assert event.progress_metadata() == md
    assert "progress_last" not in md and "progress_raw" not in md


def test_observe_progress_returns_last_and_max():
    cfg = LogEventConfig(name="stalled", pattern_type="progress", pattern=ITER_PATTERN)
    event = LogEvent(cfg)
    text = _iter_line(3458) + _iter_line(3459) + _iter_line(3460)
    last, mx, raw = event.observe_progress(text)
    assert (last, mx, raw) == (3460.0, 3460.0, "3460")


def test_observe_progress_last_differs_from_max_after_a_resume():
    """A restart rewinds the counter: `last` is the current position (3001),
    `max` is still the high-water mark (3460)."""
    cfg = LogEventConfig(name="stalled", pattern_type="progress", pattern=ITER_PATTERN)
    event = LogEvent(cfg)
    text = _iter_line(3460) + "…ft_launcher restarted the worker group…\n" + _iter_line(3001)
    last, mx, _ = event.observe_progress(text)
    assert last == 3001.0
    assert mx == 3460.0


def test_observe_progress_with_no_match_is_the_no_movement_signal():
    cfg = LogEventConfig(name="stalled", pattern_type="progress", pattern=ITER_PATTERN)
    event = LogEvent(cfg)
    assert event.observe_progress("") == (None, None, None)
    assert event.observe_progress("building GPT model ...\n") == (None, None, None)


def test_observe_progress_ignores_unparseable_groups():
    """A group that matches but is not a number must not raise or poison the
    result — the other matches still count."""
    cfg = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=r"iteration +(\S+)/", progress_group=1
    )
    event = LogEvent(cfg)
    last, mx, raw = event.observe_progress("iteration NaNsense/\n" + _iter_line(7))
    assert (last, mx, raw) == (7.0, 7.0, "7")


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
def test_progress_qualifies_truth_table(polls, timeout_s, count, elapsed_s, expected):
    cfg = LogEventConfig(
        name="stalled",
        pattern_type="progress",
        pattern=ITER_PATTERN,
        progress_polls=polls,
        progress_timeout_s=timeout_s,
    )
    event = LogEvent(cfg)
    assert event.progress_qualifies(count=count, elapsed_s=elapsed_s) is expected


def _blank_record() -> EventRecord:
    return EventRecord(event_id="e", name="stalled", source="log", count=0, payload={})


@pytest.mark.parametrize("mode", ["increase", "max"])
def test_progress_advanced_first_sighting_counts_as_movement(mode):
    """The clock starts at the first iteration, not at job submission."""
    cfg = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=ITER_PATTERN, progress_mode=mode
    )
    rec = _blank_record()
    assert _progress_advanced(cfg, rec, 10.0, 10.0) is True


@pytest.mark.parametrize("mode", ["increase", "max"])
def test_progress_advanced_no_match_never_moves(mode):
    """A job that emits no counter at all never moves, in either mode — this is
    what catches a startup hang (jobs 1392564, 1392777: zero iterations)."""
    cfg = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=ITER_PATTERN, progress_mode=mode
    )
    rec = _blank_record()
    assert _progress_advanced(cfg, rec, None, None) is False
    assert _progress_advanced(cfg, rec, None, None) is False


def test_increase_mode_tolerates_a_resume_rewind():
    """3460 -> 3001 is a healthy ft_launcher restart, not a stall."""
    cfg = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=ITER_PATTERN, progress_mode="increase"
    )
    rec = _blank_record()
    _progress_advanced(cfg, rec, 3460.0, 3460.0)
    assert _progress_advanced(cfg, rec, 3001.0, 3460.0) is True
    assert rec.payload["progress_last"] == 3001.0


def test_increase_mode_flags_a_frozen_counter():
    cfg = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=ITER_PATTERN, progress_mode="increase"
    )
    rec = _blank_record()
    _progress_advanced(cfg, rec, 100.0, 100.0)
    assert _progress_advanced(cfg, rec, 100.0, 100.0) is False
    assert _progress_advanced(cfg, rec, 100.0, 100.0) is False


def test_max_mode_rejects_re_running_the_same_iterations():
    """The 1375720 shape: every cycle replays 1..100 and dies. `increase` sees
    movement, `max` does not — which is the whole reason the mode exists."""
    increase = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=ITER_PATTERN, progress_mode="increase"
    )
    mx = LogEventConfig(
        name="stalled", pattern_type="progress", pattern=ITER_PATTERN, progress_mode="max"
    )
    rec_i, rec_m = _blank_record(), _blank_record()

    _progress_advanced(increase, rec_i, 100.0, 100.0)
    _progress_advanced(mx, rec_m, 100.0, 100.0)
    # Next cycle replays 1..100.
    assert _progress_advanced(increase, rec_i, 100.0, 100.0) is False
    assert _progress_advanced(mx, rec_m, 100.0, 100.0) is False
    # Mid-cycle the position moves but the high-water mark does not.
    assert _progress_advanced(increase, rec_i, 50.0, 50.0) is True
    assert _progress_advanced(mx, rec_m, 50.0, 50.0) is False
    assert rec_m.payload["progress_max"] == 100.0

    # Genuine new ground resets both.
    assert _progress_advanced(mx, rec_m, 101.0, 101.0) is True


# --------------------------------------------------------------------------- #
# End-to-end tests through MonitorLoop with real local jobs
# --------------------------------------------------------------------------- #


class FakeClock:
    """Minimal stand-in for the ``time`` module used by the monitor loop."""

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
    progress_polls: int = 1,
    progress_timeout_s: float = 0.0,
    progress_mode: str = "increase",
) -> tuple[JobRecord, Path]:
    """A local sleep job whose only log event restarts on a progress stall.

    The job itself writes nothing; the test drives ``train.log`` directly so the
    counter is fully controlled.
    """
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    definition = LocalJobConfig(
        name="trainer",
        command=["sleep", "120"],
        log_path=str(log_path),  # no %j/%a/%t -> loop and client resolve identically
        log_events=[
            LogEventConfig(
                name="stalled",
                pattern_type="progress",
                pattern=ITER_PATTERN,
                progress_mode=progress_mode,
                progress_polls=progress_polls,
                progress_timeout_s=progress_timeout_s,
                action=RestartActionConfig(reason="no forward progress"),
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


def test_restart_after_n_polls_without_progress(tmp_path, client):
    """A log that keeps GROWING but never advances the counter still restarts —
    the case the inactivity check cannot see."""
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, progress_polls=3)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # poll 1: submit
    job = _reload(store)
    assert job.runtime.attempts == 1
    first_job_id = job.runtime.runtime_job_id

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(_iter_line(100))
    monitor.observe_once()  # poll 2: first sighting -> movement, streak 0
    assert _streak_count(_reload(store)) == 0

    # Three polls of fresh-but-useless output: the restart banner an
    # ft_launcher loop emits, with the counter pinned at 100.
    for _ in range(3):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("building GPT model ...\nWorker group UNHEALTHY; will restart\n")
        monitor.observe_once()

    job = _reload(store)
    assert job.runtime.attempts == 2  # restarted
    assert job.runtime.runtime_job_id != first_job_id
    assert job.runtime.events == {}  # streak cleared on restart
    assert first_job_id not in client.squeue()


def test_advancing_counter_resets_the_streak(tmp_path, client):
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, progress_polls=3)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # poll 1: submit
    monitor.observe_once()  # poll 2: no output at all -> streak 1
    monitor.observe_once()  # poll 3: streak 2
    assert _streak_count(_reload(store)) == 2

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(_iter_line(1))
    monitor.observe_once()  # poll 4: counter appears -> streak resets
    job = _reload(store)
    assert _streak_count(job) == 0
    assert job.runtime.attempts == 1

    # It must re-accumulate from scratch before firing.
    monitor.observe_once()
    monitor.observe_once()
    assert _reload(store).runtime.attempts == 1
    monitor.observe_once()
    assert _reload(store).runtime.attempts == 2


def test_startup_hang_with_no_iterations_ever(tmp_path, client):
    """Jobs 1392564 / 1392777: hundreds of MB of per-rank banners, zero
    iterations.

    Every poll has new text, so inactivity is silent.
    """
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, progress_polls=2)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # poll 1: submit
    for _ in range(2):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("WARNING:megatron.core.datasets.megatron_tokenizer: legacy tokenizer\n")
        monitor.observe_once()

    assert _reload(store).runtime.attempts == 2


def test_resume_rewind_does_not_trip_increase_mode(tmp_path, client):
    """A healthy in-job restart rewinds the counter to the last checkpoint; the
    default mode must treat that as progress, not as a stall."""
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, progress_polls=2, progress_mode="increase")
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # submit
    for value in (3459, 3460):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_iter_line(value))
        monitor.observe_once()

    # ft_launcher restarts the group; training resumes from checkpoint 3000.
    for value in (3001, 3002):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_iter_line(value))
        monitor.observe_once()

    job = _reload(store)
    assert job.runtime.attempts == 1  # never restarted by the monitor
    assert _streak_count(job) == 0


def test_max_mode_fires_on_a_net_zero_restart_loop(tmp_path, client):
    """Same rewind, but `max` mode: replaying already-done iterations is not
    progress, so a loop that never gets past its high-water mark is caught."""
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, progress_polls=2, progress_mode="max")
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # submit
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(_iter_line(3460))
    monitor.observe_once()  # high-water mark 3460

    for value in (3001, 3002):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(_iter_line(value))
        monitor.observe_once()

    assert _reload(store).runtime.attempts == 2


def test_progress_timeout_gates_the_poll_count(tmp_path, client, monkeypatch):
    """Polls=2 AND timeout=300s: reaching 2 stalled polls is not enough while
    real elapsed time is under 300s."""
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)

    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, progress_polls=2, progress_timeout_s=300.0)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # submit
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(_iter_line(1))
    monitor.observe_once()  # first sighting -> anchors the streak

    for _ in range(3):
        clock.advance(60.0)
        monitor.observe_once()
    job = _reload(store)
    assert _streak_count(job) == 3  # count satisfied
    assert job.runtime.attempts == 1  # ...but only 180s elapsed

    clock.advance(180.0)
    monitor.observe_once()  # elapsed 360 >= 300 -> restart
    assert _reload(store).runtime.attempts == 2
