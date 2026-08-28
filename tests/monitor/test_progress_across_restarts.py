"""A ``progress_mode: furthest`` streak must survive a restart; its CLOCK must
not.

Without this, a restart LOOP is invisible to the progress events. Each cycle is
a fresh SLURM job, ``_restart_job`` wipes ``runtime.events``, and the streak
never reaches its window -- so the only bound left is the blunt
per-event restart budget. Keeping the furthest iteration reached makes "the job keeps
relaunching and never gets back past the iteration it already reached"
detectable, which is precisely what ``progress_mode: furthest`` means.

The subtlety is the clock. A restart is followed by an unbounded queue wait, and
``_process_log_events`` returns early while the new job's log does not exist, so
no poll touches the record in between. Anchoring ``first_seen_ts`` at restart
time would fold that entire wait into ``elapsed_s`` and silently reduce the AND
in ``progress_qualifies`` to its poll-count half. The record is therefore flagged
for re-anchoring and gets its clock from the first poll that sees a log.

The queue is simulated by UNLINKING the local job's log: the monitor gates purely
on ``log_path.exists()``, which is exactly the property that makes a PENDING
SLURM job invisible to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oellm_autoexp.monitor import loop as loop_mod
from oellm_autoexp.monitor.actions import LogEventConfig, RestartActionConfig
from oellm_autoexp.monitor.local_client import LocalCommandClient
from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime, MonitorLoop
from oellm_autoexp.monitor.submission import LocalJobConfig

ITER_PATTERN = r"iteration +([0-9]+)/"
POLL = 60.0


class FakeClock:
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


def _iter_line(n: int) -> str:
    return f"[default0]: iteration {n:>8}/  894000 | elapsed time per iteration (ms): 4131.0 |\n"


def _make_job(tmp_path: Path, *, polls: int = 3, timeout_s: float = 180.0):
    """A local job with an independent restart trigger plus the furthest-mode
    guard.

    The trigger models "some infra fault restarted the job" -- the Shape B
    scenario -- so the guard under test is not also the thing causing the
    restarts.
    """
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    definition = LocalJobConfig(
        name="trainer",
        command=["sleep", "600"],
        log_path=str(log_path),
        log_events=[
            LogEventConfig(
                name="infra_fault",
                pattern_type="substring",
                pattern="RESTARTME",
                action=RestartActionConfig(reason="simulated infra fault"),
            ),
            LogEventConfig(
                name="stuck_below_furthest_iteration",
                pattern_type="progress",
                pattern=ITER_PATTERN,
                progress_mode="furthest",
                progress_polls=polls,
                progress_timeout_s=timeout_s,
                action=RestartActionConfig(reason="no net progress"),
            ),
        ],
    )
    return JobRecord(job_id="job", definition=definition, runtime=JobRuntime()), log_path


def _reload(store: JobFileStore) -> JobRecord:
    (job,) = store.load_all()
    return job


def _record(job: JobRecord) -> dict | None:
    """The single progress streak record, if one is being tracked."""
    for entry in job.runtime.events.values():
        if "furthest_value" in (entry.get("payload") or {}):
            return entry
    return None


def _append(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def _reach_furthest_iteration(monitor, store, log_path, clock, mark: int):
    """Submit, train up to `mark`, then take one infra restart."""
    monitor.observe_once()  # submit
    _append(log_path, _iter_line(mark))
    clock.advance(POLL)
    monitor.observe_once()  # first sighting -> furthest reached = mark
    assert _record(_reload(store))["payload"]["furthest_value"] == mark

    _append(log_path, "RESTARTME\n")
    clock.advance(POLL)
    monitor.observe_once()  # infra_fault -> restart
    job = _reload(store)
    assert job.runtime.attempts == 2
    return job


def _simulate_queue(monitor, store, log_path, clock, *, hours: float, polls: int = 5):
    """No log on disk == a PENDING job, as far as the monitor is concerned."""
    log_path.unlink()
    for _ in range(polls):
        clock.advance(hours * 3600.0 / polls)
        monitor.observe_once()
    return _reload(store)


def test_furthest_iteration_survives_a_restart_but_the_clock_does_not(
    tmp_path, client, monkeypatch
):
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    _reach_furthest_iteration(monitor, store, log_path, clock, mark=100)

    # The mark is carried; the streak is flagged and zeroed.
    carried = _record(_reload(store))
    assert carried is not None, "the furthest iteration reached must survive the restart"
    assert carried["payload"]["furthest_value"] == 100
    assert carried["payload"].get("clock_paused_for_requeue") is True
    assert carried["count"] == 0

    # Three hours of "queue". No log -> no poll touches the record, nothing fires.
    job = _simulate_queue(monitor, store, log_path, clock, hours=3)
    assert job.runtime.attempts == 2, "a queue wait must not look like a stall"
    assert _record(job)["count"] == 0
    assert _record(job)["payload"].get("clock_paused_for_requeue") is True

    # The job finally starts and resumes BELOW the mark (checkpoint rewind).
    log_path.write_text(_iter_line(50), encoding="utf-8")
    clock.advance(POLL)
    monitor.observe_once()
    job = _reload(store)

    rec = _record(job)
    assert rec["payload"].get("clock_paused_for_requeue") is None, "flag consumed"
    assert rec["payload"]["furthest_value"] == 100, "rewind must not lower the mark"
    assert rec["count"] == 1, "one poll of no progress counted, not three hours of queue"
    # Nothing had been counted before the restart (it had just reached new ground),
    # so the restarted clock begins at zero -- the queue contributed NOTHING.
    assert rec["last_seen_ts"] - rec["first_seen_ts"] == 0.0
    assert job.runtime.attempts == 2


def test_no_progress_time_already_counted_carries_across_a_restart(tmp_path, client, monkeypatch):
    """The queue is dropped, but time the job spent RUNNING without net
    progress is not.

    This is what makes a fast restart loop detectable: without it every cycle
    would reset the streak to zero and a ~7 min cycle could never reach the
    40-poll window the shipped config uses.
    """
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    _reach_furthest_iteration(monitor, store, log_path, clock, mark=100)
    _simulate_queue(monitor, store, log_path, clock, hours=2)

    # Two polls below it -> 2 polls / 60 s counted.
    log_path.write_text("", encoding="utf-8")
    for n in (50, 60):
        _append(log_path, _iter_line(n))
        clock.advance(POLL)
        monitor.observe_once()
    rec = _record(_reload(store))
    assert rec["count"] == 2
    assert rec["last_seen_ts"] - rec["first_seen_ts"] == 60.0

    # Another infra restart, and another long queue.
    _append(log_path, "RESTARTME\n")
    clock.advance(POLL)
    monitor.observe_once()
    assert _reload(store).runtime.attempts == 3
    _simulate_queue(monitor, store, log_path, clock, hours=4)

    # The first poll of the new attempt resumes the clock from the counted 60 s,
    # NOT from zero and NOT from six hours of queue.
    log_path.write_text(_iter_line(55), encoding="utf-8")
    clock.advance(POLL)
    monitor.observe_once()
    rec = _record(_reload(store))
    assert rec["count"] == 3, "poll count continues across the restart"
    assert rec["last_seen_ts"] - rec["first_seen_ts"] == 60.0, "counted time resumes; queue dropped"


def test_healthy_rollover_with_a_long_queue_never_trips_the_guard(tmp_path, client, monkeypatch):
    """Restart -> 3 h queue -> resume -> climb back past the mark. Must not
    fire.

    This is the shape of every healthy wall-clock rollover, so a false
    positive here would restart a perfectly good 512-node run.
    """
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, polls=3, timeout_s=180.0)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    _reach_furthest_iteration(monitor, store, log_path, clock, mark=100)
    _simulate_queue(monitor, store, log_path, clock, hours=3)

    # Resume below the mark and climb. Regaining ground takes a couple of polls,
    # which must stay under the window (3 polls / 180 s here).
    log_path.write_text("", encoding="utf-8")
    for n in (50, 75):
        _append(log_path, _iter_line(n))
        clock.advance(POLL)
        monitor.observe_once()
    assert _reload(store).runtime.attempts == 2, "climbing back is not a stall"

    # Past the old mark -> genuine net progress -> streak resets.
    _append(log_path, _iter_line(150))
    clock.advance(POLL)
    monitor.observe_once()
    job = _reload(store)
    assert job.runtime.attempts == 2
    rec = _record(job)
    assert rec["payload"]["furthest_value"] == 150
    assert rec["count"] == 0, "net progress resets the streak"

    # And it keeps not firing as the run continues.
    for n in (200, 250, 300, 350):
        _append(log_path, _iter_line(n))
        clock.advance(POLL)
        monitor.observe_once()
    assert _reload(store).runtime.attempts == 2


def test_a_restart_loop_that_never_regains_ground_is_caught(tmp_path, client, monkeypatch):
    """The Shape B case that was previously invisible.

    Each cycle relaunches, resumes below the mark and dies again without
    ever passing it. Before the mark was carried, every cycle wiped the
    streak and the guard could never accumulate its window.
    """
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, polls=3, timeout_s=180.0)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    _reach_furthest_iteration(monitor, store, log_path, clock, mark=100)
    _simulate_queue(monitor, store, log_path, clock, hours=3)

    # --- cycle A: runs briefly below the mark, then dies. Banks 2 polls / 60 s.
    log_path.write_text("", encoding="utf-8")
    for n in (50, 60):
        _append(log_path, _iter_line(n))
        clock.advance(POLL)
        monitor.observe_once()
    assert _reload(store).runtime.attempts == 2, "too short to trip on its own"

    _append(log_path, "RESTARTME\n")
    clock.advance(POLL)
    monitor.observe_once()
    assert _reload(store).runtime.attempts == 3
    _simulate_queue(monitor, store, log_path, clock, hours=3)

    # --- cycle B: resumes, still never gets past it. The carried-over count means
    # it trips on the THIRD poll here. A non-carrying implementation would be at
    # count=3 / elapsed=120 s at this point -- under the 3-poll AND 180 s gate --
    # and every subsequent cycle would reset it the same way, forever.
    log_path.write_text("", encoding="utf-8")
    for n, expect_attempts in ((55, 3), (58, 3), (59, 4)):
        _append(log_path, _iter_line(n))
        clock.advance(POLL)
        monitor.observe_once()
        assert _reload(store).runtime.attempts == expect_attempts

    job = _reload(store)
    assert job.runtime.attempts == 4, "cumulative no-net-progress window reached"
    # A firing progress event pops its OWN streak before restarting ("clear the
    # streak so it must re-accumulate before firing again"), so its mark is
    # deliberately not carried here -- it re-baselines on the next cycle, which
    # is what keeps it to at most one restart per window. Cross-cycle memory for
    # the give-up decision lives in the separate `stuck_below_furthest_iteration_cancel` record, whose
    # streak is never popped and therefore does carry.
    assert _record(job) is None


def test_any_change_mode_streaks_are_not_carried(tmp_path, client, monkeypatch):
    """Only the furthest-iteration record crosses a restart.

    progress_mode: any_change treats the backwards jump of a resume as movement on
    purpose, so carrying its last-seen value would be meaningless. Keeping it
    job-local also preserves the calibration of `iteration_counter_frozen`, which was
    measured against single-job logs.
    """
    clock = FakeClock(1000.0)
    monkeypatch.setattr(loop_mod, "time", clock)
    store = JobFileStore(tmp_path / "state")
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    definition = LocalJobConfig(
        name="trainer",
        command=["sleep", "600"],
        log_path=str(log_path),
        log_events=[
            LogEventConfig(
                name="infra_fault",
                pattern_type="substring",
                pattern="RESTARTME",
                action=RestartActionConfig(reason="simulated infra fault"),
            ),
            LogEventConfig(
                name="iteration_counter_frozen",
                pattern_type="progress",
                pattern=ITER_PATTERN,
                progress_mode="any_change",
                progress_polls=3,
                progress_timeout_s=180.0,
                action=RestartActionConfig(reason="stuck"),
            ),
        ],
    )
    store.upsert(JobRecord(job_id="job", definition=definition, runtime=JobRuntime()))
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()
    _append(log_path, _iter_line(100))
    clock.advance(POLL)
    monitor.observe_once()
    assert any(
        "last_value" in (e.get("payload") or {}) for e in _reload(store).runtime.events.values()
    )

    _append(log_path, "RESTARTME\n")
    clock.advance(POLL)
    monitor.observe_once()
    assert _reload(store).runtime.events == {}, "increase-mode streaks are dropped on restart"
