"""Tests for PER-EVENT restart budgets (``MaxActionFiresCondition``).

The motivating requirement: different events need different restart allowances.
A benign wall-clock rollover should be allowed to fire ~150 times (a 15 TT
schedule needs ~91 healthy segments); a "no training step" stall should be
allowed only ~3 before a human is asked.

``MaxAttemptsCondition`` cannot express that, because it reads the JOB-WIDE
``runtime.attempts`` — every event shares one counter, so on a chained run the
healthy rollovers spend the stall event's budget and silently disable it. These
tests pin the difference down.

See ``MaxActionFiresCondition`` / ``MonitorLoop._update_action_state`` /
``MonitorLoop._evaluate_event_condition``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oellm_autoexp.monitor.conditions import (
    ConditionContext,
    MaxActionFiresCondition,
    MaxActionFiresConditionConfig,
    MaxAttemptsCondition,
    MaxAttemptsConditionConfig,
)
from oellm_autoexp.monitor.actions import LogEventConfig, RestartActionConfig
from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime, MonitorLoop
from oellm_autoexp.monitor.local_client import LocalCommandClient
from oellm_autoexp.monitor.submission import LocalJobConfig


# --------------------------------------------------------------------------- #
# Unit: the condition itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fires, limit, expected",
    [
        (0, 3, True),  # never fired
        (2, 3, True),  # one left
        (3, 3, False),  # exactly spent
        (9, 3, False),  # over (state restored from an older, looser config)
        (0, 0, False),  # a zero budget disables the event outright
    ],
)
def test_max_action_fires_truth_table(fires, limit, expected):
    cond = MaxActionFiresCondition(MaxActionFiresConditionConfig(max_fires=limit))
    ctx = ConditionContext(extra={"action_fires": fires})
    assert bool(cond.check(ctx)) is expected


def test_missing_counter_passes_so_non_event_conditions_do_not_hard_fail():
    """Start/cancel/finish conditions carry no action, hence no counter.

    Fail-open matches CooldownCondition's behaviour when there is no
    event, and keeps a misplaced budget from silently blocking a job's
    start condition.
    """
    cond = MaxActionFiresCondition(MaxActionFiresConditionConfig(max_fires=1))
    assert bool(cond.check(ConditionContext())) is True


def test_the_two_budget_conditions_read_different_counters():
    """The regression this whole condition exists to prevent.

    Same situation — an event that has never fired, on a job that has
    already restarted 5 times for unrelated reasons. MaxAttempts is
    spent; MaxActionFires still has its full budget.
    """
    ctx = ConditionContext(attempts=5, extra={"action_fires": 0})
    assert (
        bool(MaxAttemptsCondition(MaxAttemptsConditionConfig(max_attempts=3)).check(ctx)) is False
    )
    assert (
        bool(MaxActionFiresCondition(MaxActionFiresConditionConfig(max_fires=3)).check(ctx)) is True
    )


# --------------------------------------------------------------------------- #
# End-to-end through MonitorLoop
# --------------------------------------------------------------------------- #


@pytest.fixture
def client():
    c = LocalCommandClient()
    try:
        yield c
    finally:
        c.cleanup()


def _make_job(tmp_path: Path, *, strict_budget: int, benign_budget: int):
    """A local job with two restart events on DIFFERENT budgets.

    The job writes nothing itself; the test drives the log so exactly
    one marker is visible per poll and it is unambiguous which event
    fired.
    """
    log_path = tmp_path / "train.log"
    log_path.write_text("", encoding="utf-8")
    definition = LocalJobConfig(
        name="trainer",
        command=["sleep", "120"],
        log_path=str(log_path),
        log_events=[
            LogEventConfig(
                name="strict",
                pattern_type="substring",
                pattern="STALL",
                condition=MaxActionFiresConditionConfig(max_fires=strict_budget),
                action=RestartActionConfig(reason="stall"),
            ),
            LogEventConfig(
                name="benign",
                pattern_type="substring",
                pattern="ROLLOVER",
                condition=MaxActionFiresConditionConfig(max_fires=benign_budget),
                action=RestartActionConfig(reason="rollover"),
            ),
        ],
    )
    return JobRecord(job_id="job", definition=definition, runtime=JobRuntime()), log_path


def _reload(store: JobFileStore) -> JobRecord:
    (job,) = store.load_all()
    return job


def _fires(job: JobRecord, action_id: str) -> int:
    return job.runtime.action_state.get(action_id, {}).get("fire_count", 0)


def _cycle(monitor, store, log_path, marker) -> JobRecord:
    """One poll with `marker` newly visible in the log.

    APPENDS rather than overwrites, which is what a real log does and
    what the cursor arithmetic requires. On a poll that restarts, the
    client truncates the file and the loop resets log_cursor to 0, so
    append still leaves exactly one unread marker. On a poll that does
    NOT restart (a spent budget), the cursor stays at end-of-file —
    overwriting there would leave the cursor past the new content and
    the next marker would be silently invisible.
    """
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{marker}\n")
    monitor.observe_once()
    return _reload(store)


def test_budgets_are_independent_per_event(tmp_path, client):
    """The requirement: a tight budget on one event must not be spent by another
    event's restarts, and must not spend theirs."""
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, strict_budget=2, benign_budget=5)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()  # submit
    assert _reload(store).runtime.attempts == 1

    # Spend the BENIGN budget four times over.
    for expected in (1, 2, 3, 4):
        job = _cycle(monitor, store, log_path, "ROLLOVER")
        assert _fires(job, "log:benign:1") == expected

    # Four restarts have happened, so a job-wide MaxAttempts(2) would already be
    # exhausted. The strict event has not fired once, so its budget is intact.
    assert job.runtime.attempts == 5
    assert _fires(job, "log:strict:0") == 0

    job = _cycle(monitor, store, log_path, "STALL")
    assert _fires(job, "log:strict:0") == 1
    job = _cycle(monitor, store, log_path, "STALL")
    assert _fires(job, "log:strict:0") == 2

    # Strict is now spent and must stop acting...
    before = job.runtime.attempts
    job = _cycle(monitor, store, log_path, "STALL")
    assert _fires(job, "log:strict:0") == 2, "strict fired past its budget"
    assert job.runtime.attempts == before, "a spent budget must not restart the job"

    # ...while the benign event, on its own counter, still works.
    job = _cycle(monitor, store, log_path, "ROLLOVER")
    assert _fires(job, "log:benign:1") == 5
    assert job.runtime.attempts == before + 1


def test_fire_count_survives_restarts(tmp_path, client):
    """The budget is only meaningful if it is not reset by the restart it
    causes.

    _restart_job deliberately preserves action_state; this pins that
    contract.
    """
    store = JobFileStore(tmp_path / "state")
    record, log_path = _make_job(tmp_path, strict_budget=3, benign_budget=3)
    store.upsert(record)
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    monitor.observe_once()
    ids = []
    for expected in (1, 2, 3):
        job = _cycle(monitor, store, log_path, "STALL")
        assert _fires(job, "log:strict:0") == expected
        ids.append(job.runtime.runtime_job_id)

    assert len(set(ids)) == len(ids), "each restart should be a genuinely new job"

    before = job.runtime.attempts
    job = _cycle(monitor, store, log_path, "STALL")
    assert _fires(job, "log:strict:0") == 3
    assert job.runtime.attempts == before
