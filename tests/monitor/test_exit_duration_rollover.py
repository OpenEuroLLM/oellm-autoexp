"""Tests for the ``exit_duration_rollover`` log event.

`exit_duration_in_mins` ends a Megatron segment with ``sys.exit(0)``, so SLURM
reports COMPLETED and the monitor's terminal branch (``loop.py:236``) marks the
run "finished" even though the schedule is barely started — job 1487924 stopped
at iteration 19183/894000 that way. The `exit_duration_rollover` event in
``config/job/auto_restart{,_ckptreset}.yaml`` turns that clean exit back into a
restart, which is the chain-free alternative to ``job.chain_repeat``.

What makes the pattern safe is that Megatron's two exits come from mutually
exclusive code paths in ``megatron/training/training.py``:

  * the duration branch prints ``exiting program after <N> minutes`` (:2070)
    from INSIDE the training loop and leaves via ``sys.exit`` (:2651), so
    ``after training is done`` is never reached;
  * a genuinely finished schedule leaves the loop normally, ``train()``
    RETURNS, and :781 prints ``after training is done``.

So the line can only ever be emitted mid-schedule, and `finished_training`
stays the sole authority on real completion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from oellm_autoexp.monitor.actions import (
    FinishActionConfig,
    LogEvent,
    LogEventConfig,
    RestartActionConfig,
)
from oellm_autoexp.monitor.conditions import MaxAttemptsConditionConfig
from oellm_autoexp.monitor.local_client import LocalCommandClient
from oellm_autoexp.monitor.loop import (
    JobFileStore,
    JobRecord,
    JobRuntime,
    MonitorLoop,
)
from oellm_autoexp.monitor.submission import LocalJobConfig

ROLLOVER_PATTERN = "exiting program after [0-9.]+ minutes"

# Verbatim from slurm-1487924.log (512 nodes, 2026-08-25).
REAL_EXIT_LINE = (
    "[default0]:[exiting program after 690.0159624854724 minutes] datetime: 2026-08-25 19:54:10 "
)
# training.py:781 — printed only when train() returns, i.e. schedule complete.
REAL_DONE_LINE = "[default0]:[after training is done] datetime: 2026-08-25 19:54:10 "


# --------------------------------------------------------------------------- #
# Pattern discrimination
# --------------------------------------------------------------------------- #


def test_pattern_matches_the_real_exit_duration_line():
    assert re.search(ROLLOVER_PATTERN, REAL_EXIT_LINE)


@pytest.mark.parametrize(
    "line",
    [
        # Genuine end of schedule -> finished_training must win, not a restart.
        REAL_DONE_LINE,
        # training.py:2086, exit_interval -> a deliberately bounded run.
        "[default0]:[exiting program at iteration 6000] datetime: 2026-08-25 19:54:10",
        # training.py:2015 -> indistinguishable from a manual scancel.
        "[default0]:[exiting program after receiving SIGTERM.] datetime: 2026-08-25 19:54:10",
    ],
)
def test_pattern_ignores_the_sibling_exit_messages(line):
    assert re.search(ROLLOVER_PATTERN, line) is None


def test_log_event_triggers_on_the_real_line():
    cfg = LogEventConfig(
        name="exit_duration_rollover",
        pattern=ROLLOVER_PATTERN,
        pattern_type="regex",
        action=RestartActionConfig(reason="rollover"),
    )
    assert LogEvent(cfg).check_triggers(REAL_EXIT_LINE)
    assert LogEvent(cfg).check_triggers(REAL_DONE_LINE) == []


# --------------------------------------------------------------------------- #
# Shipped job policies actually carry the event, in the right order
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("policy", ["auto_restart", "auto_restart_ckptreset"])
def test_policy_has_capped_rollover_after_the_finish_events(policy):
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="autoexp", overrides=[f"job={policy}"])

    names = [e["name"] for e in cfg.job.log_events]
    assert "exit_duration_rollover" in names

    rollover = names.index("exit_duration_rollover")
    # Must sit AFTER the finish events: _process_log_events returns on the first
    # terminal effect, so a genuinely finished run can never reach the restart.
    assert rollover > names.index("finished_training")
    assert rollover > names.index("finish")
    # ...and BEFORE the generic "Exited with exit code 1" restart, so the reason
    # recorded for a rollover is the specific one.
    assert rollover < names.index("error")

    event = cfg.job.log_events[rollover]
    assert event["action"]["class_name"] == "RestartAction"
    # RestartAction has no cap of its own and this event fires on every HEALTHY
    # segment, so the ceiling is mandatory rather than defensive.
    #
    # It must be the PER-EVENT budget, not the job-wide MaxAttemptsCondition.
    # With a shared counter, this benign event's ~91 healthy rollovers would
    # spend the tight budget the stall events depend on and silently disable
    # them a few segments in.
    assert event["condition"]["class_name"] == "MaxActionFiresCondition"
    # A 15 TT schedule needs ~91 rollovers, so the ceiling must clear that with
    # room for the extra resubmissions error restarts add.
    assert event["condition"]["max_fires"] >= 100


@pytest.mark.parametrize("policy", ["auto_restart", "auto_restart_ckptreset"])
def test_stall_events_are_tightly_budgeted_and_end_in_a_cancel(policy):
    """The rollover is generous; a stall must not be — and when the stall
    budget runs out something must still stop the job, or the blind spot
    returns."""
    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="autoexp", overrides=[f"job={policy}"])

    events = {e["name"]: e for e in cfg.job.log_events}
    names = [e["name"] for e in cfg.job.log_events]

    for stall in ("stalled_iterations", "stalled_high_water"):
        cond = events[stall]["condition"]
        assert cond["class_name"] == "MaxActionFiresCondition"
        assert cond["max_fires"] <= 5, "a stall is a real fault; retry it a few times, not 150"
        assert cond["max_fires"] < events["exit_duration_rollover"]["condition"]["max_fires"]

    # A budget makes an event stop ACTING, not the job stop RUNNING. Without a
    # terminal case the run would sit unwatched exactly as job 1512329 did.
    giveup = events["stalled_giveup"]
    assert giveup["action"]["class_name"] == "CancelAction"
    assert giveup.get("condition") is None, "the giveup must not itself be budgeted"
    # Ordered last, so while either stall event still has budget it restarts first.
    assert names.index("stalled_giveup") > names.index("stalled_iterations")
    assert names.index("stalled_giveup") > names.index("stalled_high_water")


# --------------------------------------------------------------------------- #
# End-to-end through MonitorLoop with real local jobs
# --------------------------------------------------------------------------- #


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
    line: str,
    max_attempts: int = 150,
) -> JobRecord:
    """A local job that prints ``line`` and exits 0 — the shape of a Megatron
    segment that hit its duration budget (exit code 0, so a plain
    COMPLETED)."""
    definition = LocalJobConfig(
        name="segment",
        command=["python3", "-c", f"print({line!r})"],
        log_path=str(tmp_path / "train.log"),  # no %j/%a/%t -> resolves identically
        log_events=[
            LogEventConfig(
                name="finished_training",
                pattern="[after training is done]",
                pattern_type="substring",
                action=FinishActionConfig(reason="Training finished"),
            ),
            LogEventConfig(
                name="exit_duration_rollover",
                pattern=ROLLOVER_PATTERN,
                pattern_type="regex",
                condition=MaxAttemptsConditionConfig(max_attempts=max_attempts),
                action=RestartActionConfig(reason="exit_duration_in_mins reached"),
            ),
        ],
    )
    return JobRecord(job_id="job", definition=definition, runtime=JobRuntime())


def _reload(store: JobFileStore) -> JobRecord:
    jobs = store.load_all(include_finished=True)
    assert len(jobs) == 1
    return jobs[0]


def _run(tmp_path, client, *, line: str, polls: int, max_attempts: int = 150) -> JobRecord:
    store = JobFileStore(tmp_path / "state")
    store.upsert(_make_job(tmp_path, line=line, max_attempts=max_attempts))
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)
    for _ in range(polls):
        monitor.observe_once()
    return _reload(store)


def test_duration_exit_restarts_instead_of_finishing(tmp_path, client):
    """The regression: exit code 0 + the duration line must resubmit, not end
    the run."""
    job = _run(tmp_path, client, line=REAL_EXIT_LINE, polls=3)
    assert job.runtime.attempts > 1, "duration exit did not resubmit the segment"
    assert job.runtime.final_state is None, "run was marked terminal despite work remaining"


def test_genuine_completion_still_finishes(tmp_path, client):
    """The guard: 'after training is done' must still end the run, otherwise the
    rollover event would relaunch a completed schedule forever."""
    job = _run(tmp_path, client, line=REAL_DONE_LINE, polls=3)
    assert job.runtime.final_state == "finished"
    assert job.runtime.attempts == 1


def test_max_attempts_stops_the_rollover_loop(tmp_path, client):
    """Without the cap this event would relaunch every healthy segment forever;
    at the ceiling the job falls through to the normal COMPLETED handling."""
    job = _run(tmp_path, client, line=REAL_EXIT_LINE, polls=6, max_attempts=2)
    assert job.runtime.attempts == 2
    assert job.runtime.final_state == "finished"
