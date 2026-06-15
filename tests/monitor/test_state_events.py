"""Tests for state-transition events driving the job lifecycle.

These cover the ``state_events`` path in :meth:`MonitorLoop._status_action`:
a configured action (e.g. ``FinishAction`` on a ``RUNNING -> TIMEOUT``
transition, as used by ``config/job/auto_cancel.yaml``) must actually finish
and remove the job from monitoring, not merely record an "informational" event.
The dependency chain handles the restart, so the monitor must let go of a
timed-out job instead of polling it forever.
"""

from __future__ import annotations

from pathlib import Path

from oellm_autoexp.monitor.actions import (
    CancelActionConfig,
    FinishActionConfig,
    StateEventConfig,
)
from oellm_autoexp.monitor.loop import (
    JobFileStore,
    JobRecord,
    JobRuntime,
    MonitorLoop,
)
from oellm_autoexp.monitor.submission import LocalJobConfig


class FakeClient:
    """Minimal JobClientProtocol stand-in with a scriptable queue.

    ``squeue`` returns whatever ``statuses`` currently holds; ``remove`` and
    ``cancel`` record the job ids they were called with so tests can assert the
    loop released the job.
    """

    def __init__(self) -> None:
        self.statuses: dict[str, str] = {}
        self.removed: list[str] = []
        self.cancelled: list[str] = []

    def squeue(self) -> dict[str, str]:
        return dict(self.statuses)

    def remove(self, job_id: str) -> None:
        self.removed.append(job_id)

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


def _submitted_job(
    tmp_path: Path,
    *,
    state_events: list[StateEventConfig],
    job_id: str = "job",
    runtime_id: str = "1001",
) -> JobRecord:
    """An already-submitted local job (no start on the first poll)."""
    definition = LocalJobConfig(
        name="job",
        command=["true"],
        log_path=str(tmp_path / "train.log"),  # never created -> log events are no-ops
        state_events=state_events,
    )
    runtime = JobRuntime(
        submitted=True,
        runtime_job_id=runtime_id,
        attempts=1,
        last_status="RUNNING",
    )
    return JobRecord(job_id=job_id, definition=definition, runtime=runtime)


def _active(store: JobFileStore) -> list[JobRecord]:
    return store.load_all()


def test_timeout_state_event_finishes_and_removes_job(tmp_path):
    """RUNNING -> TIMEOUT with a FinishAction marks the job finished, releases
    it on the client, and drops it from the active poll set."""
    store = JobFileStore(tmp_path / "state")
    record = _submitted_job(
        tmp_path,
        state_events=[
            StateEventConfig(
                name="timeout",
                transition=("RUNNING", "TIMEOUT"),
                action=FinishActionConfig(reason="Job timed out"),
            )
        ],
    )
    store.upsert(record)

    client = FakeClient()
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    # Still RUNNING: no transition, job stays active.
    client.statuses = {"1001": "RUNNING"}
    monitor.observe_once()
    assert len(_active(store)) == 1

    # Transition to TIMEOUT: FinishAction fires -> job finished + removed.
    client.statuses = {"1001": "TIMEOUT"}
    monitor.observe_once()

    assert _active(store) == []  # no longer monitored
    finished = store.load_all(include_finished=True)
    assert len(finished) == 1
    assert finished[0].runtime.final_state == "finished"
    assert client.removed == ["1001"]
    assert client.cancelled == []  # finish must not scancel


def test_timeout_without_state_event_keeps_job_active(tmp_path):
    """Guard: TIMEOUT is not a natural-completion terminal state, so without a
    configured handler the job remains monitored (the bug's mirror image)."""
    store = JobFileStore(tmp_path / "state")
    store.upsert(_submitted_job(tmp_path, state_events=[]))

    client = FakeClient()
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    client.statuses = {"1001": "TIMEOUT"}
    monitor.observe_once()

    assert len(_active(store)) == 1
    assert client.removed == []


def test_failed_state_event_finishes_job(tmp_path):
    """The auto_cancel config also finishes on RUNNING -> FAILED."""
    store = JobFileStore(tmp_path / "state")
    store.upsert(
        _submitted_job(
            tmp_path,
            state_events=[
                StateEventConfig(
                    name="failed",
                    transition=("RUNNING", "FAILED"),
                    action=FinishActionConfig(reason="Job failed"),
                )
            ],
        )
    )

    client = FakeClient()
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    client.statuses = {"1001": "FAILED"}
    monitor.observe_once()

    assert _active(store) == []
    assert store.load_all(include_finished=True)[0].runtime.final_state == "finished"
    assert client.removed == ["1001"]


def test_cancel_state_event_cancels_job(tmp_path):
    """A CancelAction on a state transition cancels (scancel) and removes."""
    store = JobFileStore(tmp_path / "state")
    store.upsert(
        _submitted_job(
            tmp_path,
            state_events=[
                StateEventConfig(
                    name="preempted",
                    transition=("RUNNING", "PREEMPTED"),
                    action=CancelActionConfig(reason="preempted"),
                )
            ],
        )
    )

    client = FakeClient()
    monitor = MonitorLoop(store, local_client=client, show_poll_state=False, no_error_catching=True)

    client.statuses = {"1001": "PREEMPTED"}
    monitor.observe_once()

    assert _active(store) == []
    assert store.load_all(include_finished=True)[0].runtime.final_state == "cancelled"
    assert client.cancelled == ["1001"]
    assert client.removed == ["1001"]
