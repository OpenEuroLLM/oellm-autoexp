"""RestartAction.wait_for_job_end: the resubmission waits for the runtime job
to leave the queue instead of cancelling it (a graceful exit prints 'exiting
program ...' before its async checkpoint write completes)."""

from __future__ import annotations

from types import SimpleNamespace

from oellm_autoexp.monitor.actions import (
    RestartAction,
    RestartActionConfig,
    ActionContext,
    EventRecord,
)
from oellm_autoexp.monitor.loop import JobRecord, JobRuntime, MonitorLoop


class FakeClient:
    def __init__(self):
        self.cancelled, self.removed = [], []

    def cancel(self, rid):
        self.cancelled.append(rid)

    def remove(self, rid):
        self.removed.append(rid)


def make_loop(job, client):
    loop = MonitorLoop.__new__(MonitorLoop)
    loop._pending_restart_hooks = {}
    loop._store = SimpleNamespace(upsert=lambda j: None, mark_finished=lambda *a: None)
    loop._get_client = lambda j: client
    loop.restarted, loop.hooks_run = [], []
    loop._restart_job = lambda j: loop.restarted.append(j.job_id)
    loop._run_restart_hooks = lambda j, rid: loop.hooks_run.append(rid)
    return loop


def make_job(status):
    return JobRecord(
        job_id="j",
        definition=None,
        runtime=JobRuntime(submitted=True, runtime_job_id="123", last_status=status),
    )


def test_action_carries_the_flag():
    res = RestartAction(RestartActionConfig(reason="r", wait_for_job_end=True)).execute(
        ActionContext(
            event=EventRecord(event_id="e", name="n", source="log", payload={}, metadata={})
        )
    )
    assert res.metadata["wait_for_job_end"] is True
    assert RestartActionConfig().wait_for_job_end is False


def test_restart_is_deferred_while_the_job_runs_and_resumes_after_it_ends():
    job, client = make_job("RUNNING"), FakeClient()
    loop = make_loop(job, client)
    loop._pending_restart_hooks["j"] = {
        "pre_command": "scan",
        "exclude_file": "/x",
        "wait_for_job_end": True,
    }
    assert loop._apply_effect(job, "restart", "123") is True
    assert loop.restarted == [] and client.cancelled == []
    assert job.runtime.deferred_restart["pre_command"] == "scan"
    # still running: keep waiting
    assert loop._resume_deferred_restart(job, "123") is True and loop.restarted == []
    # left the queue (COMPLETED): hooks run, restart happens, hooks carry the pre_command
    job.runtime.last_status = "COMPLETED"
    assert loop._resume_deferred_restart(job, "123") is True
    assert loop.restarted == ["j"] and loop.hooks_run == ["123"]
    assert loop._pending_restart_hooks["j"]["pre_command"] == "scan"
    assert job.runtime.deferred_restart == {}


def test_immediate_restart_without_the_flag():
    job, client = make_job("RUNNING"), FakeClient()
    loop = make_loop(job, client)
    loop._pending_restart_hooks["j"] = {
        "pre_command": "",
        "exclude_file": "",
        "wait_for_job_end": False,
    }
    assert loop._apply_effect(job, "restart", "123") is True
    assert loop.restarted == ["j"] and job.runtime.deferred_restart == {}


def test_flag_with_ended_job_restarts_at_once():
    job, client = make_job("FAILED"), FakeClient()
    loop = make_loop(job, client)
    loop._pending_restart_hooks["j"] = {"wait_for_job_end": True}
    assert loop._apply_effect(job, "restart", "123") is True
    assert loop.restarted == ["j"]


def test_cancel_first_cancels_now_and_defers_the_resubmission():
    job, client = make_job("RUNNING"), FakeClient()
    loop = make_loop(job, client)
    loop._pending_restart_hooks["j"] = {
        "pre_command": "scan",
        "exclude_file": "",
        "wait_for_job_end": False,
        "cancel_first": True,
    }
    assert loop._apply_effect(job, "restart", "123") is True
    assert client.cancelled == ["123"] and loop.restarted == []  # killed now, not resubmitted yet
    assert job.runtime.deferred_restart.get("cancel_first") is True
    job.runtime.last_status = "CANCELLED"
    assert loop._resume_deferred_restart(job, "123") is True
    assert loop.restarted == ["j"] and loop.hooks_run == ["123"]  # hooks (scan) run after the end


def test_cancel_first_on_an_ended_job_restarts_at_once():
    job, client = make_job("FAILED"), FakeClient()
    loop = make_loop(job, client)
    loop._pending_restart_hooks["j"] = {"cancel_first": True}
    assert loop._apply_effect(job, "restart", "123") is True
    assert client.cancelled == [] and loop.restarted == ["j"]
