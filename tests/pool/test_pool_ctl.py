"""pool_ctl refuses what cannot work and reports what the monitor did."""

from __future__ import annotations

import importlib.util
import json
import os
import time

import pytest

from oellm_autoexp.monitor.pool import write_rc
from tests.pool.test_pool_unit import REPO, Env

spec = importlib.util.spec_from_file_location("pool_ctl", REPO / "scripts" / "pool_ctl.py")
pool_ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool_ctl)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    e = Env(tmp_path)
    for record_id in ("a", "b"):
        e.add_run(record_id)
    e.poll(2)
    hid = e.host_up(e.hosts()[0])
    e.poll()
    e.hid = hid
    return e


def _ctl(env, *args):
    return pool_ctl.main([str(env.layout.root), *args])


def _requests(env):
    return sorted(env.layout.requests_dir.glob("*.json"))


def test_cancel_of_a_finished_job_is_refused_up_front(env, capsys):
    rid = env.rec("a").runtime.runtime_job_id
    env.start_run(rid, env.hid)
    write_rc(env.layout, rid, 0, "completed")
    env.poll()
    assert env.rec("a").runtime.final_state == "finished"
    assert _ctl(env, "cancel", rid) == 1
    err = capsys.readouterr().err
    assert "nothing to cancel" in err and "a already ended (finished at" in err
    assert _requests(env) == []  # nothing was written


def test_cancel_of_an_unknown_target(env, capsys):
    assert _ctl(env, "cancel", "nope") == 1
    assert "nothing in this pool matches 'nope'" in capsys.readouterr().err
    assert _requests(env) == []


def test_cancel_by_an_earlier_run_id_reaches_the_job_and_reports_the_result(env, capsys):
    first = env.rec("a").runtime.runtime_job_id
    env.start_run(first, env.hid)
    write_rc(env.layout, first, 0, "preempted")
    env.poll()
    assert env.rec("a").runtime.runtime_job_id != first  # requeued: a new run
    assert _ctl(env, "cancel", first, "--no-wait") == 0
    request = _requests(env)[0]
    assert json.loads(request.read_text())["targets"] == ["a"]
    env.poll()  # the monitor executes it (and ticks)
    assert pool_ctl._await_result(env.layout, request, 5) == 0
    assert "monitor: cancelled ['a']" in capsys.readouterr().out
    assert env.rec("a").runtime.final_state == "cancelled"
    assert env.rec("b").runtime.final_state is None


def test_a_name_matching_several_active_jobs_needs_all(env, capsys):
    env.store.upsert(
        type(env.rec("b"))(
            job_id="b_q2",
            definition=env.rec("b").definition,
            runtime=type(env.rec("b").runtime)(),
        )
    )
    assert _ctl(env, "cancel", "b") == 1
    assert "matches 2 active jobs" in capsys.readouterr().err
    assert _ctl(env, "cancel", "b", "--all", "--no-wait") == 0
    assert sorted(json.loads(_requests(env)[0].read_text())["targets"]) == ["b", "b_q2"]


def test_waiting_without_a_live_monitor_returns_at_once(env, capsys):
    old = time.time() - 3600
    os.utime(env.layout.monitor_alive, (old, old))
    t0 = time.time()
    assert _ctl(env, "close", "--wait", "60") == 0
    assert time.time() - t0 < 5
    assert "no live monitor (last tick: 3600s ago)" in capsys.readouterr().out


def test_preempt_only_a_running_run(env, capsys):
    queued = env.rec("b").runtime.runtime_job_id
    assert _ctl(env, "preempt", queued) == 1
    assert "is PENDING" in capsys.readouterr().err
    running = env.rec("a").runtime.runtime_job_id
    env.start_run(running, env.hid)
    assert _ctl(env, "preempt", running) == 0
    assert (env.layout.run_dir(running) / "preempt").exists()
    write_rc(env.layout, running, 0, "preempted")
    assert _ctl(env, "preempt", running) == 1
    assert "is PREEMPTED (rc 0 preempted)" in capsys.readouterr().err
    assert _ctl(env, "preempt", "pool.9999") == 1
