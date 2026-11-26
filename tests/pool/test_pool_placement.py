"""P3: a placement failure is not a failed attempt (docs/pool_allocation_improvement.md §11).

The monitor assigns a run only where it fits, counting the run's own EXCLUDE; a run the supervisor
still refuses keeps its rid and waits; a lost host is counted per host; run faults back off.
"""

from __future__ import annotations

import logging
import time

import pytest

from oellm_autoexp.monitor.pool import (
    expand_hostlist,
    node_set,
    read_run_state,
    write_rc,
)
from tests.pool.test_pool_unit import REPO, Env


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    return Env(tmp_path)


def _queued_on_two_node_host(env, *names, nodes=1):
    for name in names:
        env.add_run(name, nodes=nodes)
    env.poll(2)
    return env.host_up(env.hosts()[0])  # nodes a,b; supervisor state: 2 usable


def _set_job_exclude(env, rid, nodes):
    meta = env.layout.run_dir(rid) / "meta.env"
    text = meta.read_text().replace("EXCLUDE=''", f"EXCLUDE={nodes}", 1)
    meta.write_text(text.replace("JOB_EXCLUDE=''", f"JOB_EXCLUDE={nodes}", 1))


def test_hostlist_expansion():
    assert expand_hostlist("as01r1b[01-03,07]") == ["as01r1b01", "as01r1b02", "as01r1b03", "as01r1b07"]
    assert expand_hostlist("n[1-2]x[8-9]") == ["n1x8", "n1x9", "n2x8", "n2x9"]
    assert expand_hostlist("plain") == ["plain"]
    assert node_set("a,b\nc  # why\nd[1-2]") == {"a", "b", "c", "d1", "d2"}


# ---------------------------------------------------------------- mode A: fit with own EXCLUDE
def test_a_run_whose_own_exclude_leaves_too_few_nodes_is_not_assigned(env, caplog):
    """11-24: the monitor counted the host's usable nodes without the run's own EXCLUDE; the
    supervisor then refused the 256-node run every poll."""
    hid = _queued_on_two_node_host(env, "big", nodes=2)
    rid = env.rec("big").runtime.runtime_job_id
    _set_job_exclude(env, rid, "b")
    with caplog.at_level(logging.INFO):
        env.poll(3)
    assert env.assigned(hid) == []
    waits = [r.message for r in caplog.records if f"{rid} waits" in r.message]
    assert len(waits) == 1, "logged once per change, not every poll"
    assert f"needs 2, {hid} has 1 usable for it (own EXCLUDE 1 in the allocation)" in waits[0]
    _set_job_exclude(env, rid, "")  # its exclusion lifted (e.g. readmitted): placed at once
    meta = env.layout.run_dir(rid) / "meta.env"
    meta.write_text(meta.read_text().replace("EXCLUDE=b", "EXCLUDE=''").replace("JOB_EXCLUDE=b", "JOB_EXCLUDE=''"))
    env.poll()
    assert env.assigned(hid) == [rid]


def test_own_exclude_nodes_excluded_for_the_host_anyway_do_not_count_twice(env):
    hid = _queued_on_two_node_host(env, "one")
    rid = env.rec("one").runtime.runtime_job_id
    _set_job_exclude(env, rid, "b,x9")  # b is excluded for the host already; x9 is not in it
    (env.layout.host_dir(hid) / "exclude_nodes").write_text("b\n")
    hdir = env.layout.host_dir(hid)
    (hdir / "state").write_text(f"idle {int(time.time())} 1\n")  # the supervisor counts 1 usable
    env.poll()
    assert env.assigned(hid) == [rid]


# ---------------------------------------------------------------- mode B: refusal keeps the rid
def _refuse(env, rid, hid, need=2, usable=1, total=2):
    """What the P3 supervisor does when it cannot place the run."""
    (env.layout.run_dir(rid) / "placement_refused").write_text(
        f"host={hid}\nneed={need}\nusable={usable}\ntotal_usable={total}\nts={int(time.time()) + 1}\n")
    (env.layout.run_dir(rid) / "assigned").unlink()
    (env.layout.host_dir(hid) / "assign" / rid).unlink()


def test_a_refused_run_keeps_its_rid_and_waits_until_the_host_changes(env, caplog):
    hid = _queued_on_two_node_host(env, "big", nodes=2)
    env.poll()
    rid = env.rec("big").runtime.runtime_job_id
    assert env.assigned(hid) == [rid]
    _refuse(env, rid, hid)
    with caplog.at_level(logging.INFO):
        env.poll(3)
    assert env.rec("big").runtime.runtime_job_id == rid, "same run, no new run dir"
    assert read_run_state(env.layout, rid).status == "PENDING"
    assert env.assigned(hid) == [], "not sent back while nothing changed"
    assert sum(f"{rid} waits: refused by {hid}" in r.message for r in caplog.records) == 1
    assert env.rec("big").runtime.attempts == 1
    # the host's usable count changes (a node readmitted, the supervisor reports 3): tried again
    (env.layout.host_dir(hid) / "state").write_text(f"idle {int(time.time())} 3\n")
    env.poll()
    assert env.assigned(hid) == [rid]
    assert not (env.layout.run_dir(rid) / "placement_refused").exists()


def test_an_exclusion_change_lets_a_refused_run_try_again(env):
    hid = _queued_on_two_node_host(env, "big", nodes=2)
    env.poll()
    rid = env.rec("big").runtime.runtime_job_id
    _refuse(env, rid, hid)
    env.poll()
    assert env.assigned(hid) == []
    time.sleep(1.1)
    (env.layout.host_dir(hid) / "exclude_nodes").write_text("")  # touched: lists changed
    env.poll()
    assert env.assigned(hid) == [rid]


def test_an_old_supervisors_insufficient_nodes_does_not_go_back_to_that_host(env):
    """Compatibility: a supervisor of older code ends the run with rc -1 insufficient_nodes. The
    job is requeued (a new run), not counted, and not sent to that host until it changes."""
    hid = _queued_on_two_node_host(env, "big", nodes=2)
    env.poll()
    first = env.rec("big").runtime.runtime_job_id
    write_rc(env.layout, first, -1, "insufficient_nodes")
    (env.layout.run_dir(first) / "started").unlink(missing_ok=True)
    env.poll(3)
    rec = env.rec("big")
    assert rec.runtime.final_state is None and rec.runtime.runtime_job_id != first
    assert rec.runtime.action_state["pool:refused"]["host"] == hid
    assert env.assigned(hid) in ([], [first]) and rec.runtime.runtime_job_id not in env.assigned(hid)


# ---------------------------------------------------------------- lost hosts, per host
def test_a_lost_host_is_counted_on_the_host_not_the_job(env):

    hid = _queued_on_two_node_host(env, "job_a")
    env.poll()
    rid = env.rec("job_a").runtime.runtime_job_id
    env.start_run(rid, hid)
    env.fake.set_state(env.hosts()[0].runtime.runtime_job_id, "FAILED")  # the allocation is gone
    env.poll(2)
    assert read_run_state(env.layout, rid).reason == "host_lost"
    assert (env.layout.host_dir(hid) / "lost_runs").read_text().split()[0] == rid
    assert env.rec("job_a").runtime.final_state is None, "requeued, no attempt used"


def test_a_live_host_that_lost_three_runs_is_a_health_problem(env):
    from oellm_autoexp.monitor.pool_doctor import doctor_report

    hid = _queued_on_two_node_host(env, "job_a")
    (env.layout.host_dir(hid) / "lost_runs").write_text("p.1 1\np.2 2\np.3 3\n")
    report = [line for level, line in doctor_report(env.layout, slurm=None) if level == "PROBLEM"]
    assert any(f"host {hid}: lost 3 run(s)" in line for line in report), report


# ---------------------------------------------------------------- backoff after run faults
def _fail_current_run(env, hid, lived_s=10):
    rid = env.rec("job_a").runtime.runtime_job_id
    (env.layout.run_dir(rid) / "started").write_text(
        f"host={hid}\njob=1\nstep_index=1\nnodes=a\nts={int(time.time() - lived_s)}\n")
    write_rc(env.layout, rid, 1, "failed")
    return rid


def test_consecutive_faults_back_off_and_a_long_run_resets_the_count(env):
    env.add_run("job_a")
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    first = _fail_current_run(env, hid)
    rec = env.rec("job_a")
    rec.runtime.last_status = "FAILED"
    env.loop._pool_record_fault(rec)
    info = rec.runtime.action_state["pool:backoff"]
    assert info["n"] == 1 and info["rid"] == first and 50 < info["until"] - time.time() <= 60
    env.loop._pool_record_fault(rec)
    assert info["n"] == 1, "recorded once per fault"
    info["rid"] = "older"  # a next fault after a short run: 180 s
    env.loop._pool_record_fault(rec)
    assert info["n"] == 2 and 170 < info["until"] - time.time() <= 180
    (env.layout.run_dir(first) / "started").write_text(f"host={hid}\nts={int(time.time() - 7200)}\n")
    info["rid"] = "older2"  # a fault after 2 h of running starts the count anew
    env.loop._pool_record_fault(rec)
    assert info["n"] == 1


def test_a_fault_is_recorded_before_the_jobs_restart_event(env):
    """The job's state event restarts it on FAILED: its next run must already carry the backoff."""
    from oellm_autoexp.monitor.actions import RestartActionConfig, StateEventConfig

    rid0 = env.add_run("job_a")
    rec = env.rec(rid0)
    rec.definition.state_events = [StateEventConfig(name="retry", transition=(None, "FAILED"),
                                                    action=RestartActionConfig(reason="retry"))]
    env.store.upsert(rec)
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    first = _fail_current_run(env, hid)
    env.poll()
    rec = env.rec("job_a")
    assert rec.runtime.runtime_job_id != first, "restarted by its own event"
    assert rec.runtime.action_state["pool:backoff"]["rid"] == first
    env.poll()
    assert env.assigned(hid) == [] or rec.runtime.runtime_job_id not in env.assigned(hid)


def test_a_job_in_backoff_is_not_placed(env, caplog):
    env.add_run("job_a")
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    rec = env.rec("job_a")
    rec.runtime.action_state["pool:backoff"] = {"rid": "x", "n": 2, "until": time.time() + 120}
    env.store.upsert(rec)
    with caplog.at_level(logging.INFO):
        env.poll()
    assert env.assigned(hid) == []
    assert any("backoff after a run fault" in r.message for r in caplog.records)
    rec = env.rec("job_a")
    rec.runtime.action_state["pool:backoff"]["until"] = time.time() - 1
    env.store.upsert(rec)
    env.poll()
    assert env.assigned(hid) == [rec.runtime.runtime_job_id]
