"""Jobs as the unit (docs/pool_allocation_improvement.md §2, §11 P2): a durable cancel
tombstone, hold, replace / supersede, status --job, and one monitor per session."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import replace

import pytest
from compoconf import asdict

from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, MonitorLoop
from oellm_autoexp.monitor.pool import (
    PoolHostConfig,
    create_or_update_pool,
    job_control_dir,
    job_marker,
    pool_inputs_waiting,
    set_job_marker,
    submit_request,
    write_inbox,
)
from oellm_autoexp.orchestrator import (
    LOCK_DIRNAME,
    _sleep_interruptibly,
    acquire_monitor_lock,
    release_monitor_lock,
)
from tests.pool.test_pool_ctl import pool_ctl
from tests.pool.test_pool_multiuser import COLLEAGUE, OWNER, MultiEnv
from tests.pool.test_pool_unit import REPO, Env, _host_definition, _run_definition


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    return MultiEnv(tmp_path, monkeypatch)


def _request(env, uid, op, **payload) -> str:
    return env.request_as(uid, op, **payload)["result"]


# ---------------------------------------------------------------- cancel tombstone
def test_cancel_job_closes_every_record_and_survives_a_monitor_restart(env):
    env.add_run("train")
    env.add_run("train_q2")  # a second live record of the same job (same definition name)
    rec = env.rec("train_q2")
    env.store.upsert(replace(rec, definition=replace(rec.definition, name="train")))
    env.poll()
    res = _request(env, OWNER, "cancel_job", job="train")
    assert res.startswith("job owner:train cancelled"), res
    for job_id in ("train", "train_q2"):
        assert env.rec(job_id).runtime.final_state == "cancelled"
    assert job_marker(env.layout, "owner:train", "cancel")["by"] == "owner"
    # a restarted monitor (new loop, same session) with a record that reappears: closed again
    env.store.upsert(replace(env.rec("train"), runtime=type(env.rec("train").runtime)()))
    env.loop = MonitorLoop(env.store, slurm_client=env.slurm, show_poll_state=False,
                           no_error_catching=True)
    env.loop.rehydrate()
    env.poll()
    assert env.rec("train").runtime.final_state == "cancelled"
    assert "job cancelled by owner" in env.layout.ledger.read_text()


def test_a_tombstone_acts_on_its_own_pool_only(tmp_path, monkeypatch):
    """User 2026-11-26: a job lives in one pool; moving it is a cancel + submit, so a cancel in
    one pool never closes a same-named job (or a singleton copy) in another pool."""
    monkeypatch.chdir(REPO)
    env = Env(tmp_path)
    other_session = tmp_path / "session2"
    other = create_or_update_pool(tmp_path / "pool2", host_definition=_host_definition(tmp_path),
                                  config=PoolHostConfig(name="pool2", min_remaining_s=0),
                                  session_dir=other_session)
    store2 = JobFileStore(other_session)
    store2.upsert(JobRecord(job_id="main", definition=_run_definition(tmp_path, other.root, "main")))
    loop2 = MonitorLoop(store2, slurm_client=env.slurm, show_poll_state=False, no_error_catching=True)
    loop2.observe_once()
    me = pool_ctl.pool_owner(env.layout)[0]
    set_job_marker(env.layout, f"{me}:main", "cancel", "pool_ctl")  # given in the first pool
    assert job_control_dir(env.layout, "k").is_relative_to(env.layout.root)
    loop2.observe_once()
    assert store2.load("main", include_finished=True).runtime.final_state is None
    set_job_marker(other, f"{me}:main", "cancel", "pool_ctl")  # in its own pool it does
    loop2.observe_once()
    assert store2.load("main", include_finished=True).runtime.final_state == "cancelled"


def test_a_cancelled_job_is_refused_on_ingest_until_replace(env):
    env.set(hold_foreign_jobs=False)
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert _request(env, COLLEAGUE, "cancel_job", job="fx").startswith("job colleague:fx cancelled")
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    res = env.result(path)
    assert res["status"] == "rejected" and "enqueue with --replace" in res["message"]
    path = env.submit_as(COLLEAGUE, "fx", pool_replace=True)
    env.poll()
    res = env.result(path)
    assert res["status"] == "queued" and res["job_id"] == "fx_q2"
    assert job_marker(env.layout, "colleague:fx", "cancel") is None


def test_replace_supersedes_the_older_live_record(env):
    env.set(hold_foreign_jobs=False)
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    path = env.submit_as(COLLEAGUE, "fx", pool_replace=True)
    env.poll()
    res = env.result(path)
    assert res["job_id"] == "fx_q2" and "superseded ['fx'] by fx_q2" in res["message"]
    assert env.rec("fx").runtime.final_state == "cancelled"
    assert env.rec("fx_q2").runtime.final_state is None


def test_a_held_replace_supersedes_at_the_permit(env):
    env.set(hold_foreign_jobs=False)
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    env.set(hold_foreign_jobs=True)
    env.submit_as(COLLEAGUE, "fx", pool_replace=True)
    env.poll()
    assert env.rec("fx").runtime.final_state is None, "the old one runs until the permit"
    _request(env, OWNER, "permit", targets=["fx_q2"])
    assert env.rec("fx").runtime.final_state == "cancelled"


def test_supersede_request_of_the_owner(env):
    env.add_run("a")
    env.add_run("b")
    env.poll()
    assert _request(env, OWNER, "supersede", key="owner:a", keep="nothing") == "superseded ['a'] by nothing"
    assert env.rec("b").runtime.final_state is None
    assert _request(env, COLLEAGUE, "supersede", key="owner:b", keep="x").startswith("error:")


def test_a_submitter_cancels_only_their_own_job(env):
    env.add_run("train")
    env.poll()
    res = _request(env, COLLEAGUE, "cancel_job", job="train", user="owner")
    assert res == "error: colleague may only steer their own jobs"
    # without --user it means the colleague's own job of that name: none here, nothing closed
    assert _request(env, COLLEAGUE, "cancel_job", job="train").startswith("job colleague:train")
    env.poll()
    assert env.rec("train").runtime.final_state is None


def test_a_name_of_several_users_needs_user(env):
    env.set(hold_foreign_jobs=False)
    env.add_run("ev")
    env.poll()
    # the name-clash guard refuses a second user's "ev"; a record written by older code can
    # still exist: it names its submitter in the metadata
    rec = env.rec("ev")
    md = {**rec.definition.metadata, "pool_submitted_by": {"user": "colleague", "uid": COLLEAGUE}}
    env.store.upsert(JobRecord(job_id="ev_c", definition=replace(rec.definition, metadata=md)))
    env.poll()
    assert "pass --user" in _request(env, OWNER, "cancel_job", job="ev")
    assert _request(env, OWNER, "cancel_job", job="ev", user="colleague").startswith("job colleague:ev")
    assert env.rec("ev_c").runtime.final_state == "cancelled"
    assert env.rec("ev").runtime.final_state is None


# ---------------------------------------------------------------- hold
def test_a_held_job_is_not_placed_until_unhold(env):
    env.add_run("a")
    env.poll()
    assert _request(env, OWNER, "hold_job", job="a") == "job owner:a held (its queued runs are not placed)"
    env.poll()
    hid = env.host_up(env.hosts()[0])
    env.poll()
    assert env.assigned(hid) == []
    assert _request(env, OWNER, "unhold_job", job="a") == "job owner:a released"
    assert env.assigned(hid) == [env.rec("a").runtime.runtime_job_id]


# ---------------------------------------------------------------- pool_ctl
def test_pool_ctl_cancel_job_and_status_job(env, capsys):
    env.add_run("train")
    env.poll()
    assert pool_ctl.main([str(env.layout.root), "cancel", "--job", "train", "--no-wait"]) == 0
    request = sorted(env.layout.requests_dir.glob("*.json"))[0]
    assert json.loads(request.read_text())["op"] == "cancel_job"
    env.poll()
    capsys.readouterr()
    assert pool_ctl.main([str(env.layout.root), "status", "--job", "train"]) == 0
    out = capsys.readouterr().out
    assert "job owner:train  CANCELLED" in out and "cancel: by owner" in out
    assert "record train: cancelled" in out


def test_pool_ctl_hold_job_is_allowed_for_a_submitter(env, monkeypatch):
    monkeypatch.setattr(pool_ctl, "_my_uid", lambda: COLLEAGUE)
    assert pool_ctl.main([str(env.layout.root), "hold", "--job", "fx", "--no-wait"]) == 0
    assert pool_ctl.main([str(env.layout.root), "hold", "host001"]) == 1, "a host hold is the owner's"


# ---------------------------------------------------------------- one monitor per session
def test_the_monitor_lock_refuses_a_live_holder_and_takes_dead_or_stale_ones(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    acquire_monitor_lock(session)
    acquire_monitor_lock(session)  # re-entrant for its holder
    owner = session / LOCK_DIRNAME / "owner"
    host = socket.gethostname()
    # a live process on this host holds it
    owner.write_text(f"{os.getppid()} {host} 0\n")
    with pytest.raises(SystemExit, match="Another monitor holds this session"):
        acquire_monitor_lock(session)
    # --force takes it
    acquire_monitor_lock(session, force=True)
    assert owner.read_text().split()[0] == str(os.getpid())
    # a dead pid on this host
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    owner.write_text(f"{dead.pid} {host} 0\n")
    acquire_monitor_lock(session)
    # another host: live until stale
    owner.write_text("1 other-login-node 0\n")
    with pytest.raises(SystemExit):
        acquire_monitor_lock(session)
    old = time.time() - 3600
    os.utime(owner, (old, old))
    acquire_monitor_lock(session)
    release_monitor_lock(session)
    assert not (session / LOCK_DIRNAME).exists()


def test_release_leaves_another_holders_lock(tmp_path):
    session = tmp_path / "session"
    (session / LOCK_DIRNAME).mkdir(parents=True)
    (session / LOCK_DIRNAME / "owner").write_text(f"1 other-login-node {time.time():.0f}\n")
    release_monitor_lock(session)
    assert (session / LOCK_DIRNAME).exists()


def test_a_waiting_request_ends_the_monitors_sleep(env):
    assert not pool_inputs_waiting(env.session)
    t0 = time.monotonic()
    submit_request(env.layout.root, "close")
    assert pool_inputs_waiting(env.session)
    _sleep_interruptibly(30.0, wake=lambda: pool_inputs_waiting(env.session), wake_every_s=0.5)
    assert time.monotonic() - t0 < 5


def test_an_inbox_file_ends_the_monitors_sleep_too(env):
    definition = _run_definition(env.tmp, env.layout.root, "fx")
    write_inbox(env.layout.root, "fx", asdict(JobRecord(job_id="fx", definition=definition)))
    assert pool_inputs_waiting(env.session)


def test_a_lock_without_owner_is_a_starting_monitor_until_it_is_old(tmp_path):
    session = tmp_path / "session"
    (session / LOCK_DIRNAME).mkdir(parents=True)
    with pytest.raises(SystemExit, match="another monitor is starting"):
        acquire_monitor_lock(session)
    old = time.time() - 120
    os.utime(session / LOCK_DIRNAME, (old, old))
    acquire_monitor_lock(session)
    assert (session / LOCK_DIRNAME / "owner").read_text().split()[0] == str(os.getpid())



def test_uncancel_clears_the_tombstone_without_enqueuing(env):
    env.add_run("train")
    env.poll()
    _request(env, OWNER, "cancel_job", job="train")
    assert _request(env, OWNER, "uncancel_job", job="train") == (
        "job owner:train uncancelled (it may be enqueued again; nothing was enqueued)")
    assert job_marker(env.layout, "owner:train", "cancel") is None
    assert env.rec("train").runtime.final_state == "cancelled", "the closed record stays closed"
    assert _request(env, OWNER, "uncancel_job", job="train") == "job owner:train was not cancelled"
    assert "job uncancelled by owner" in env.layout.ledger.read_text()


def test_pool_ctl_jobs_lists_jobs_and_markers(env, capsys):
    env.add_run("train")
    env.add_run("bench")
    env.submit_as(COLLEAGUE, "evalx")
    env.poll()
    _request(env, OWNER, "hold_job", job="bench")
    _request(env, OWNER, "cancel_job", job="gone")  # a tombstone without a record here
    capsys.readouterr()
    assert pool_ctl.main([str(env.layout.root), "jobs"]) == 0
    out = capsys.readouterr().out
    assert "owner:train" in out and "owner:bench" in out and "HELD" in out
    assert "colleague:evalx" in out and "AWAITING PERMIT" in out
    assert "owner:gone" not in out, "markers of jobs without a record here only with --all"
    assert pool_ctl.main([str(env.layout.root), "jobs", "--all"]) == 0
    out = capsys.readouterr().out
    assert "owner:gone" in out and "CANCELLED" in out and "(no record in this pool)" in out
