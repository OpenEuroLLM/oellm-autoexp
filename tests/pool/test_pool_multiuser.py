"""Multi-user pools: another user's jobs come in through the inbox, in an ordered way.

docs/pool_allocation_improvement.md §10. Tests cannot change uid, so the submitter of a file is
injected through ``pool.file_uid`` (the monitor's only source of identity) and ``pool.user_name``.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from compoconf import asdict

import oellm_autoexp.monitor.pool as pool_mod
from oellm_autoexp.monitor.loop import JobRecord, JobRuntime
from oellm_autoexp.monitor.pool import (
    create_or_update_pool,
    foreign_enqueue_warnings,
    held_jobs,
    inbox_result_path,
    login_node_commands,
    pools_have_pending_requests,
    read_env_file,
    submit_request,
    write_inbox,
    write_rc,
)
from tests.pool.test_pool_ctl import pool_ctl
from tests.pool.test_pool_unit import REPO, Env, _host_definition, _run_definition

OWNER = os.getuid()
COLLEAGUE = OWNER + 4242
STRANGER = OWNER + 4343
NAMES = {OWNER: "owner", COLLEAGUE: "colleague", STRANGER: "stranger"}


class MultiEnv(Env):
    """Env plus files written "as" another user."""

    def __init__(self, tmp: Path, monkeypatch, **cfg) -> None:
        self.owners: dict[str, int] = {}
        real = pool_mod.file_uid
        monkeypatch.setattr(pool_mod, "file_uid", lambda p: self.owners.get(Path(p).name, real(p)))
        monkeypatch.setattr(pool_mod, "user_name", lambda uid: NAMES.get(uid, str(uid)))
        # the colleague is a member of the host account's group "acc", the stranger of nothing
        monkeypatch.setattr(
            pool_mod, "user_in_group", lambda user, group: user == "colleague" if group == "acc" else None
        )
        monkeypatch.setattr(pool_mod, "sacctmgr_assoc_users", lambda account, user: set())
        super().__init__(tmp, **{"foreign_submitters": "colleague", **cfg})

    def submit_as(self, uid: int, name: str, **meta) -> Path:
        definition = _run_definition(self.tmp, self.layout.root, name)
        definition = replace(definition, metadata={**definition.metadata, **meta})
        payload = asdict(JobRecord(job_id=name, definition=definition))
        path = write_inbox(self.layout.root, name, payload)
        self.owners[path.name] = uid
        return path

    def request_as(self, uid: int, op: str, **payload) -> dict:
        path = submit_request(self.layout.root, op, **payload)
        self.owners[path.name] = uid
        self.poll()
        return json.loads((self.layout.requests_dir / "done" / path.name).read_text())

    def result(self, path: Path) -> dict:
        return json.loads(inbox_result_path(self.layout, path).read_text())

    def set(self, **values) -> None:
        meta = self.layout.load_meta()
        meta["config"] = {**meta["config"], **values}
        self.layout.write_meta(meta)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    return MultiEnv(tmp_path, monkeypatch)


def _session_ids(env) -> list[str]:
    return sorted(p.name[: -len(".job.json")] for p in env.session.glob("*.job.json") if "host" not in p.name)


# ---------------------------------------------------------------- hold / permit
def test_a_foreign_job_is_held_and_inert(env):
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    res = env.result(path)
    assert (res["status"], res["job_id"], res["user"]) == ("held", "fx", "colleague")
    assert "pool_ctl permit fx" in res["message"]
    assert _session_ids(env) == [], "a held job is not a record of the session"
    assert not any(env.layout.runs_dir.iterdir()), "no run dir, no step script"
    assert [p["job_id"] for p in held_jobs(env.layout)] == ["fx"]
    assert not path.exists() and (env.layout.inbox_done_dir / path.name).exists()
    assert "held by colleague" in env.layout.ledger.read_text()


def test_permit_by_the_owner_queues_the_job_and_records_the_submitter(env):
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert env.request_as(OWNER, "permit", targets=["fx"])["result"] == "permitted ['fx']"
    assert held_jobs(env.layout) == []
    env.poll()
    rec = env.rec("fx")
    assert rec.definition.metadata["pool_submitted_by"]["user"] == "colleague"
    assert rec.definition.metadata["pool_foreign"]["permitted_by"] == "owner"
    rid = rec.runtime.runtime_job_id
    run = json.loads((env.layout.run_dir(rid) / "run.json").read_text())
    assert run["submitted_by"] == "colleague" and run["accounts"] == ["acc"]
    assert "export AUTOEXP_SUBMITTED_BY=colleague" in (env.layout.run_dir(rid) / "step.sh").read_text()
    assert "permitted by owner" in env.layout.ledger.read_text()


def test_a_permit_written_by_a_foreign_user_is_refused(env):
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    result = env.request_as(COLLEAGUE, "permit", targets=["fx"])["result"]
    assert result.startswith("error: permit is for the pool owner (owner)"), result
    assert [p["job_id"] for p in held_jobs(env.layout)] == ["fx"]


def test_without_the_hold_a_foreign_job_queues_directly(env):
    env.set(hold_foreign_jobs=False)
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert env.result(path)["status"] == "queued"
    assert env.rec("fx").runtime.runtime_job_id, "submitted to the pool in the same poll"


def test_switching_the_hold_off_leaves_held_jobs_held(env):
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    env.set(hold_foreign_jobs=False)
    env.poll(2)
    assert [p["job_id"] for p in held_jobs(env.layout)] == ["fx"]


def test_standing_permit_for_named_users(env):
    env.set(foreign_auto_permit="colleague")
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert env.result(path)["status"] == "queued"


def test_permit_by_user_and_deny_by_the_submitter(env):
    env.submit_as(COLLEAGUE, "fx")
    env.submit_as(COLLEAGUE, "fy")
    env.poll()
    assert env.request_as(STRANGER, "deny", targets=["fx"])["result"] == "error: not yours to deny: ['fx']"
    assert env.request_as(COLLEAGUE, "deny", targets=["fx"], reason="wrong lr")["result"] == "denied ['fx']"
    assert env.request_as(OWNER, "permit", user="colleague")["result"] == "permitted ['fy']"
    assert "denied by colleague: wrong lr" in env.layout.ledger.read_text()


# ---------------------------------------------------------------- who may submit
def test_a_user_not_in_foreign_submitters_is_rejected(env):
    path = env.submit_as(STRANGER, "sx")
    env.poll()
    res = env.result(path)
    assert res["status"] == "rejected" and "stranger may not submit" in res["message"]
    assert held_jobs(env.layout) == [] and _session_ids(env) == []


def test_the_owner_through_the_inbox_is_never_held(env):
    path = env.submit_as(OWNER, "ox", pool_priority=900, pool_singleton="main")
    env.poll()
    assert env.result(path)["status"] == "queued"
    md = env.rec("ox").definition.metadata
    assert (md["pool_priority"], md["pool_singleton"]) == (900, "main"), "the owner is not sanitized"


def test_group_submitters(env, monkeypatch):
    import grp

    env.set(foreign_submitters="@group")
    monkeypatch.setattr(grp, "getgrgid", lambda gid: SimpleNamespace(gr_name="acc"))
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert env.result(path)["status"] == "held"


# ---------------------------------------------------------------- sanitizing
def test_host_role_metadata_is_rejected(env):
    path = env.submit_as(COLLEAGUE, "fx", pool_role="host", pool_host_id="host001")
    env.poll()
    res = env.result(path)
    assert res["status"] == "rejected" and "pool_role, pool_host_id" in res["message"]


def test_priority_is_clamped_and_singleton_dropped_until_the_permit_restores_them(env):
    env.submit_as(COLLEAGUE, "fx", pool_priority=999, pool_singleton="ambitious")
    env.poll()
    md = held_jobs(env.layout)[0]["definition"]["metadata"]
    assert md["pool_priority"] == 60 and "pool_singleton" not in md
    assert md["pool_foreign"] == {"requested_priority": 999, "dropped_singleton": "ambitious", "commands": []}
    env.request_as(OWNER, "permit", targets=["fx"], priority=150, singleton=True)
    md = env.rec("fx").definition.metadata
    assert (md["pool_priority"], md["pool_singleton"]) == (150, "ambitious")


def test_a_submitted_runtime_is_never_trusted(env):
    """A record naming another job's run would let a cancel stop that run."""
    definition = _run_definition(env.tmp, env.layout.root, "fx")
    payload = asdict(
        JobRecord(job_id="fx", definition=definition,
                  runtime=JobRuntime(submitted=True, runtime_job_id="unitpool.0042", attempts=7))
    )
    path = write_inbox(env.layout.root, "fx", payload)
    env.owners[path.name] = COLLEAGUE
    env.set(hold_foreign_jobs=False)
    env.poll()
    runtime = env.rec("fx").runtime
    assert runtime.runtime_job_id != "unitpool.0042" and runtime.attempts == 1, "one attempt: its own"


def test_monitor_keys_are_overwritten(env):
    env.submit_as(COLLEAGUE, "fx", pool_submitted_by={"user": "owner", "uid": OWNER},
                  pool_accounts=["other"])
    env.poll()
    md = held_jobs(env.layout)[0]["definition"]["metadata"]
    assert md["pool_submitted_by"]["user"] == "colleague" and md["pool_accounts"] == ["acc"]


def test_a_name_of_another_submitters_active_job_is_rejected(env):
    env.add_run("train")  # the owner's job
    env.poll()
    path = env.submit_as(COLLEAGUE, "train")
    env.poll()
    res = env.result(path)
    assert res["status"] == "rejected" and "belongs to owner" in res["message"]


def test_login_node_commands_are_listed_for_the_owner():
    payload = {"log_events": [{"actions": [{"class_name": "RunCommandAction", "command": "rm -rf x"},
                                           {"class_name": "RestartAction", "pre_command": "scan.sh"}]}],
               "start_condition": {"class_name": "ShellCommandCondition", "command": "true"}}
    assert login_node_commands(payload) == [
        "RunCommandAction: rm -rf x", "RestartAction.pre_command: scan.sh", "ShellCommandCondition: true"]


# ---------------------------------------------------------------- accounts / writability
def test_account_check_rejects_a_non_member_unless_switched_off(env, monkeypatch):
    monkeypatch.setattr(pool_mod, "user_in_group", lambda user, group: False)
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    res = env.result(path)
    assert res["status"] == "rejected"
    assert "colleague is not a member of acc" in res["message"]
    assert "foreign_account_check=false" in res["message"]
    env.set(foreign_account_check=False)
    path = env.submit_as(COLLEAGUE, "fy")
    env.poll()
    assert env.result(path)["status"] == "held"
    assert "pool_accounts" not in held_jobs(env.layout)[0]["definition"]["metadata"]


def test_an_undecidable_membership_is_rejected(env, monkeypatch):
    monkeypatch.setattr(pool_mod, "user_in_group", lambda user, group: None)
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert "could not verify colleague's membership of acc" in env.result(path)["message"]


def test_sacctmgr_proves_membership_where_visible(env, monkeypatch):
    monkeypatch.setattr(pool_mod, "user_in_group", lambda user, group: None)
    monkeypatch.setattr(pool_mod, "sacctmgr_assoc_users", lambda account, user: {user})
    path = env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert env.result(path)["status"] == "held"


def test_a_foreign_job_only_takes_hosts_of_its_accounts(env):
    env.set(hold_foreign_jobs=False)
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    rid = env.rec("fx").runtime.runtime_job_id
    run_json = env.layout.run_dir(rid) / "run.json"
    run = json.loads(run_json.read_text())
    run["accounts"] = ["cns143"]  # the host is account "acc"
    run_json.write_text(json.dumps(run))
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    assert env.assigned(hid) == [], "the host's account is not one the submitter belongs to"
    run["accounts"] = ["acc"]
    run_json.write_text(json.dumps(run))
    env.poll()
    assert env.assigned(hid) == [rid]


def test_an_unwritable_log_dir_is_rejected(env):
    if OWNER == 0:
        pytest.skip("root writes everywhere")
    locked = env.tmp / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        definition = _run_definition(env.tmp, env.layout.root, "fx")
        definition = replace(definition, log_path=str(locked / "sub" / "slurm-%j.log"))
        path = write_inbox(env.layout.root, "fx", asdict(JobRecord(job_id="fx", definition=definition)))
        env.owners[path.name] = COLLEAGUE
        env.poll()
        res = env.result(path)
        assert res["status"] == "rejected" and "not writable for the pool owner" in res["message"]
    finally:
        locked.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------- steering own jobs
def test_a_submitter_cancels_their_own_job_but_not_the_owners(env):
    env.set(hold_foreign_jobs=False)
    env.add_run("train")
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    res = env.request_as(COLLEAGUE, "cancel", targets=["train", "fx"])["result"]
    assert res == "error: not yours to cancel: ['train']; cancelled ['fx']"
    assert env.rec("train").runtime.final_state is None
    assert env.rec("fx").runtime.final_state == "cancelled"


def test_a_submitter_preempts_their_own_running_run_through_a_request(env):
    env.set(hold_foreign_jobs=False)
    env.add_run("train")
    env.submit_as(COLLEAGUE, "fx")
    env.poll(2)
    hid = env.host_up(env.hosts()[0])
    env.poll()
    mine, theirs = env.rec("fx").runtime.runtime_job_id, env.rec("train").runtime.runtime_job_id
    for rid in (mine, theirs):
        env.start_run(rid, hid)
    assert env.request_as(COLLEAGUE, "preempt", run=theirs)["result"] == f"error: run {theirs} is not yours to preempt"
    assert env.request_as(COLLEAGUE, "preempt", run=mine)["result"] == f"preempt requested for {mine}"
    assert (env.layout.run_dir(mine) / "preempt").exists()
    assert not (env.layout.run_dir(theirs) / "preempt").exists()


def test_owner_only_requests_are_refused_for_others(env):
    for op in ("close", "drain", "adopt"):
        res = env.request_as(COLLEAGUE, op)["result"]
        assert res.startswith(f"error: {op} is for the pool owner"), res
    assert not env.layout.load_state().get("closed")


# ---------------------------------------------------------------- pool_ctl
def test_pool_ctl_refuses_owner_commands_for_a_submitter(env, monkeypatch, capsys):
    monkeypatch.setattr(pool_ctl, "_my_uid", lambda: COLLEAGUE)
    assert pool_ctl.main([str(env.layout.root), "set", "keep_pending=3"]) == 1
    assert "set is for the pool owner (owner)" in capsys.readouterr().err
    assert env.layout.load_config().keep_pending == 0
    assert pool_ctl.main([str(env.layout.root), "status"]) == 0


def test_pool_admins_may_steer(env, monkeypatch):
    env.set(pool_admins="colleague")
    monkeypatch.setattr(pool_ctl, "_my_uid", lambda: COLLEAGUE)
    assert pool_ctl.main([str(env.layout.root), "set", "keep_pending=1"]) == 0
    env.submit_as(COLLEAGUE, "fx")
    env.poll()
    assert held_jobs(env.layout) == [], "an admin is treated as the owner"


def test_pool_ctl_status_lists_held_jobs_and_permit_resolves_names(env, capsys):
    env.submit_as(COLLEAGUE, "fx", pool_priority=999)
    env.poll()
    assert pool_ctl.main([str(env.layout.root), "status"]) == 0
    out = capsys.readouterr().out
    assert "held jobs of other users" in out and "by colleague" in out and "asked priority 999" in out
    assert pool_ctl.main([str(env.layout.root), "permit", "fx", "--no-wait"]) == 0
    request = sorted(env.layout.requests_dir.glob("*.json"))[0]
    assert json.loads(request.read_text())["targets"] == ["fx"]
    assert pool_ctl.main([str(env.layout.root), "permit", "nope"]) == 1


# ---------------------------------------------------------------- pool creation / helpers
def test_only_the_owner_updates_a_pool(env, monkeypatch):
    from oellm_autoexp.monitor.pool import PoolHostConfig

    assert env.layout.load_meta()["owner"] == {"user": "owner", "uid": OWNER}
    monkeypatch.setattr(os, "getuid", lambda: COLLEAGUE)
    with pytest.raises(PermissionError, match="belongs to owner"):
        create_or_update_pool(env.layout.root, host_definition=_host_definition(env.tmp),
                              config=PoolHostConfig(name="unitpool"), session_dir=env.session)


def test_a_pool_without_owner_field_belongs_to_the_owner_of_pool_json(env):
    meta = env.layout.load_meta()
    meta.pop("owner")
    env.layout.write_meta(meta)
    assert pool_mod.pool_owner(env.layout) == ("owner", OWNER)


def test_an_inbox_file_keeps_the_monitor_alive(env):
    assert not pools_have_pending_requests(env.session)
    env.submit_as(COLLEAGUE, "fx")
    assert pools_have_pending_requests(env.session)


def test_foreign_enqueue_warnings(tmp_path):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    workdir.chmod(0o700)
    definition = _run_definition(tmp_path, tmp_path / "pool", "fx")
    definition = replace(definition, log_path="$HOME/logs/slurm-%j.log")
    warnings = foreign_enqueue_warnings([definition], workdir, "owner")
    assert any("not group-readable" in w for w in warnings)
    assert any("uses $HOME" in w and "owner's" in w for w in warnings)


def test_secure_leaves_only_inbox_and_requests_group_writable(env, capsys):
    env.add_run("train")
    env.poll()
    for path in [env.layout.root, *env.layout.root.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWGRP)
    args = SimpleNamespace(apply=False, no_acl=True)
    assert pool_ctl.cmd_secure(env.layout, args) == 0
    assert "chmod -R g-w" in capsys.readouterr().out
    step = next(env.layout.runs_dir.glob("*/step.sh"))
    assert step.stat().st_mode & stat.S_IWGRP, "plan only: nothing changed"
    args.apply = True
    assert pool_ctl.cmd_secure(env.layout, args) == 0
    assert not step.stat().st_mode & stat.S_IWGRP
    assert not env.layout.pool_json.stat().st_mode & stat.S_IWGRP
    for d in (env.layout.inbox_dir, env.layout.requests_dir):
        assert d.stat().st_mode & stat.S_IWGRP, d
    assert read_env_file(env.layout.pool_env), "the pool still reads"
    write_rc(env.layout, env.rec("train").runtime.runtime_job_id, 0, "completed")  # owner still writes


def test_a_pool_from_before_multi_user_gets_its_inbox_on_the_next_poll(env):
    import shutil

    shutil.rmtree(env.layout.inbox_dir)
    env.layout.requests_dir.chmod(0o755)
    env.poll()
    for d in (env.layout.inbox_dir, env.layout.requests_dir):
        mode = d.stat().st_mode
        assert mode & stat.S_IWGRP and mode & stat.S_ISGID, (d, oct(mode))
    assert env.layout.held_dir.is_dir() and env.layout.inbox_done_dir.is_dir()
