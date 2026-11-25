"""pool_ctl doctor: a read-only check that names each 2026-11-24 failure signature."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from tests.pool.test_pool_ctl import pool_ctl
from tests.pool.test_pool_unit import REPO, Env


def _fake_slurm(live: str):
    """squeue -o %N answers ``live``; scontrol show hostnames splits on commas."""

    def slurm(argv):
        if argv[0] == "squeue":
            return live + "\n"
        if argv[:3] == ["scontrol", "show", "hostnames"]:
            return "\n".join(argv[3].split(",")) + "\n"
        raise AssertionError(f"doctor must stay read-only, called {argv}")

    return slurm


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    e = Env(tmp_path)
    e.add_run("a")
    e.poll(2)
    e.hid = e.host_up(e.hosts()[0])
    e.poll()
    e.rid = e.rec("a").runtime.runtime_job_id
    return e


def _problems(env, live="a,b"):
    return [line for level, line in pool_ctl.doctor_report(env.layout, _fake_slurm(live)) if level == "PROBLEM"]


def test_a_healthy_pool_has_no_problem(env):
    assert env.assigned(env.hid) == [env.rid]
    report = pool_ctl.doctor_report(env.layout, _fake_slurm("a,b"))
    assert [line for level, line in report if level == "PROBLEM"] == [], report
    assert any(f"run {env.rid} on {env.hid}: need 1" in line and "slack +1" in line for _, line in report)


def test_stale_info_env_names_the_gone_node(env):
    problems = _problems(env, live="a")
    assert any("STALE, gone b" in p for p in problems), problems


def test_two_exclude_lines_in_meta_env(env):
    meta = env.layout.run_dir(env.rid) / "meta.env"
    meta.write_text(meta.read_text() + "EXCLUDE=x1,x2\n")
    assert any("2 EXCLUDE= lines" in p for p in _problems(env)), _problems(env)


def test_a_run_exclude_covering_its_host(env):
    """11-24: the run's own EXCLUDE held almost the whole allocation."""
    meta = env.layout.run_dir(env.rid) / "meta.env"
    meta.write_text(meta.read_text().replace("EXCLUDE=''", "EXCLUDE=a,b", 1))
    problems = _problems(env)
    assert any(f"run {env.rid} on {env.hid}" in p and "slack -1" in p for p in problems), problems


def test_duplicate_live_records_and_an_unreadable_one(env):
    record = env.rec("a")
    env.store.upsert(type(record)(job_id="a_q2", definition=record.definition, runtime=type(record.runtime)()))
    (env.session / "broken.job.json").write_bytes(b"\0" * 64)
    problems = _problems(env)
    assert any("job a: 2 live records ['a', 'a_q2']" in p for p in problems), problems
    assert any("1 unreadable: broken.job.json" in p for p in problems), problems


def test_stale_monitor_and_old_request_fail_the_command(env, capsys):
    old = time.time() - 3600
    os.utime(env.layout.monitor_alive, (old, old))
    req = env.layout.requests_dir / "1.000000-abc.json"
    req.write_text('{"op": "close"}')
    os.utime(req, (old, old))
    args = SimpleNamespace(no_slurm=True)
    assert pool_ctl.cmd_doctor(env.layout, args) == 1
    out = capsys.readouterr().out
    assert "monitor: last tick 3600s ago" in out and "requests: 1 not executed" in out
    assert "live allocation unknown" in out  # --no-slurm: no squeue
