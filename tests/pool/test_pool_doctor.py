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
    args = SimpleNamespace(no_slurm=True, probe=False)
    assert pool_ctl.cmd_doctor(env.layout, args) == 1
    out = capsys.readouterr().out
    assert "monitor: last tick 3600s ago" in out and "requests: 1 not executed" in out
    assert "live allocation unknown" in out  # --no-slurm: no squeue


# ---------------------------------------------------------------- periodic health check
def test_the_monitor_runs_the_health_check_and_logs_a_new_problem_once(env, caplog):
    import logging

    from oellm_autoexp.monitor.pool_doctor import read_health

    meta = env.layout.run_dir(env.rid) / "meta.env"
    meta.write_text(meta.read_text() + "EXCLUDE=x1\n")  # an appending writer
    env.manager = env.loop._pool
    env.manager._health_ts.clear()
    with caplog.at_level(logging.WARNING):
        env.poll()
    first, n, age = read_health(env.layout)
    assert n == 1 and age < 60 and "1 problem(s)" in first
    assert sum("2 EXCLUDE= lines" in r.message for r in caplog.records) == 1
    caplog.clear()
    env.manager._health_ts.clear()  # next check due: the same problem is not logged again
    with caplog.at_level(logging.WARNING):
        env.poll()
    assert not any("health:" in r.message for r in caplog.records)
    assert "monitor:" not in (env.layout.root / "health.txt").read_text()


def test_the_health_check_is_throttled_and_can_be_off(env):
    env.loop._pool._health_ts.clear()
    env.poll()
    stamp = (env.layout.root / "health.txt").stat().st_mtime_ns
    env.poll(2)
    assert (env.layout.root / "health.txt").stat().st_mtime_ns == stamp, "once per health_check_s"
    meta = env.layout.load_meta()
    meta["config"]["health_check_s"] = 0
    env.layout.write_meta(meta)
    (env.layout.root / "health.txt").unlink()
    env.loop._pool._health_ts.clear()
    env.poll()
    assert not (env.layout.root / "health.txt").exists()


def test_status_shows_the_last_health_check(env, capsys):
    env.loop._pool._health_ts.clear()
    env.poll()
    assert pool_ctl.main([str(env.layout.root), "status"]) == 0
    assert "health: 0 problem(s), checked" in capsys.readouterr().out


# ---------------------------------------------------------------- probe
def _probe_answers(answers):
    calls = []

    def probe(jid, node):
        calls.append((jid, node))
        return answers[node]

    return probe, calls


def test_probe_flags_missing_gpus_ib_below_the_norm_and_silent_nodes(env):
    from oellm_autoexp.monitor.pool_doctor import probe_report

    answers = {"a": "GPUS 4\nIB 5/6\n", "b": "GPUS 3\nIB 5/6\n", "c": "GPUS 4\nIB 4/6\n",
               "d": "srun: error: Unable to create step\n", "e": "GPUS 4\nIB 5/6\n"}
    probe, calls = _probe_answers(answers)
    report = probe_report(env.layout, gpus=4, slurm=_fake_slurm("a,b,c,d,e"), probe=probe)
    text = "\n".join(line for _, line in report)
    assert sorted(n for _, n in calls) == ["a", "b", "c", "d", "e"], "one step per node"
    assert f"host {env.hid} job" in text and "probed 5 node(s), 3 bad, IB norm 5 ACTIVE" in text
    assert "b: 3 GPU(s), expected 4" in text
    assert "c: IB 4/6 ACTIVE, host norm 5" in text
    assert "d: no answer (srun: error: Unable to create step)" in text
    assert "a:" not in text and "e:" not in text
    assert "c gpus=4 ib=4/6" in (env.layout.host_dir(env.hid) / "doctor_probe.txt").read_text()


def test_probe_skips_a_frozen_host(env):
    from oellm_autoexp.monitor.pool_doctor import probe_report

    (env.layout.host_dir(env.hid) / "control").mkdir(exist_ok=True)
    (env.layout.host_dir(env.hid) / "control" / "freeze").write_text("x")
    probe, calls = _probe_answers({})
    report = probe_report(env.layout, slurm=_fake_slurm("a,b"), probe=probe)
    assert calls == [] and "frozen (resize prepared), not probed" in report[0][1]


def test_doctor_probe_through_pool_ctl(env, monkeypatch, capsys):
    import oellm_autoexp.monitor.pool_doctor as doc

    monkeypatch.setattr(doc, "probe_node", lambda jid, node: "GPUS 4\nIB 5/6\n")
    args = SimpleNamespace(no_slurm=False, probe=True, hosts=None, gpus=0, parallel=2)
    assert pool_ctl.cmd_doctor(env.layout, args, slurm=_fake_slurm("a,b")) == 0
    assert "probed 2 node(s), 0 bad" in capsys.readouterr().out
