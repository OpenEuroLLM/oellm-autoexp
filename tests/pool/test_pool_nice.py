"""pool_ctl park / promote / queue and the pool's submit_nice, against a fake SLURM queue.

The fake follows what docs/mn5_renice.md measured on MareNostrum 5: priority = natural priority
minus Nice, floored at 1; Nice=0 gives the natural priority back; nothing else changes.
"""

from __future__ import annotations

import importlib.util

import pytest

from tests.pool.test_pool_unit import REPO, Env

spec = importlib.util.spec_from_file_location("pool_ctl", REPO / "scripts" / "pool_ctl.py")
pool_ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool_ctl)


class _FakeQueue:
    """squeue / scontrol update for a few jobs: {job id: [state, nice, natural priority, name]}."""

    def __init__(self, jobs, refuse=()):
        self.jobs = {j: list(v) for j, v in jobs.items()}
        self.refuse = set(refuse)  # updates SLURM silently ignores (permission, ...)
        self.updates = []

    def prio(self, jid):
        _, nice, base, _ = self.jobs[jid]
        return max(1, base - nice)

    def __call__(self, argv):
        if argv[:2] == ["scontrol", "update"]:
            fields = dict(a.split("=", 1) for a in argv[2:])
            self.updates.append(fields)
            if fields["JobId"] not in self.refuse:
                self.jobs[fields["JobId"]][1] = int(fields["Nice"])
            return ""
        if argv[0] == "squeue" and "-j" in argv:
            ids = argv[argv.index("-j") + 1].split(",")
            return "".join(
                f"{j}|{self.jobs[j][0]}|{self.jobs[j][1]}|{self.prio(j)}\n" for j in ids if j in self.jobs
            )
        if argv[0] == "squeue" and "--me" in argv:
            state = argv[argv.index("-t") + 1]
            rows = [(j, v) for j, v in self.jobs.items() if v[0] == {"PD": "PENDING", "R": "RUNNING"}[state]]
            if state == "R":
                return "".join("cns144|acc_xehpc\n" for _ in rows)
            return "".join(
                f"{j}|{v[3]}|{v[1]}|{self.prio(j)}|cns144|acc_xehpc|264|QOSMaxJobsPerUserLimit\n"
                for j, v in rows
            )
        raise AssertionError(argv)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Two hosts queued (keep_pending=1): the first runs, the second waits."""
    monkeypatch.chdir(REPO)
    e = Env(tmp_path, keep_pending=1)
    e.add_run("a")
    e.poll(3)  # host001 submitted on the 2nd pass, its pending successor host002 on the 3rd
    hosts = e.hosts()
    assert len(hosts) == 2 and all(h.runtime.runtime_job_id for h in hosts)
    e.h_run, e.h_wait = (h.definition.metadata["pool_host_id"] for h in hosts)
    e.j_run, e.j_wait = (str(h.runtime.runtime_job_id) for h in hosts)
    e.host_up(hosts[0])
    e.poll()
    e.q = _FakeQueue({e.j_run: ["RUNNING", 0, 95000, "p_host001"],
                      e.j_wait: ["PENDING", 0, 95101, "p_host002"]})
    return e


def _nice(env, cmd, *hosts, apply=False, nice=pool_ctl.PARK_NICE):
    args = type("A", (), {"cmd": cmd, "hosts": list(hosts), "apply": apply, "nice": nice})()
    return pool_ctl.cmd_nice(env.layout, args, env.q)


def _ledger(env):
    return env.layout.ledger.read_text() if env.layout.ledger.exists() else ""


def test_park_is_a_plan_unless_applied(env, capsys):
    assert _nice(env, "park") == 0
    out = capsys.readouterr().out
    assert f"job {env.j_wait}: nice 0 -> 250000  priority 95101 -> ~1" in out
    assert f"job {env.j_run}: RUNNING, left alone" in out
    assert "submit_nice 0 -> 250000" in out and "plan only" in out
    assert env.q.updates == []
    assert env.layout.load_config().submit_nice == 0
    assert not (env.layout.host_dir(env.h_wait) / "control" / "parked").exists()


def test_park_and_promote_the_whole_pool(env, capsys):
    assert _nice(env, "park", apply=True) == 0
    assert env.q.updates == [{"JobId": env.j_wait, "Nice": "250000"}]  # the running host untouched
    assert env.q.jobs[env.j_wait][1] == 250000 and env.q.prio(env.j_wait) == 1
    marker = env.layout.host_dir(env.h_wait) / "control" / "parked"
    assert marker.read_text().startswith("nice=250000 ")
    assert env.layout.load_config().submit_nice == 250000
    assert f"{env.h_wait}\t{env.j_wait}\t-\tpark nice 0 -> 250000, priority 95101 -> 1" in _ledger(env)
    # parking twice changes nothing
    assert _nice(env, "park", apply=True) == 0 and len(env.q.updates) == 1
    capsys.readouterr()
    assert _nice(env, "promote") == 0
    assert "priority 1 -> its natural priority (hidden while floored at 1)" in capsys.readouterr().out
    assert _nice(env, "promote", apply=True) == 0
    assert env.q.prio(env.j_wait) == 95101, "Nice=0 gives the natural priority back"
    assert not marker.exists()
    assert env.layout.load_config().submit_nice == 0
    assert "promote nice 250000 -> 0, priority 1 -> 95101" in _ledger(env)


def test_named_hosts_leave_submit_nice_alone(env):
    assert _nice(env, "park", env.h_wait, apply=True, nice=1000) == 0
    assert env.q.prio(env.j_wait) == 94101
    assert env.layout.load_config().submit_nice == 0


def test_an_update_slurm_did_not_take_is_an_error(env, capsys):
    env.q.refuse.add(env.j_wait)
    assert _nice(env, "park", env.h_wait, apply=True) == 1
    assert "nice is 0 after the update, 250000 expected" in capsys.readouterr().err
    assert not (env.layout.host_dir(env.h_wait) / "control" / "parked").exists()
    assert "park" not in _ledger(env)


def test_unknown_hosts_and_negative_nice_are_refused(env, capsys):
    assert _nice(env, "park", "host999", apply=True) == 1
    assert "no open SLURM job for host(s): host999" in capsys.readouterr().err
    assert _nice(env, "park", apply=True, nice=-5) == 1
    assert "needs root" in capsys.readouterr().err
    assert env.q.updates == []


def test_no_answer_from_squeue_changes_nothing(env, capsys):
    env.q.jobs.clear()
    assert _nice(env, "park", apply=True) == 1
    assert "no answer from squeue" in capsys.readouterr().err
    assert env.layout.load_config().submit_nice == 0


def test_cli_park_parses_and_plans(env, capsys, monkeypatch):
    monkeypatch.setattr(pool_ctl, "_slurm", env.q)
    monkeypatch.setattr(pool_ctl.cmd_nice, "__defaults__", (env.q,))
    assert pool_ctl.main([str(env.layout.root), "park", env.h_wait]) == 0
    assert "plan only" in capsys.readouterr().out and env.q.updates == []


def test_queue_lists_pending_jobs_in_slurm_order(env, capsys):
    env.q.jobs["999"] = ["PENDING", 250000, 96000, "other_host"]
    assert pool_ctl.cmd_queue(env.layout, None, env.q) == 0
    out = capsys.readouterr().out
    assert "cns144 / acc_xehpc: 2 pending (1 parked), 1 running" in out
    lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
    assert lines[0].startswith(f"  *  {env.j_wait}"), lines  # this pool's, not parked, first
    assert lines[1].startswith("   P 999"), lines


def test_new_hosts_are_submitted_with_submit_nice(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    e = Env(tmp_path, keep_pending=1, submit_nice=250000)
    e.add_run("a")
    e.poll(3)
    for host in e.hosts():
        script = e.layout.host_dir(host.definition.metadata["pool_host_id"]) / "host.sbatch"
        assert "#SBATCH --nice=250000" in script.read_text()


def test_no_nice_line_by_default(env):
    script = env.layout.host_dir(env.h_wait) / "host.sbatch"
    assert "--nice" not in script.read_text()


def test_a_parked_host_that_starts_loses_its_marker(env):
    marker = env.layout.host_dir(env.h_run) / "control" / "parked"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("nice=250000 ts=1\n")
    env.poll()
    assert not marker.exists()
    assert "started while parked (nice=250000 ts=1)" in _ledger(env)
