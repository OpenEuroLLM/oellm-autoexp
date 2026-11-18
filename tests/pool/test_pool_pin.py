"""pool_ctl pin / unpin: a queued run can be held to one host.

The pin is ``runs/<rid>/pin_host``, which the monitor reads fresh on every pass. A record field
would be clobbered by the monitor's own writes of the job record, so the marker file is the
mechanism. The assignment loop then skips every host except the pinned one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from oellm_autoexp.monitor.pool import PoolLayout, _read_pin, host_allows_pin  # noqa: E402
from tests.pool.test_pool_unit import Env  # noqa: E402

spec = importlib.util.spec_from_file_location("pool_ctl", REPO / "scripts" / "pool_ctl.py")
pool_ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool_ctl)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    e = Env(tmp_path)
    e.add_run("a")
    e.poll(2)
    e.hid = e.host_up(e.hosts()[0])
    e.poll()
    return e


def _ctl(env, *args):
    return pool_ctl.main([str(env.layout.root), *args])


def test_host_allows_pin():
    assert host_allows_pin("", "host001")  # not pinned: any host
    assert host_allows_pin("host001", "host001")
    assert not host_allows_pin("host001", "host002")


def test_read_pin(tmp_path):
    layout = PoolLayout(tmp_path)
    assert _read_pin(layout, "p.0001") == ""
    rd = layout.run_dir("p.0001")
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "pin_host").write_text("host002\n")
    assert _read_pin(layout, "p.0001") == "host002"
    (rd / "pin_host").write_text("host002")  # no trailing newline
    assert _read_pin(layout, "p.0001") == "host002"


def test_pin_and_unpin_write_the_marker(env, capsys):
    rid = env.rec("a").runtime.runtime_job_id
    assert _ctl(env, "pin", rid, env.hid) == 0
    assert "pinned to" in capsys.readouterr().out
    assert (env.layout.run_dir(rid) / "pin_host").read_text().strip() == env.hid
    assert _ctl(env, "unpin", rid) == 0
    assert "pin cleared" in capsys.readouterr().out
    assert not (env.layout.run_dir(rid) / "pin_host").exists()


def test_unpin_without_a_pin_is_not_an_error(env, capsys):
    rid = env.rec("a").runtime.runtime_job_id
    assert _ctl(env, "unpin", rid) == 0
    assert "pin cleared" in capsys.readouterr().out


def test_pin_to_an_unknown_host_is_refused(env, capsys):
    rid = env.rec("a").runtime.runtime_job_id
    assert _ctl(env, "pin", rid, "host999") == 1
    assert "no such host: host999" in capsys.readouterr().err
    assert not (env.layout.run_dir(rid) / "pin_host").exists()


def test_pin_of_an_unknown_run_is_refused(env, capsys):
    assert _ctl(env, "pin", "p.9999", env.hid) == 1
    assert "nothing to pin" in capsys.readouterr().err


def test_a_pinned_run_still_takes_its_host(env):
    """The guard must not lock a run out of the host it is pinned to."""
    rid = env.rec("a").runtime.runtime_job_id
    assert _ctl(env, "pin", rid, env.hid) == 0
    env.poll()
    assert env.assigned(env.hid) == [rid]


def test_a_pinned_run_waits_for_its_host(tmp_path, monkeypatch):
    """A free host that is not the pinned one does not take the run; the pinned one does."""
    monkeypatch.chdir(REPO)
    e = Env(tmp_path, keep_pending=1)
    e.add_run("a")
    e.poll(2)
    first, second = (h.definition.metadata["pool_host_id"] for h in e.hosts()[:2])
    rid = e.rec("a").runtime.runtime_job_id
    assert _ctl(e, "pin", rid, second) == 0
    e.host_up(e.hosts()[0])
    e.poll()
    assert e.assigned(first) == []
    e.host_up(e.hosts()[1])
    e.poll()
    assert e.assigned(second) == [rid]
    assert e.assigned(first) == []
