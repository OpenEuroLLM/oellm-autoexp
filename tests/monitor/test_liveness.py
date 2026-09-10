"""Tests for the session liveness/control contract.

Two files in the session directory carry the whole protocol between a monitor
and its (optional) supervisor:

  ``.monitor.alive``  liveness, where the TIMESTAMP IS THE MTIME. Also the
                      lease: it is what stops a second monitor starting on
                      another login node and racing on every job record.
  ``.monitor.stop``   presence means stop. Written by the user, or by the
                      monitor on Ctrl-C -- which is the only thing that
                      distinguishes "was stopped" from "died". SIGTERM
                      deliberately does NOT write it, so a supervisor can clear
                      a wedged monitor without blocking its own restart.
"""

from __future__ import annotations

import json
import os
import signal
import socket
from pathlib import Path

import pytest

from oellm_autoexp.orchestrator import (
    ensure_no_live_monitor,
    heartbeat_path,
    read_heartbeat,
    request_stop,
    stop_path,
    stop_requested,
    write_heartbeat,
)
from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime
from oellm_autoexp.monitor.submission import LocalJobConfig


# --- heartbeat ------------------------------------------------------------


def test_heartbeat_absent_reads_as_none(tmp_path: Path):
    assert read_heartbeat(tmp_path) is None


def test_heartbeat_records_pid_host_and_is_fresh(tmp_path: Path):
    write_heartbeat(tmp_path)
    pid, host, _target, age = read_heartbeat(tmp_path)
    assert pid == os.getpid()
    assert host == socket.gethostname()
    assert age < 5


def test_heartbeat_target_comes_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOEXP_TMUX_TARGET", "autoexp:mon-42")
    write_heartbeat(tmp_path)
    _pid, _host, target, _age = read_heartbeat(tmp_path)
    assert target == "autoexp:mon-42"


def test_heartbeat_survives_a_garbage_file(tmp_path: Path):
    """A truncated or hand-mangled heartbeat must read as 'no heartbeat'."""
    heartbeat_path(tmp_path).write_text("not-a-pid\n")
    assert read_heartbeat(tmp_path) is None


# --- lease ----------------------------------------------------------------


def test_lease_allows_start_when_no_heartbeat(tmp_path: Path):
    ensure_no_live_monitor(tmp_path)  # must not raise


def test_lease_allows_start_for_our_own_heartbeat(tmp_path: Path):
    write_heartbeat(tmp_path)
    ensure_no_live_monitor(tmp_path)  # a restart in place is fine


def test_lease_refuses_a_fresh_foreign_heartbeat(tmp_path: Path):
    heartbeat_path(tmp_path).write_text("999999 otherhost autoexp:mon-1\n")
    with pytest.raises(SystemExit) as excinfo:
        ensure_no_live_monitor(tmp_path)
    message = str(excinfo.value)
    assert "999999" in message and "otherhost" in message
    # The message must say how to get out of it, not just refuse.
    assert str(stop_path(tmp_path)) in message


def test_lease_takes_over_from_a_dead_pid_on_this_host(tmp_path: Path):
    """A crash leaves a FRESH heartbeat naming a pid that no longer exists.

    Checking the pid before the age is what lets a supervisor relaunch
    straight away instead of being locked out by its predecessor's
    corpse for a whole stale window -- which made every restart in the
    end-to-end run fail.
    """
    heartbeat_path(tmp_path).write_text(f"999999 {socket.gethostname()} autoexp:mon-1\n")
    ensure_no_live_monitor(tmp_path)  # must not raise


def test_lease_still_refuses_a_live_pid_on_this_host(tmp_path: Path):
    heartbeat_path(tmp_path).write_text(f"{os.getppid()} {socket.gethostname()} -\n")
    with pytest.raises(SystemExit):
        ensure_no_live_monitor(tmp_path)


def test_lease_ignores_a_stale_foreign_heartbeat(tmp_path: Path):
    heartbeat_path(tmp_path).write_text("999999 otherhost -\n")
    ensure_no_live_monitor(tmp_path, stale_after_s=-1)


def test_lease_force_overrides(tmp_path: Path):
    heartbeat_path(tmp_path).write_text("999999 otherhost -\n")
    ensure_no_live_monitor(tmp_path, force=True)


# --- stop file ------------------------------------------------------------


def test_stop_absent_is_none(tmp_path: Path):
    assert stop_requested(tmp_path) is None


def test_stop_roundtrips_its_reason(tmp_path: Path):
    request_stop(tmp_path, "stopped by SIGINT")
    assert stop_requested(tmp_path) == "stopped by SIGINT"


def test_empty_stop_file_still_stops(tmp_path: Path):
    """`touch .monitor.stop` is the documented way to stop; it has no
    content."""
    stop_path(tmp_path).touch()
    assert stop_requested(tmp_path) == "no reason given"


# --- atomic state writes --------------------------------------------------


def _record(tmp_path: Path, job_id: str = "j") -> JobRecord:
    return JobRecord(
        job_id=job_id,
        definition=LocalJobConfig(name=job_id, command=["true"], log_path=str(tmp_path / "x.log")),
        runtime=JobRuntime(submitted=True, runtime_job_id="1", log_cursor=17),
    )


def test_upsert_leaves_no_partial_file_behind(tmp_path: Path):
    store = JobFileStore(tmp_path / "state")
    store.upsert(_record(tmp_path))
    files = sorted(p.name for p in (tmp_path / "state").iterdir())
    assert files == ["j.job.json"]  # the .tmp must have been renamed away


def test_upsert_replaces_atomically(tmp_path: Path, monkeypatch):
    """A crash between write and rename must leave the OLD record intact.

    Losing a record is not a cosmetic failure: load_all() silently skips files
    it cannot parse, so a torn write drops the job from monitoring with no log
    line at all.
    """
    store = JobFileStore(tmp_path / "state")
    store.upsert(_record(tmp_path))
    original = (tmp_path / "state" / "j.job.json").read_text()

    updated = _record(tmp_path)
    updated.runtime.log_cursor = 999

    def boom(*_args, **_kwargs):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.upsert(updated)

    # Old content still readable and parseable.
    assert (tmp_path / "state" / "j.job.json").read_text() == original
    assert json.loads(original)["runtime"]["log_cursor"] == 17
    assert len(store.load_all()) == 1


# --- signals ---------------------------------------------------------------


class _NoopClient:
    """Reports nothing, so the job stays active and the loop keeps going."""

    def squeue(self) -> dict[str, str]:
        return {}

    def remove(self, job_id: str) -> None:
        pass

    def cancel(self, job_id: str) -> None:
        pass


def _loop_with_one_active_job(tmp_path: Path, signum: int):
    from oellm_autoexp.monitor.loop import MonitorLoop

    store = JobFileStore(tmp_path / "state")
    store.upsert(_record(tmp_path))
    loop = MonitorLoop(
        store, local_client=_NoopClient(), poll_interval_seconds=0.05, show_poll_state=False
    )

    def fake_observe():
        os.kill(os.getpid(), signum)

    loop.observe_once = fake_observe  # type: ignore[method-assign]
    return loop, store


def test_sigint_records_the_stop(tmp_path: Path):
    """Ctrl-C is a human taking over, so a supervisor must stand down."""
    from oellm_autoexp.orchestrator import run_loop

    loop, store = _loop_with_one_active_job(tmp_path, signal.SIGINT)
    assert run_loop(loop) == "signal"
    assert stop_requested(store.root) == "stopped by SIGINT (Ctrl-C)"


def test_sigterm_does_not_record_the_stop(tmp_path: Path):
    """SIGTERM is how a supervisor clears a WEDGED monitor before restarting
    it.

    If it recorded a stop, the supervisor would block its own relaunch -- which
    is exactly what happened in the first end-to-end run: the wedged monitor was
    terminated, wrote .monitor.stop, and every following launch was refused.
    """
    from oellm_autoexp.orchestrator import run_loop

    loop, store = _loop_with_one_active_job(tmp_path, signal.SIGTERM)
    assert run_loop(loop) == "signal"
    assert stop_requested(store.root) is None


def test_a_pre_existing_stop_file_ends_the_loop_before_any_poll(tmp_path: Path):
    from oellm_autoexp.orchestrator import run_loop
    from oellm_autoexp.monitor.loop import MonitorLoop

    store = JobFileStore(tmp_path / "state")
    store.upsert(_record(tmp_path))
    request_stop(store.root, "by hand")

    polled = []
    loop = MonitorLoop(store, local_client=_NoopClient(), poll_interval_seconds=0.05)
    loop.observe_once = lambda: polled.append(1)  # type: ignore[method-assign]

    assert run_loop(loop) == "stopped"
    assert polled == []


def test_tmp_files_are_not_loaded_as_jobs(tmp_path: Path):
    store = JobFileStore(tmp_path / "state")
    store.upsert(_record(tmp_path))
    (tmp_path / "state" / "j.job.tmp").write_text("{ truncated")
    assert [j.job_id for j in store.load_all()] == ["j"]
