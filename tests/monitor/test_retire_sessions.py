"""Tests for closing out sessions whose jobs are long gone.

A monitor killed by a login-node reboot never writes ``final_state``, so its
records stay "active" forever: 367 of 506 session directories on JUPITER were in
that state. ``scripts/retire_sessions.py`` resolves them against sacct instead
of flattening them, and must never touch a session that is still live.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime
from oellm_autoexp.monitor.submission import LocalJobConfig

_SPEC = importlib.util.spec_from_file_location(
    "retire_sessions",
    Path(__file__).resolve().parents[2] / "scripts" / "retire_sessions.py",
)
retire = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(retire)


def _session(root: Path, name: str, *, job_id: str | None, final_state=None) -> Path:
    d = root / name
    store = JobFileStore(d)
    store.upsert(
        JobRecord(
            job_id="j",
            definition=LocalJobConfig(name="j", command=["true"], log_path=str(d / "j.log")),
            runtime=JobRuntime(
                submitted=job_id is not None,
                runtime_job_id=job_id,
                final_state=final_state,
            ),
        )
    )
    return d


def _runtime(session: Path) -> dict:
    return json.loads((session / "j.job.json").read_text())["runtime"]


@pytest.fixture
def no_slurm(monkeypatch):
    """Default: no queue, and sacct knows nothing (the local-dev situation)."""
    monkeypatch.setattr(retire, "queued_job_ids", lambda: set())
    monkeypatch.setattr(retire, "sacct_states", lambda ids: {})


def test_normalize_strips_the_cancelling_uid():
    assert retire._normalize("CANCELLED by 29685") == "CANCELLED"
    assert retire._normalize("COMPLETED") == "COMPLETED"
    assert retire._normalize("") == ""


def test_dry_run_writes_nothing(tmp_path, no_slurm, capsys):
    s = _session(tmp_path, "old", job_id="111")
    assert retire.main(["--state-dir", str(tmp_path)]) == 0
    assert _runtime(s)["final_state"] is None
    assert "Dry run" in capsys.readouterr().out


def test_resolves_each_outcome_from_sacct(tmp_path, monkeypatch):
    done = _session(tmp_path, "done", job_id="1")
    killed = _session(tmp_path, "killed", job_id="2")
    forgotten = _session(tmp_path, "forgotten", job_id="3")
    never = _session(tmp_path, "never", job_id=None)

    monkeypatch.setattr(retire, "queued_job_ids", lambda: set())
    monkeypatch.setattr(
        retire,
        "sacct_states",
        lambda ids: {
            "1": ("COMPLETED", "2026-08-14T10:38:04"),
            "2": ("CANCELLED", "2026-08-14T17:53:14"),
            # "3" deliberately absent: sacct retention expired
        },
    )
    assert retire.main(["--state-dir", str(tmp_path), "--apply"]) == 0

    assert _runtime(done)["final_state"] == "finished"
    assert _runtime(killed)["final_state"] == "cancelled"
    assert _runtime(forgotten)["final_state"] == "retired"
    assert _runtime(never)["final_state"] == "retired"


def test_a_queued_job_is_never_retired(tmp_path, monkeypatch):
    """The safety property: a live run must not be closed out under its monitor."""
    live = _session(tmp_path, "live", job_id="1524558")
    dead = _session(tmp_path, "dead", job_id="1")

    monkeypatch.setattr(retire, "queued_job_ids", lambda: {"1524558"})
    monkeypatch.setattr(retire, "sacct_states", lambda ids: {"1": ("FAILED", "")})
    assert retire.main(["--state-dir", str(tmp_path), "--apply"]) == 0

    assert _runtime(live)["final_state"] is None
    assert _runtime(dead)["final_state"] == "cancelled"


def test_a_job_still_running_per_sacct_is_left_alone(tmp_path, monkeypatch):
    s = _session(tmp_path, "running", job_id="9")
    monkeypatch.setattr(retire, "queued_job_ids", lambda: set())
    monkeypatch.setattr(retire, "sacct_states", lambda ids: {"9": ("RUNNING", "Unknown")})
    retire.main(["--state-dir", str(tmp_path), "--apply"])
    assert _runtime(s)["final_state"] is None


def test_already_finished_records_are_untouched(tmp_path, no_slurm):
    s = _session(tmp_path, "done", job_id="1", final_state="finished")
    before = (s / "j.job.json").read_text()
    retire.main(["--state-dir", str(tmp_path), "--apply"])
    assert (s / "j.job.json").read_text() == before


def test_end_ts_is_backfilled_from_sacct(tmp_path, monkeypatch):
    s = _session(tmp_path, "done", job_id="1")
    monkeypatch.setattr(retire, "queued_job_ids", lambda: set())
    monkeypatch.setattr(
        retire, "sacct_states", lambda ids: {"1": ("COMPLETED", "2026-08-14T10:38:04")}
    )
    retire.main(["--state-dir", str(tmp_path), "--apply"])
    assert _runtime(s)["end_ts"] == pytest.approx(1786775884.0, abs=86400)


def test_retired_records_still_parse_and_leave_the_active_set(tmp_path, monkeypatch):
    """The failure mode that matters: an unknown field would make load_all()
    skip the record SILENTLY, i.e. the job disappears instead of erroring."""
    s = _session(tmp_path, "old", job_id="1")
    monkeypatch.setattr(retire, "queued_job_ids", lambda: set())
    monkeypatch.setattr(retire, "sacct_states", lambda ids: {"1": ("FAILED", "")})
    retire.main(["--state-dir", str(tmp_path), "--apply"])

    store = JobFileStore(s)
    assert len(store.load_all(include_finished=True)) == 1, "record became unparseable"
    assert store.load_all() == [], "retired record is still counted as active"
    job = store.load_all(include_finished=True)[0]
    assert job.runtime.action_state["retired"]["slurm_state"] == "FAILED"


def test_unreadable_records_are_reported_not_rewritten(tmp_path, no_slurm, capsys):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "j.job.json").write_text("{ truncated")
    retire.main(["--state-dir", str(tmp_path), "--apply"])
    assert (d / "j.job.json").read_text() == "{ truncated"
    assert "unreadable" in capsys.readouterr().out


def test_single_session_mode(tmp_path, monkeypatch):
    s = _session(tmp_path, "one", job_id="1")
    _session(tmp_path, "two", job_id="2")
    monkeypatch.setattr(retire, "queued_job_ids", lambda: set())
    monkeypatch.setattr(retire, "sacct_states", lambda ids: {})
    retire.main(["--session-dir", str(s), "--apply"])
    assert _runtime(s)["final_state"] == "retired"
    assert _runtime(tmp_path / "two")["final_state"] is None
