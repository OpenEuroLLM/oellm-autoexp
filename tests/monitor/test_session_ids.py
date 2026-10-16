"""New monitoring sessions never share a directory, even when started in the same second."""

from pathlib import Path
from types import SimpleNamespace

from oellm_autoexp import orchestrator


def test_sessions_in_the_same_second_get_distinct_dirs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(orchestrator.time, "time", lambda: 1_700_000_000.5)
    plan = SimpleNamespace(config_setup=SimpleNamespace(monitor_state_dir=str(tmp_path)))
    ids = [orchestrator._ensure_state_store(plan)[1] for _ in range(3)]
    assert ids == ["1700000000", "1700000001", "1700000002"]
    assert all((tmp_path / i).is_dir() for i in ids)


def test_an_explicit_session_is_reused(tmp_path: Path):
    plan = SimpleNamespace(config_setup=SimpleNamespace(monitor_state_dir=str(tmp_path)))
    (tmp_path / "42").mkdir()
    assert orchestrator._ensure_state_store(plan, session_id="42")[1] == "42"
