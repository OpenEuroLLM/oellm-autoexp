"""Which session directory scripts/monitor_autoexp.py monitors."""

import pytest

from scripts import monitor_autoexp


@pytest.fixture
def monitored_dir(monkeypatch):
    """Run main() without monitoring; return the store dir it would monitor."""

    def run(argv):
        seen = []
        monkeypatch.setattr(monitor_autoexp, "run_loop", lambda loop: seen.append(loop._store.root))
        monitor_autoexp.main(argv)
        return seen[0]

    return run


def test_session_is_looked_up_in_monitor_state_dir(tmp_path, monitored_dir):
    (tmp_path / "my-session").mkdir()
    argv = ["--session", "my-session", "--monitor-state-dir", str(tmp_path)]
    assert monitored_dir(argv) == tmp_path / "my-session"


def test_session_dir_takes_precedence(tmp_path, monitored_dir):
    argv = [
        "--session",
        "ignored",
        "--session-dir",
        str(tmp_path),
        "--monitor-state-dir",
        "/nonexistent",
    ]
    assert monitored_dir(argv) == tmp_path
