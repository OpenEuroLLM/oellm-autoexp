"""job_watchdog.py cancels a stuck run only through pool_ctl (the pool RUN, never the SLURM job:
inside a pool that is the whole allocation); outside a pool it only reports."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/korbi/hybrid_speed/job_watchdog.py"


@pytest.fixture
def watchdog():
    spec = importlib.util.spec_from_file_location("job_watchdog", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outside_a_pool_nothing_is_cancelled(watchdog, monkeypatch):
    monkeypatch.delenv("AUTOEXP_POOL_DIR", raising=False)
    monkeypatch.delenv("AUTOEXP_RUN_ID", raising=False)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    assert "not a pool run" in watchdog.cancel_pool_run()
    assert calls == []


def test_a_pool_run_is_cancelled_via_pool_ctl(watchdog, monkeypatch):
    monkeypatch.setenv("AUTOEXP_POOL_DIR", "/pools/mn5_pool_x")
    monkeypatch.setenv("AUTOEXP_RUN_ID", "mn5-pool-x.0042")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="cancelling job_1\n  monitor: cancelled\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = watchdog.cancel_pool_run()
    (cmd,) = calls
    assert cmd[-3:] == ["/pools/mn5_pool_x", "cancel", "mn5-pool-x.0042"]
    assert any(str(c).endswith("pool_ctl.py") for c in cmd)
    assert "scancel" not in " ".join(map(str, cmd))
    assert "rc 0" in out


def test_the_script_has_no_scancel_call():
    code = "\n".join(
        line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    assert '"scancel"' not in code and "'scancel'" not in code
