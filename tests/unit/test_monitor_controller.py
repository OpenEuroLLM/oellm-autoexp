from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from oellm_autoexp.monitor.controller import (
    JobRegistration,
    MonitorController,
    MonitorCycleResult,
    MonitorRecord,
)
from oellm_autoexp.monitor.policy import NoRestartPolicyConfig, NoRestartPolicy
from oellm_autoexp.monitor.watcher import NullMonitor, NullMonitorConfig
from oellm_autoexp.persistence.state_store import MonitorStateStore
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.workflow import host as host_runtime


@pytest.fixture
def controller(tmp_path: Path) -> MonitorController:
    monitor = NullMonitor(NullMonitorConfig())
    policies = {
        "crash": NoRestartPolicy(NoRestartPolicyConfig()),
        "success": NoRestartPolicy(NoRestartPolicyConfig(message="done")),
    }
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    state_store = MonitorStateStore(tmp_path / "state")
    return MonitorController(monitor, slurm, policies, state_store=state_store)


def test_stop_action_removed_from_monitoring(controller: MonitorController, tmp_path: Path) -> None:
    slurm = controller._slurm  # type: ignore[attr-defined]
    script_path = tmp_path / "demo.sbatch"
    log_path = tmp_path / "demo.log"
    job_id = slurm.submit("demo", str(script_path), str(log_path))

    controller.register_job(
        job_id,
        JobRegistration(
            name="demo",
            script_path=str(script_path),
            log_path=str(log_path),
        ),
    )

    decision = controller.handle_state_change(job_id, "crash")
    assert decision.action == "stop"
    assert not list(controller.jobs())


def test_monitor_loop_writes_single_action(tmp_path: Path, monkeypatch) -> None:
    class DummyMonitor:
        def __init__(self) -> None:
            self.config = type("Cfg", (), {"check_interval_seconds": 0})()

    class DummyController:
        def __init__(self) -> None:
            self._done = False
            self._record = MonitorRecord(
                job_id="1",
                job_name="demo",
                event="manual",
                state=None,
                action="notify",
                payload={"message": "hello"},
                metadata={"signal": "sig"},
            )

        def jobs(self):
            return [] if self._done else [object()]

        async def observe_once(self):
            self._done = True
            return MonitorCycleResult(events=[self._record])

        def drain_events(self):
            return [self._record]

        def clear_state(self):
            self.cleared = True

    async def fake_sleep(_):  # pragma: no cover - simple stub
        return None

    monkeypatch.setattr(host_runtime.asyncio, "sleep", fake_sleep)

    controller = DummyController()
    monitor = DummyMonitor()

    asyncio.run(host_runtime._monitor_loop(controller, monitor, tmp_path))

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
