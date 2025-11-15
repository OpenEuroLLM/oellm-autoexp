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
from oellm_autoexp.monitor.watcher import NullMonitor, NullMonitorConfig
from oellm_autoexp.persistence.state_store import MonitorStateStore
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
from oellm_autoexp.workflow import host as host_runtime


@pytest.fixture
def controller(tmp_path: Path) -> MonitorController:
    monitor = NullMonitor(NullMonitorConfig())
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    state_store = MonitorStateStore(tmp_path / "state")
    return MonitorController(monitor, slurm, state_store=state_store)


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


def test_restore_jobs_re_registers_slurm_client(tmp_path: Path) -> None:
    monitor = NullMonitor(NullMonitorConfig())
    state_dir = tmp_path / "state"
    state_store = MonitorStateStore(state_dir)
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    controller = MonitorController(monitor, slurm, state_store=state_store)

    script_path = tmp_path / "demo.sbatch"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "demo_%j.log"

    job_id = slurm.submit("demo", str(script_path), str(log_path))
    controller.register_job(
        job_id,
        JobRegistration(
            name="demo",
            script_path=str(script_path),
            log_path=str(log_path),
        ),
    )

    stored = state_store.load()[job_id]
    assert stored.resolved_log_path is not None
    assert stored.resolved_log_path.endswith("demo_1.log")

    fresh_slurm = FakeSlurmClient(FakeSlurmClientConfig())
    fresh_monitor = NullMonitor(NullMonitorConfig())
    fresh_controller = MonitorController(
        fresh_monitor,
        fresh_slurm,
        state_store=state_store,
    )
    runtime = host_runtime.HostRuntime(
        manifest=object(),
        monitor=fresh_monitor,
        slurm_client=fresh_slurm,
        state_store=state_store,
        action_queue_dir=tmp_path / "actions",
    )

    restored = host_runtime.restore_jobs(runtime, fresh_controller)
    assert "demo" in restored

    queue = fresh_slurm.squeue()
    assert job_id in queue
    assert queue[job_id] == "PENDING"
    assert fresh_slurm.job_ids_by_name("demo") == [job_id]
    assert any(state.job_id == job_id for state in fresh_controller.jobs())


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
