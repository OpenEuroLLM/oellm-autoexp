import asyncio
import time
from pathlib import Path
from typing import Any, Dict

from oellm_autoexp.monitor.controller import JobRegistration, MonitorController
from oellm_autoexp.monitor.policy import AlwaysRestartPolicy, AlwaysRestartPolicyConfig
from oellm_autoexp.monitor.policy import NoRestartPolicy, NoRestartPolicyConfig
from oellm_autoexp.monitor.watcher import (
    NullMonitor,
    NullMonitorConfig,
    SlurmLogMonitor,
    SlurmLogMonitorConfig,
    LogSignalConfig,
    MonitoredJob,
    default_log_signals,
)
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig


def test_monitor_controller_restart_increments():
    monitor = NullMonitor(NullMonitorConfig())
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "stall": AlwaysRestartPolicy(AlwaysRestartPolicyConfig(max_retries=2)),
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
    }
    controller = MonitorController(monitor, slurm, policies)

    script = Path("script")
    log = Path("log")
    job_id = slurm.submit("demo", script, log)
    controller.register_job(job_id, JobRegistration(name="demo", script_path=script, log_path=log))

    decision = controller.handle_state_change(job_id, "stall")
    assert decision.action == "restart"
    job_states = list(controller.jobs())
    assert len(job_states) == 1
    assert job_states[0].attempts == 2
    assert job_states[0].job_id != job_id
    new_job_id = job_states[0].job_id

    decision = controller.handle_state_change(new_job_id, "missing-mode")
    assert decision.action == "stop"


def test_slurm_log_monitor_detects_stall(tmp_path: Path) -> None:
    config = SlurmLogMonitorConfig(inactivity_threshold_seconds=1, check_interval_seconds=1)
    monitor = SlurmLogMonitor(config)
    log_path = tmp_path / "job.log"
    log_path.write_text("initial\n")

    job = MonitoredJob(job_id=1, name="demo", log_path=log_path, check_interval_seconds=1)

    outcomes = asyncio.run(monitor.watch([job]))
    assert outcomes[1].status == "active"

    time.sleep(1.1)
    outcomes = asyncio.run(monitor.watch([job]))
    assert outcomes[1].status == "stall"


def test_monitor_controller_observe_restart(tmp_path: Path) -> None:
    monitor = SlurmLogMonitor(SlurmLogMonitorConfig(inactivity_threshold_seconds=0, check_interval_seconds=1))
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "stall": AlwaysRestartPolicy(AlwaysRestartPolicyConfig(max_retries=2)),
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
        "timeout": NoRestartPolicy(NoRestartPolicyConfig(message="timeout")),
    }
    controller = MonitorController(monitor, slurm, policies)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("start\n")

    job_id = slurm.submit("demo", script, log)
    controller.register_job(
        job_id,
        JobRegistration(name="demo", script_path=script, log_path=log),
    )

    async def _run() -> Dict[int, Any]:
        await controller.observe_once()
        result = await controller.observe_once()
        return result.decisions

    decisions = asyncio.run(_run())

    assert any(dec.action == "restart" for dec in decisions.values())
    job_states = list(controller.jobs())
    assert len(job_states) == 1
    assert job_states[0].attempts == 2


def test_slurm_monitor_defaults_populated():
    monitor = SlurmLogMonitor(SlurmLogMonitorConfig())
    names = {signal.name for signal in monitor.config.log_signals}
    assert {"error", "checkpoint", "training_complete"}.issubset(names)
    defaults = default_log_signals()
    assert defaults[0].mode == "crash"


def test_monitor_signal_triggers_action(tmp_path: Path) -> None:
    config = SlurmLogMonitorConfig(
        inactivity_threshold_seconds=0,
        check_interval_seconds=1,
        log_signals=[
            LogSignalConfig(
                name="checkpoint",
                pattern=r"Checkpoint saved: (?P<path>.+)",
                pattern_type="regex",
                action="enqueue_evaluation",
                metadata={"kind": "checkpoint"},
                extract_groups={"checkpoint_path": "path"},
            )
        ],
    )
    monitor = SlurmLogMonitor(config)
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
    }
    controller = MonitorController(monitor, slurm, policies)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("Checkpoint saved: /ckpt/path\n")

    job_id = slurm.submit("demo", script, log)
    controller.register_job(
        job_id,
        JobRegistration(name="demo", script_path=script, log_path=log),
    )

    result = controller.observe_once_sync()

    assert not result.decisions
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.action == "enqueue_evaluation"
    assert action.metadata["checkpoint_path"] == "/ckpt/path"

    # Same content should not fire duplicate signals
    result_repeat = controller.observe_once_sync()
    assert not result_repeat.actions


def test_monitor_signal_triggers_restart(tmp_path: Path) -> None:
    config = SlurmLogMonitorConfig(
        inactivity_threshold_seconds=0,
        check_interval_seconds=1,
        log_signals=[
            LogSignalConfig(
                name="fatal",
                pattern="FATAL ERROR",
                pattern_type="substring",
                mode="crash",
            )
        ],
    )
    monitor = SlurmLogMonitor(config)
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "crash": AlwaysRestartPolicy(AlwaysRestartPolicyConfig(max_retries=2)),
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
    }
    controller = MonitorController(monitor, slurm, policies)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("FATAL ERROR occurred\n")

    job_id = slurm.submit("demo", script, log)
    controller.register_job(
        job_id,
        JobRegistration(name="demo", script_path=script, log_path=log),
    )

    result = controller.observe_once_sync()

    assert any(dec.action == "restart" for dec in result.decisions.values())


def test_output_file_emit_checkpoint_signal(tmp_path: Path) -> None:
    output_file = tmp_path / "train.log"
    output_file.write_text("Checkpoint saved: /tmp/check.ckpt\n")

    config = SlurmLogMonitorConfig(
        inactivity_threshold_seconds=0,
        check_interval_seconds=1,
        output_paths=[str(output_file)],
    )
    monitor = SlurmLogMonitor(config)
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {"success": NoRestartPolicy(NoRestartPolicyConfig())}
    controller = MonitorController(monitor, slurm, policies)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("running\n")

    job_id = slurm.submit("demo", script, log)
    controller.register_job(
        job_id,
        JobRegistration(
            name="demo",
            script_path=script,
            log_path=log,
            output_paths=[output_file],
        ),
    )

    result = controller.observe_once_sync()

    assert any(action.action == "new_checkpoint" for action in result.actions)
    checkpoint_action = next(action for action in result.actions if action.action == "new_checkpoint")
    assert checkpoint_action.metadata["checkpoint_path"] == "/tmp/check.ckpt"
    assert checkpoint_action.metadata["source"] == "output"


def test_slurm_state_transitions_emit_actions(tmp_path: Path) -> None:
    monitor = SlurmLogMonitor(SlurmLogMonitorConfig(check_interval_seconds=1))
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
    }
    controller = MonitorController(monitor, slurm, policies)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("starting\n")

    job_id = slurm.submit("demo", script, log)
    controller.register_job(
        job_id,
        JobRegistration(name="demo", script_path=script, log_path=log),
    )

    slurm.set_state(job_id, "RUNNING")
    first = controller.observe_once_sync()
    assert any(action.action == "run_started" for action in first.actions)

    slurm.set_state(job_id, "COMPLETED")
    second = controller.observe_once_sync()
    assert any(action.action == "run_ended" for action in second.actions)


def test_log_completion_triggers_run_finished(tmp_path: Path) -> None:
    monitor = SlurmLogMonitor(SlurmLogMonitorConfig(check_interval_seconds=1))
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
    }
    controller = MonitorController(monitor, slurm, policies)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("Training complete\n")

    job_id = slurm.submit("demo", script, log)
    controller.register_job(
        job_id,
        JobRegistration(name="demo", script_path=script, log_path=log),
    )

    result = controller.observe_once_sync()
    assert any(action.action == "run_finished" for action in result.actions)


def test_cancelled_job_restarts_with_selective_policy(tmp_path: Path) -> None:
    """Test that a CANCELLED job restarts when policy allows it."""
    from oellm_autoexp.monitor.policy import SelectiveRestartPolicy, SelectiveRestartPolicyConfig

    monitor = SlurmLogMonitor(SlurmLogMonitorConfig(check_interval_seconds=1))
    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    policies = {
        "crash": SelectiveRestartPolicy(
            SelectiveRestartPolicyConfig(
                max_retries=3,
                default_action="restart",
                restart_on_error_types=["cancelled"],
                restart_on_subsystems=["slurm"],
                exclude_error_types=["oom"],
            )
        ),
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
    }
    controller = MonitorController(monitor, slurm, policies)

    script = tmp_path / "script.sh"
    script.write_text("#!/bin/bash\n")
    log = tmp_path / "log.txt"
    log.write_text("running\n")

    job_id = slurm.submit("demo", script, log)
    controller.register_job(
        job_id,
        JobRegistration(name="demo", script_path=script, log_path=log),
    )

    # Simulate job running then being cancelled
    slurm.set_state(job_id, "RUNNING")
    first = controller.observe_once_sync()
    assert any(action.action == "run_started" for action in first.actions)

    # Cancel the job
    slurm.set_state(job_id, "CANCELLED")
    second = controller.observe_once_sync()

    # Should trigger restart
    assert any(dec.action == "restart" for dec in second.decisions.values())
    assert any(action.action == "run_ended" for action in second.actions)

    # Verify job was restarted
    job_states = list(controller.jobs())
    assert len(job_states) == 1
    assert job_states[0].attempts == 2
    assert job_states[0].job_id != job_id


def test_cancelled_job_logging_visibility(tmp_path: Path) -> None:
    """Test that cancelled job events are properly logged."""
    import logging
    from oellm_autoexp.monitor.policy import SelectiveRestartPolicy, SelectiveRestartPolicyConfig

    # Capture log output
    import io

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("oellm_autoexp.monitor.controller")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        monitor = SlurmLogMonitor(SlurmLogMonitorConfig(check_interval_seconds=1))
        slurm = FakeSlurmClient(FakeSlurmClientConfig())
        policies = {
            "crash": SelectiveRestartPolicy(
                SelectiveRestartPolicyConfig(
                    max_retries=3,
                    default_action="restart",
                    restart_on_error_types=["cancelled"],
                )
            ),
            "success": NoRestartPolicy(NoRestartPolicyConfig()),
        }
        controller = MonitorController(monitor, slurm, policies)

        script = tmp_path / "script.sh"
        script.write_text("#!/bin/bash\n")
        log = tmp_path / "log.txt"
        log.write_text("running\n")

        job_id = slurm.submit("demo", script, log)
        controller.register_job(
            job_id,
            JobRegistration(name="demo", script_path=script, log_path=log),
        )

        # Set to CANCELLED
        slurm.set_state(job_id, "CANCELLED")
        controller.observe_once_sync()

        # Verify logging happened
        log_output = log_capture.getvalue()
        assert "SLURM state transition" in log_output
        assert "CANCELLED" in log_output
        assert "classified as mode 'crash'" in log_output
        assert "policy decision" in log_output
        assert "restart" in log_output.lower()

    finally:
        logger.removeHandler(handler)
