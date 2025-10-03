from pathlib import Path

from oellm_autoexp.monitor.controller import MonitorController
from oellm_autoexp.monitor.policy import AlwaysRestartPolicy, AlwaysRestartPolicyConfig
from oellm_autoexp.monitor.policy import NoRestartPolicy, NoRestartPolicyConfig
from oellm_autoexp.monitor.watcher import NullMonitor, NullMonitorConfig
from oellm_autoexp.slurm.fake_sbatch import FakeSlurm


def test_monitor_controller_restart_increments():
    monitor = NullMonitor(NullMonitorConfig())
    slurm = FakeSlurm()
    policies = {
        "stall": AlwaysRestartPolicy(AlwaysRestartPolicyConfig(max_retries=2)),
        "success": NoRestartPolicy(NoRestartPolicyConfig()),
    }
    controller = MonitorController(monitor, slurm, policies)

    job_id = slurm.submit("demo", Path("script"), Path("log"))
    controller.register_job(job_id, "demo")

    decision = controller.handle_state_change(job_id, "stall")
    assert decision.action == "restart"

    decision = controller.handle_state_change(job_id, "missing-mode")
    assert decision.action == "stop"
