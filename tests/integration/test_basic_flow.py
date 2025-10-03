from pathlib import Path

from oellm_autoexp.monitor.controller import MonitorController
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts
from oellm_autoexp.slurm.fake_sbatch import FakeSlurm
from oellm_autoexp.config.loader import load_config_reference

CONFIG = """
project:
  name: demo
  base_output_dir: {base_output}
sweep:
  axes:
    lr: [0.1]
slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  launcher_cmd: ""
  srun_opts: ""
monitoring:
  implementation:
    class_name: NullMonitor
backend:
  implementation:
    class_name: NullBackend
restart_policies:
  - mode: stall
    implementation:
      class_name: AlwaysRestartPolicy
      max_retries: 2
  - mode: success
    implementation:
      class_name: NoRestartPolicy
"""


def test_fake_slurm_monitoring_cycle(tmp_path: Path) -> None:
    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"
    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    config_text = CONFIG.format(
        base_output=base_output,
        template_path=template_path,
        script_dir=script_dir,
        log_dir=log_dir,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    plan = build_execution_plan(config_path)
    scripts = render_scripts(plan)

    slurm = FakeSlurm()
    controller = MonitorController(plan.runtime.monitor, slurm, plan.runtime.restart_policies)

    job = plan.jobs[0]
    job_id = slurm.submit(job.name, scripts[0], job.log_path)
    controller.register_job(job_id, job.name)

    decision = controller.handle_state_change(job_id, "stall")
    assert decision.action == "restart"

    decision = controller.handle_state_change(job_id, "success")
    assert decision.action == "stop"

def test_hydra_plan(tmp_path: Path) -> None:
    cfg = load_config_reference("autoexp", Path("config"), overrides=["project=juwels", "slurm=juwels"])
    plan = build_execution_plan(cfg)
    scripts = render_scripts(plan)
    assert scripts
