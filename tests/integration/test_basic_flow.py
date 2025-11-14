from pathlib import Path

import pytest

from oellm_autoexp.monitor.controller import JobRegistration, MonitorController
from oellm_autoexp.orchestrator import build_execution_plan, render_scripts
from oellm_autoexp.slurm.client import FakeSlurmClient, FakeSlurmClientConfig
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
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
backend:
  class_name: NullBackend
"""


def test_fake_slurm_monitoring_cycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
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
    artifacts = render_scripts(plan)

    slurm = FakeSlurmClient(FakeSlurmClientConfig())
    controller = MonitorController(plan.runtime.monitor, slurm)

    job = plan.jobs[0]
    job_id = slurm.submit(job.name, artifacts.job_scripts[0], job.log_path)
    controller.register_job(
        job_id,
        JobRegistration(name=job.name, script_path=artifacts.job_scripts[0], log_path=job.log_path),
    )

    decision = controller.handle_state_change(job_id, "stall")
    assert decision.action == "restart"

    new_job_id = next(iter(controller.jobs())).job_id
    decision = controller.handle_state_change(new_job_id, "success")
    assert decision.action == "success"


def test_hydra_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    monkeypatch.setenv("CONTAINER_CACHE_DIR", "debug")
    project_cfg = Path("config/project/default.yaml")
    slurm_cfg = Path("config/slurm/juwels.yaml")
    if not (project_cfg.exists() and slurm_cfg.exists()):
        pytest.skip("juwels config not available in this checkout")

    cfg = load_config_reference(
        "autoexp", Path("config"), overrides=["project=default", "slurm=juwels"]
    )
    plan = build_execution_plan(cfg)
    artifacts = render_scripts(plan)
    assert artifacts.job_scripts
