"""Integration tests for basic workflow.

Focuses on testing the orchestrator's ability to build execution plans
from configuration files.
"""

from pathlib import Path
import os
import pytest

from oellm_autoexp.orchestrator import build_execution_plan
from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup


def test_load_real_config_and_build_plan(tmp_path: Path, monkeypatch) -> None:
    """Test that we can load a real config and build an execution plan."""
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    monkeypatch.setenv("CONTAINER_CACHE_DIR", str(tmp_path))

    project_cfg = Path("config/project/default.yaml")
    slurm_cfg = Path("config/slurm/base.yaml")
    if not (project_cfg.exists() and slurm_cfg.exists()):
        pytest.skip("Base config not available")

    config_setup = ConfigSetup(
        pwd=os.path.abspath(os.curdir),
        config_name="autoexp",
        config_dir="config",
        overrides=["job=default", "slurm=base"],
    )
    cfg = load_config_reference(config_setup=config_setup)

    plan = build_execution_plan(cfg, config_setup)

    # Should have jobs from the sweep
    assert len(plan.jobs) > 0
    assert plan.config is not None
    assert plan.sweep_points is not None


def test_simple_execution_plan_from_inline_config(tmp_path: Path, monkeypatch) -> None:
    """Test building an execution plan with a minimal inline config."""
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")

    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"
    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    config_text = f"""
job:
  name: demo_${{index}}
  base_output_dir: {base_output}
  log_path: {log_dir}/test-%j.log
  log_path_current: {log_dir}/test.log
  slurm: ${{slurm}}

sweep:
  type: product
  groups:
    - params:
        backend.dummy: [0, 1]

slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  array: false
  launcher_cmd: ""
  srun_opts: ""

backend:
  class_name: NullBackend
  base_command: ["echo", "1"]

index: 0
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(config_text)

    config_setup = ConfigSetup(config_path=config_path)
    root = load_config_reference(config_setup=config_setup)

    # Test that we can build a plan
    plan = build_execution_plan(root, config_setup)

    # Should have 2 jobs (2 values in sweep)
    assert len(plan.jobs) == 2
    assert all(job.config for job in plan.jobs)
    assert all(job.config.job.name for job in plan.jobs)


def test_multi_stage_execution_plan(tmp_path: Path, monkeypatch) -> None:
    """Test building a multi-stage execution plan with dependencies."""
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")

    base_output = tmp_path / "outputs"
    template_path = tmp_path / "template.sbatch"
    script_dir = tmp_path / "scripts"
    log_dir = tmp_path / "logs"
    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    config_text = f"""
job:
  name: demo_${{index}}_${{stage}}
  base_output_dir: {base_output}
  log_path: {log_dir}/test-%j.log
  log_path_current: {log_dir}/test.log
  slurm: ${{slurm}}

sweep:
  type: product
  groups:
    - params:
        backend.dummy: [1, 2]
    - type: list
      configs:
        - stage: stable
        - type: list
          defaults:
            job.start_condition:
              class_name: FileExistsCondition
              path: "\\\\${{sibling.stable.job.base_output_dir}}/done.txt"
          configs:
            - stage: decay

slurm:
  template_path: {template_path}
  script_dir: {script_dir}
  log_dir: {log_dir}
  array: false
  launcher_cmd: ""
  srun_opts: ""

backend:
  class_name: NullBackend
  base_command: ["echo", "1"]

stage: ""
index: 0
"""
    config_path = tmp_path / "multi_stage.yaml"
    config_path.write_text(config_text)

    config_setup = ConfigSetup(config_path=config_path)
    root = load_config_reference(config_setup=config_setup)

    # Test that we can build a plan
    plan = build_execution_plan(root, config_setup)

    # Should have 4 jobs (2 backend values × 2 stages)
    assert len(plan.jobs) == 4

    # Verify job names contain stage info
    job_names = [job.config.job.name for job in plan.jobs]
    assert any("stable" in name for name in job_names)
    assert any("decay" in name for name in job_names)

    # Verify start conditions are set for decay stage
    decay_jobs = [job for job in plan.jobs if "decay" in job.config.job.name]
    for job in decay_jobs:
        # Access the job config to check start_condition
        assert hasattr(job.config, "job")
        # The start condition should be inherited from the config
        assert job.config.job.start_condition is not None
