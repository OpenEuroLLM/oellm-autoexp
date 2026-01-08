"""Integration tests for multi-stage training workflow."""

from pathlib import Path

from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.config.loader import load_config
from oellm_autoexp.sweep.expander import expand_sweep
from oellm_autoexp.sweep.dag_resolver import resolve_sweep_with_dag
from oellm_autoexp.sweep.validator import validate_execution_plan

# Import to register configs in registry
import oellm_autoexp.monitor.watcher  # noqa: F401
import oellm_autoexp.backends.base  # noqa: F401
import oellm_autoexp.slurm.client  # noqa: F401


def test_multi_stage_full_workflow(tmp_path: Path):
    """Test full workflow: expand → resolve with DAG → validate."""
    # Create config with multi-stage sweep
    config_yaml = """
project:
  name: demo
  base_output_dir: ./outputs
  log_path: ./outputs/slurm-%j.out
  log_path_current: ./outputs/current.log

sweep:
  type: product
  base_values:
    project.name: "demo_dummy\\\\${backend.dummy}_\\\\${stage}"
  groups:
    - type: product
      params:
        backend.dummy: [1, 2]
    - type: list
      configs:
        - stage: stable
        - stage: cooldown
          backend.base_command: ["echo", "\\\\${sibling.stable.output_dir}/checkpoint"]

slurm:
  template_path: template.sbatch
  script_dir: ./scripts
  log_dir: ./logs
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient

monitoring:
  class_name: NullMonitor
  log_path: ./outputs/current.log

backend:
  class_name: NullBackend
  dummy: 0

stage: ""
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(config_yaml)

    # Create config setup (required for Hydra-based resolution)
    config_setup = ConfigSetup(
        pwd=str(tmp_path),
        config_ref=str(config_path),
        config_dir=str(tmp_path),
        override=[],
    )

    # Step 1: Expand sweep
    root = load_config(config_path)
    points = expand_sweep(root.sweep)
    assert len(points) == 4

    # Step 2: Resolve with DAG (combines planning + sibling resolution)
    points_dict = {p.index: p for p in points}
    jobs = resolve_sweep_with_dag(root, points_dict, config_setup)
    assert len(jobs) == 4

    # Verify job names are correct (sibling refs resolved)
    job_names = sorted([j.name for j in jobs])
    assert "demo_dummy1_stable" in job_names
    assert "demo_dummy1_cooldown" in job_names
    assert "demo_dummy2_stable" in job_names
    assert "demo_dummy2_cooldown" in job_names

    # Step 3: Validate
    result = validate_execution_plan(jobs)
    assert result.is_valid
    assert len(result.errors) == 0


def test_multi_stage_with_cancel_conditions(tmp_path: Path):
    """Test multi-stage workflow with cancel_conditions."""
    config_yaml = """
project:
  name: demo
  base_output_dir: ./outputs
  log_path: ./outputs/slurm-%j.out
  log_path_current: ./outputs/current.log

sweep:
  type: product
  base_values:
    project.name: "demo_dummy\\\\${backend.dummy}_\\\\${stage}"
  groups:
    - type: product
      params:
        backend.dummy: [5]
    - type: list
      configs:
        - stage: stable
        - stage: cooldown
          backend.base_command: ["echo", "\\\\${sibling.stable.output_dir}/checkpoint"]
          job.cancel_conditions:
            - class_name: LogPatternCondition
              log_path: "\\\\${sibling.stable.project.log_path_current}"
              pattern: "FATAL ERROR"

slurm:
  template_path: template.sbatch
  script_dir: ./scripts
  log_dir: ./logs
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient

monitoring:
  class_name: NullMonitor
  log_path: ./outputs/current.log

backend:
  class_name: NullBackend
  dummy: 0

stage: ""
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(config_yaml)

    config_setup = ConfigSetup(
        pwd=str(tmp_path),
        config_ref=str(config_path),
        config_dir=str(tmp_path),
        override=[],
    )

    root = load_config(config_path)
    points = expand_sweep(root.sweep)
    points_dict = {p.index: p for p in points}
    jobs = resolve_sweep_with_dag(root, points_dict, config_setup)

    # Check that cancel_conditions are present
    cooldown_job = next(j for j in jobs if "_cooldown" in j.name)
    assert len(cooldown_job.cancel_conditions) >= 1

    # Validate
    result = validate_execution_plan(jobs)
    assert result.is_valid


def test_multi_stage_with_start_conditions(tmp_path: Path):
    """Test multi-stage workflow with start_conditions (new async approach)."""
    config_yaml = """
project:
  name: demo
  base_output_dir: ./outputs
  log_path: ./outputs/slurm-%j.out
  log_path_current: ./outputs/current.log

sweep:
  type: product
  base_values:
    project.name: "demo_dummy\\\\${backend.dummy}_\\\\${stage}"
  groups:
    - type: product
      params:
        backend.dummy: [2]
    - type: list
      configs:
        - stage: stable
        - stage: cooldown
          backend.base_command: ["echo", "\\\\${sibling.stable.output_dir}/checkpoint"]
          job.start_conditions:
            - class_name: FileExistsCondition
              path: "\\\\${sibling.stable.output_dir}/checkpoint/done.txt"
              blocking: true
              timeout_seconds: 7200

slurm:
  template_path: template.sbatch
  script_dir: ./scripts
  log_dir: ./logs
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient

monitoring:
  class_name: NullMonitor
  log_path: ./outputs/current.log

backend:
  class_name: NullBackend
  dummy: 0

stage: ""
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(config_yaml)

    config_setup = ConfigSetup(
        pwd=str(tmp_path),
        config_ref=str(config_path),
        config_dir=str(tmp_path),
        override=[],
    )

    root = load_config(config_path)
    points = expand_sweep(root.sweep)
    points_dict = {p.index: p for p in points}
    jobs = resolve_sweep_with_dag(root, points_dict, config_setup)

    # Check that start_conditions are present
    cooldown_job = next(j for j in jobs if "_cooldown" in j.name)
    assert len(cooldown_job.start_conditions) >= 1

    # Validate
    result = validate_execution_plan(jobs)
    assert result.is_valid


def test_multi_stage_chain(tmp_path: Path):
    """Test a 4-stage training chain."""
    config_yaml = """
project:
  name: demo
  base_output_dir: ./outputs
  log_path: ./outputs/slurm-%j.out
  log_path_current: ./outputs/current.log

sweep:
  type: list
  base_values:
    project.name: "demo_dummy\\\\${backend.dummy}_\\\\${stage}"
  groups:
    - type: list
      configs:
        - stage: pre_pre_training
        - stage: pre_training
          backend.base_command: ["echo", "\\\\${sibling.pre_pre_training.output_dir}/checkpoint"]
        - stage: mid_training
          backend.base_command: ["echo", "\\\\${sibling.pre_training.output_dir}/checkpoint"]
        - stage: post_training
          backend.base_command: ["echo", "\\\\${sibling.mid_training.output_dir}/checkpoint"]

slurm:
  template_path: template.sbatch
  script_dir: ./scripts
  log_dir: ./logs
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient

monitoring:
  class_name: NullMonitor
  log_path: ./outputs/current.log

backend:
  class_name: NullBackend
  dummy: 0

stage: ""
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(config_yaml)

    config_setup = ConfigSetup(
        pwd=str(tmp_path),
        config_ref=str(config_path),
        config_dir=str(tmp_path),
        override=[],
    )

    root = load_config(config_path)
    points = expand_sweep(root.sweep)
    assert len(points) == 4

    points_dict = {p.index: p for p in points}
    jobs = resolve_sweep_with_dag(root, points_dict, config_setup)

    # Verify job names
    job_names = [j.name for j in jobs]
    assert "demo_dummy0_pre_pre_training" in job_names
    assert "demo_dummy0_pre_training" in job_names
    assert "demo_dummy0_mid_training" in job_names
    assert "demo_dummy0_post_training" in job_names

    result = validate_execution_plan(jobs)
    assert result.is_valid


def test_backward_compatibility_with_legacy_grids(tmp_path: Path):
    """Test that legacy grid format still works."""
    config_yaml = """
project:
  name: demo
  base_output_dir: ./outputs
  log_path: ./outputs/slurm-%j.out
  log_path_current: ./outputs/current.log

sweep:
  base_values:
    project.name: "demo_dummy\\\\${backend.dummy}_\\\\${index}"
  grids:
    - backend.dummy: [3, 4]
      backend.base_command: [["echo"], ["date"]]

slurm:
  template_path: template.sbatch
  script_dir: ./scripts
  log_dir: ./logs
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient

monitoring:
  class_name: NullMonitor
  log_path: ./outputs/current.log

backend:
  class_name: NullBackend
  dummy: 0

stage: ""
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(config_yaml)

    config_setup = ConfigSetup(
        pwd=str(tmp_path),
        config_ref=str(config_path),
        config_dir=str(tmp_path),
        override=[],
    )

    root = load_config(config_path)
    points = expand_sweep(root.sweep)
    assert len(points) == 4

    points_dict = {p.index: p for p in points}
    jobs = resolve_sweep_with_dag(root, points_dict, config_setup)
    result = validate_execution_plan(jobs)

    assert result.is_valid
    assert len(jobs) == 4
