"""Integration tests for multi-stage training workflow."""

from compoconf import parse_config

from oellm_autoexp.config.schema import RootConfig, SweepConfig
from oellm_autoexp.sweep.expander import expand_sweep
from oellm_autoexp.sweep.planner import build_job_plans
from oellm_autoexp.sweep.sibling_resolver import resolve_sibling_references
from oellm_autoexp.sweep.validator import validate_execution_plan

# Import to register configs in registry
import oellm_autoexp.monitor.watcher  # noqa: F401
import oellm_autoexp.backends.base  # noqa: F401
import oellm_autoexp.slurm.client  # noqa: F401


def _basic_root() -> RootConfig:
    """Create a basic root config for testing."""
    data = {
        "project": {"name": "demo", "base_output_dir": "./outputs"},
        "sweep": {"grids": []},
        "slurm": {
            "template_path": "template.sbatch",
            "script_dir": "./scripts",
            "log_dir": "./logs",
            "launcher_cmd": "",
            "srun_opts": "",
            "client": {"class_name": "FakeSlurmClient"},
        },
        "monitoring": {"class_name": "NullMonitor"},
        "backend": {"class_name": "NullBackend"},
    }
    return parse_config(RootConfig, data)


def test_multi_stage_full_workflow():
    """Test full workflow: expand → plan → resolve → validate."""
    root = _basic_root()

    # Configure a multi-stage sweep
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {"backend.megatron.lr": [1e-4, 5e-4]},
            },
            {
                "type": "list",
                "configs": [
                    {"stage": "stable"},
                    {
                        "stage": "cooldown",
                        "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint",
                    },
                ],
            },
        ],
        name_template="{project}_lr{backend.megatron.lr}_{stage}",
    )
    root.sweep = sweep_cfg

    # Step 1: Expand sweep
    points = expand_sweep(sweep_cfg)
    assert len(points) == 4  # 2 lr * 2 stages

    # Step 2: Build job plans
    jobs = build_job_plans(root, points)
    assert len(jobs) == 4

    # Verify sibling references are present
    cooldown_jobs = [j for j in jobs if "_cooldown" in j.name]
    assert len(cooldown_jobs) == 2
    for job in cooldown_jobs:
        assert "{sibling.stable.output_dir}" in job.parameters.get("backend.megatron.load", "")

    # Step 3: Resolve sibling references
    resolved_jobs = resolve_sibling_references(jobs)
    assert len(resolved_jobs) == 4

    # Verify sibling references are resolved
    for job in resolved_jobs:
        if "_cooldown" in job.name:
            load_path = job.parameters.get("backend.megatron.load", "")
            assert "{sibling." not in load_path
            assert "_stable" in load_path

    # Step 4: Validate
    result = validate_execution_plan(resolved_jobs)
    assert result.is_valid
    assert len(result.errors) == 0


def test_multi_stage_with_cancel_conditions():
    """Test multi-stage workflow with cancel_conditions."""
    root = _basic_root()

    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {"backend.megatron.lr": [1e-4]},
            },
            {
                "type": "list",
                "configs": [
                    {"stage": "stable"},
                    {
                        "stage": "cooldown",
                        "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint",
                        "cancel_conditions": [
                            {
                                "class_name": "LogPatternCondition",
                                "log_path": "{sibling.stable.log_path_current}",
                                "pattern": "FATAL ERROR",
                            },
                        ],
                    },
                ],
            },
        ],
        name_template="{project}_lr{backend.megatron.lr}_{stage}",
    )
    root.sweep = sweep_cfg

    points = expand_sweep(sweep_cfg)
    jobs = build_job_plans(root, points)
    resolved_jobs = resolve_sibling_references(jobs)

    # Check that cancel_conditions are resolved
    cooldown_job = next(j for j in resolved_jobs if "_cooldown" in j.name)
    assert len(cooldown_job.cancel_conditions) == 1
    # The log path uses the actual job name computed from the template
    assert "stable/current.log" in cooldown_job.cancel_conditions[0]["log_path"]
    assert "{sibling." not in cooldown_job.cancel_conditions[0]["log_path"]

    # Validate
    result = validate_execution_plan(resolved_jobs)
    assert result.is_valid


def test_multi_stage_with_start_conditions():
    """Test multi-stage workflow with start_conditions (new async approach)."""
    root = _basic_root()

    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {"backend.megatron.lr": [1e-4]},
            },
            {
                "type": "list",
                "configs": [
                    {"stage": "stable"},
                    {
                        "stage": "cooldown",
                        "backend.megatron.load": "{sibling.stable.output_dir}/checkpoint",
                        "start_conditions": [
                            {
                                "class_name": "FileExistsCondition",
                                "path": "{sibling.stable.output_dir}/checkpoint/done.txt",
                                "blocking": True,
                                "timeout_seconds": 7200,
                            },
                        ],
                    },
                ],
            },
        ],
        name_template="{project}_lr{backend.megatron.lr}_{stage}",
    )
    root.sweep = sweep_cfg

    points = expand_sweep(sweep_cfg)
    jobs = build_job_plans(root, points)
    resolved_jobs = resolve_sibling_references(jobs)

    # Check that start_conditions are resolved
    cooldown_job = next(j for j in resolved_jobs if "_cooldown" in j.name)
    assert len(cooldown_job.start_conditions) == 1
    assert cooldown_job.start_conditions[0]["class_name"] == "FileExistsCondition"
    # The path should have sibling reference resolved
    assert "stable/checkpoint/done.txt" in cooldown_job.start_conditions[0]["path"]
    assert "{sibling." not in cooldown_job.start_conditions[0]["path"]

    # Validate
    result = validate_execution_plan(resolved_jobs)
    assert result.is_valid


def test_multi_stage_chain():
    """Test a 4-stage training chain."""
    root = _basic_root()

    sweep_cfg = SweepConfig(
        type="list",
        groups=[
            {
                "type": "list",
                "configs": [
                    {"stage": "pre_pre_training"},
                    {
                        "stage": "pre_training",
                        "backend.megatron.load": "{sibling.pre_pre_training.output_dir}/checkpoint",
                    },
                    {
                        "stage": "mid_training",
                        "backend.megatron.load": "{sibling.pre_training.output_dir}/checkpoint",
                    },
                    {
                        "stage": "post_training",
                        "backend.megatron.load": "{sibling.mid_training.output_dir}/checkpoint",
                    },
                ],
            },
        ],
        name_template="{project}_{stage}",
    )
    root.sweep = sweep_cfg

    points = expand_sweep(sweep_cfg)
    assert len(points) == 4

    jobs = build_job_plans(root, points)
    resolved_jobs = resolve_sibling_references(jobs)

    # Verify chain is correctly resolved
    pre_training = next(j for j in resolved_jobs if j.name == "demo_pre_training")
    mid_training = next(j for j in resolved_jobs if j.name == "demo_mid_training")
    post_training = next(j for j in resolved_jobs if j.name == "demo_post_training")

    assert "demo_pre_pre_training" in pre_training.parameters["backend.megatron.load"]
    assert "demo_pre_training" in mid_training.parameters["backend.megatron.load"]
    assert "demo_mid_training" in post_training.parameters["backend.megatron.load"]

    result = validate_execution_plan(resolved_jobs)
    assert result.is_valid


def test_backward_compatibility_with_legacy_grids():
    """Test that legacy grid format still works."""
    root = _basic_root()

    # Use legacy grid format
    sweep_cfg = SweepConfig(
        grids=[
            {"backend.megatron.lr": [1e-4, 5e-4], "backend.megatron.global_batch_size": [64, 128]}
        ],
        name_template="{project}_{index}",
    )
    root.sweep = sweep_cfg

    points = expand_sweep(sweep_cfg)
    assert len(points) == 4  # 2 lr * 2 batch_size

    jobs = build_job_plans(root, points)
    resolved_jobs = resolve_sibling_references(jobs)
    result = validate_execution_plan(resolved_jobs)

    assert result.is_valid
    assert len(resolved_jobs) == 4
