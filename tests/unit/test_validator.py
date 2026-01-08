"""Tests for execution plan validation."""

from oellm_autoexp.sweep.planner import JobPlan
from oellm_autoexp.sweep.validator import validate_execution_plan


def test_validate_execution_plan_valid():
    """Test validation passes for valid plans."""
    jobs = [
        JobPlan(
            name="lr1e-4_stable",
            parameters=["stage=stable", "backend.megatron.lr=1e-4"],
            output_dir="/outputs/lr1e-4_stable",
            log_path="/logs/lr1e-4_stable/slurm-%j.out",
            log_path_current="/logs/lr1e-4_stable/current.log",
        ),
        JobPlan(
            name="lr1e-4_cooldown",
            parameters=[
                "stage=cooldown",
                "backend.megatron.lr=1e-4",
                "backend.megatron.load=/outputs/lr1e-4_stable/checkpoint",
            ],
            output_dir="/outputs/lr1e-4_cooldown",
            log_path="/logs/lr1e-4_cooldown/slurm-%j.out",
            log_path_current="/logs/lr1e-4_cooldown/current.log",
        ),
    ]

    result = validate_execution_plan(jobs)

    assert result.is_valid
    assert len(result.errors) == 0


def test_validate_execution_plan_duplicate_names():
    """Test validation fails for duplicate job names."""
    jobs = [
        JobPlan(
            name="duplicate_name",
            parameters=[],
            output_dir="/outputs/job1",
            log_path="/logs/job1/slurm-%j.out",
            log_path_current="/logs/job1/current.log",
        ),
        JobPlan(
            name="duplicate_name",
            parameters=[],
            output_dir="/outputs/job2",
            log_path="/logs/job2/slurm-%j.out",
            log_path_current="/logs/job2/current.log",
        ),
    ]

    result = validate_execution_plan(jobs)

    assert not result.is_valid
    assert len(result.errors) > 0
    assert any("Duplicate job names" in error for error in result.errors)


def test_validate_execution_plan_invalid_cancel_condition_structure():
    """Test validation warns for invalid cancel_condition structure."""
    jobs = [
        JobPlan(
            name="lr1e-4_cooldown",
            parameters=["stage=cooldown"],
            output_dir="/outputs/lr1e-4_cooldown",
            log_path="/logs/lr1e-4_cooldown/slurm-%j.out",
            log_path_current="/logs/lr1e-4_cooldown/current.log",
            cancel_conditions=[
                {
                    # Missing class_name
                    "log_path": "/logs/stable/current.log",
                    "pattern": "FATAL ERROR",
                },
            ],
        ),
    ]

    result = validate_execution_plan(jobs)

    # Should still be valid but with warnings
    assert result.is_valid
    assert len(result.warnings) > 0
    assert any("missing 'class_name'" in warning for warning in result.warnings)


def test_validate_execution_plan_circular_dependency():
    """Test validation detects circular dependencies."""
    jobs = [
        JobPlan(
            name="job_a",
            parameters=[],
            output_dir="/outputs/job_a",
            log_path="/logs/job_a/slurm-%j.out",
            log_path_current="/logs/job_a/current.log",
            start_condition_cmd="wait_for job_b",
        ),
        JobPlan(
            name="job_b",
            parameters=[],
            output_dir="/outputs/job_b",
            log_path="/logs/job_b/slurm-%j.out",
            log_path_current="/logs/job_b/current.log",
            start_condition_cmd="wait_for job_a",
        ),
    ]

    result = validate_execution_plan(jobs)

    assert not result.is_valid
    assert len(result.errors) > 0
    assert any("Circular dependencies" in error for error in result.errors)
