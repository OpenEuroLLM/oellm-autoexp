"""Integration tests for the visualization module."""

from io import StringIO
import sys

from oellm_autoexp.sweep.expander import expand_sweep
from oellm_autoexp.sweep.planner import JobPlan
from oellm_autoexp.sweep.sibling_resolver import resolve_sibling_references
from oellm_autoexp.config.schema import SweepConfig

# Import visualization function
from scripts.visualize_plan import (
    visualize_plan,
    group_jobs_by_stage,
    extract_hyperparameters,
    format_condition,
)


def test_group_jobs_by_stage():
    """Test grouping jobs by stage parameter."""
    jobs = [
        JobPlan(
            name="job1_stable",
            parameters={"stage": "stable"},
            output_dir="/outputs/job1_stable",
            log_path="/logs/job1.out",
        ),
        JobPlan(
            name="job2_stable",
            parameters={"stage": "stable"},
            output_dir="/outputs/job2_stable",
            log_path="/logs/job2.out",
        ),
        JobPlan(
            name="job1_cooldown",
            parameters={"stage": "cooldown"},
            output_dir="/outputs/job1_cooldown",
            log_path="/logs/job1_cooldown.out",
        ),
    ]

    stages = group_jobs_by_stage(jobs)

    assert len(stages) == 2
    assert len(stages["stable"]) == 2
    assert len(stages["cooldown"]) == 1


def test_extract_hyperparameters():
    """Test extraction of hyperparameter values."""
    jobs = [
        JobPlan(
            name="job1",
            parameters={
                "backend.megatron.lr": 1e-4,
                "backend.megatron.global_batch_size": 64,
                "stage": "stable",
            },
            output_dir="/outputs/job1",
            log_path="/logs/job1.out",
        ),
        JobPlan(
            name="job2",
            parameters={
                "backend.megatron.lr": 5e-4,
                "backend.megatron.global_batch_size": 128,
                "stage": "stable",
            },
            output_dir="/outputs/job2",
            log_path="/logs/job2.out",
        ),
    ]

    hyperparams = extract_hyperparameters(jobs)

    assert "backend.megatron.lr" in hyperparams
    assert "backend.megatron.global_batch_size" in hyperparams
    assert set(hyperparams["backend.megatron.lr"]) == {1e-4, 5e-4}
    assert set(hyperparams["backend.megatron.global_batch_size"]) == {64, 128}


def test_format_condition():
    """Test formatting of different condition types."""
    # FileExistsCondition
    file_cond = {
        "class_name": "FileExistsCondition",
        "path": "/checkpoint/done.txt",
    }
    assert "FileExists" in format_condition(file_cond)
    assert "done.txt" in format_condition(file_cond)

    # SlurmStateCondition
    slurm_cond = {
        "class_name": "SlurmStateCondition",
        "job_name": "stable_job",
        "state": "COMPLETED",
    }
    assert "SlurmState" in format_condition(slurm_cond)
    assert "stable_job" in format_condition(slurm_cond)
    assert "COMPLETED" in format_condition(slurm_cond)

    # LogPatternCondition
    log_cond = {
        "class_name": "LogPatternCondition",
        "pattern": "ERROR",
    }
    assert "LogPattern" in format_condition(log_cond)
    assert "ERROR" in format_condition(log_cond)


def test_visualize_plan_output():
    """Test that visualize_plan produces expected output structure."""
    # Create a simple sweep config
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {
                    "backend.megatron.lr": [1e-4, 5e-4],
                    "backend.megatron.global_batch_size": [64, 128],
                },
            },
            {
                "type": "list",
                "configs": [
                    {"stage": "stable"},
                    {
                        "type": "list",
                        "defaults": {
                            "job.start_conditions": [
                                {
                                    "class_name": "FileExistsCondition",
                                    "path": "{sibling.stable.output_dir}/checkpoint/done.txt",
                                }
                            ],
                        },
                        "configs": [{"stage": "cooldown"}],
                    },
                ],
            },
        ],
    )

    # Expand and build jobs
    points = expand_sweep(sweep_cfg)
    jobs = []
    for point in points:
        stage = point.parameters.get("stage", "unknown")
        lr = point.parameters.get("backend.megatron.lr", 1e-4)
        bsz = point.parameters.get("backend.megatron.global_batch_size", 64)
        name = f"test_lr{lr}_bsz{bsz}_{stage}"

        jobs.append(
            JobPlan(
                name=name,
                parameters=point.parameters,
                output_dir=f"/outputs/{name}",
                log_path=f"/logs/{name}/slurm-%j.out",
                output_paths=[],
                start_conditions=point.parameters.get("job.start_conditions", []),
            )
        )

    # Resolve sibling references
    resolved_jobs = resolve_sibling_references(jobs)

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        visualize_plan(resolved_jobs, config_name="test_visualization", max_jobs_per_stage=10)
        output = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    # Verify output structure
    assert "Multi-Stage Experiment Plan" in output
    assert "test_visualization" in output
    assert "Total: 8 jobs" in output  # 4 lr×bsz combos × 2 stages
    assert "Hyperparameter Sweep" in output
    assert "Stage: stable" in output
    assert "Stage: cooldown" in output
    assert "Start Conditions:" in output
    assert "FileExists" in output


def test_multi_stage_visualization_with_conditions():
    """Test visualization of multi-stage experiment with start/cancel
    conditions."""
    # Create jobs manually with conditions
    jobs = [
        # Stable stage
        JobPlan(
            name="lr1e-4_stable",
            parameters={
                "stage": "stable",
                "backend.megatron.lr": 1e-4,
                "backend.megatron.global_batch_size": 64,
            },
            output_dir="/outputs/lr1e-4_stable",
            log_path="/logs/lr1e-4_stable/slurm-%j.out",
            log_path_current="/logs/lr1e-4_stable/current.log",
            output_paths=[],
        ),
        # Cooldown stage with conditions
        JobPlan(
            name="lr1e-4_cooldown",
            parameters={
                "stage": "cooldown",
                "backend.megatron.lr": 1e-4,
                "backend.megatron.global_batch_size": 64,
            },
            output_dir="/outputs/lr1e-4_cooldown",
            log_path="/logs/lr1e-4_cooldown/slurm-%j.out",
            output_paths=[],
            start_conditions=[
                {
                    "class_name": "FileExistsCondition",
                    "path": "/outputs/lr1e-4_stable/checkpoint/done.txt",
                    "blocking": True,
                }
            ],
            cancel_conditions=[
                {
                    "class_name": "SlurmStateCondition",
                    "job_name": "lr1e-4_stable",
                    "state": "FAILED",
                },
                {
                    "class_name": "LogPatternCondition",
                    "log_path": "/logs/lr1e-4_stable/current.log",
                    "pattern": "FATAL ERROR",
                },
            ],
        ),
    ]

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        visualize_plan(jobs, config_name="multi_stage_test", max_jobs_per_stage=10)
        output = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    # Verify conditions are displayed
    assert "Start Conditions:" in output
    assert "FileExists" in output
    assert "Cancel Conditions:" in output
    assert "SlurmState" in output
    assert "FAILED" in output
    assert "LogPattern" in output
    assert "FATAL ERROR" in output


def test_visualization_with_many_jobs():
    """Test that visualization truncates when max_jobs_per_stage is
    exceeded."""
    # Create many jobs in one stage
    jobs = [
        JobPlan(
            name=f"job_{i}_stable",
            parameters={"stage": "stable", "idx": i},
            output_dir=f"/outputs/job_{i}_stable",
            log_path=f"/logs/job_{i}.out",
        )
        for i in range(20)
    ]

    # Capture output with limit of 5 jobs per stage
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        visualize_plan(jobs, config_name="many_jobs_test", max_jobs_per_stage=5)
        output = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    # Should show "... and N more"
    assert "and 15 more" in output
    assert "job_0_stable" in output
    assert "job_4_stable" in output
    # Jobs beyond the limit should not appear individually
    assert "job_19_stable" not in output
