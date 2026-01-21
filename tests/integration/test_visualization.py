"""Integration tests for the visualization module."""

from io import StringIO
import sys
from dataclasses import asdict

from oellm_autoexp.sweep.expander import expand_sweep
from oellm_autoexp.sweep.planner import JobPlan
from oellm_autoexp.config.schema import SweepConfig
from oellm_autoexp.config.loader import load_config
from oellm_autoexp.config.schema import ConfigSetup
from oellm_autoexp.sweep.dag_resolver import resolve_sweep_with_dag, build_stage_dependencies

import yaml

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
            config=None,
            name="job1_stable",
            parameters=["stage=stable"],
            output_dir="/outputs/job1_stable",
            log_path="/logs/job1.out",
            log_path_current="/logs/job1_stable/current.log",
            stage_name="stable",
        ),
        JobPlan(
            config=None,
            name="job2_stable",
            parameters=["stage=stable"],
            output_dir="/outputs/job2_stable",
            log_path="/logs/job2.out",
            log_path_current="/logs/job2_stable/current.log",
            stage_name="stable",
        ),
        JobPlan(
            config=None,
            name="job1_cooldown",
            parameters=["stage=cooldown"],
            output_dir="/outputs/job1_cooldown",
            log_path="/logs/job1_cooldown.out",
            log_path_current="/logs/job1_cooldown/current.log",
            stage_name="cooldown",
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
            config=None,
            name="job1",
            parameters=["backend.dummy=1", "stage=stable"],
            output_dir="/outputs/job1",
            log_path="/logs/job1.out",
            log_path_current="/logs/job1/current.log",
            stage_name="stable",
        ),
        JobPlan(
            config=None,
            name="job2",
            parameters=["backend.dummy=2", "stage=stable"],
            output_dir="/outputs/job2",
            log_path="/logs/job2.out",
            log_path_current="/logs/job2/current.log",
            stage_name="stable",
        ),
    ]

    hyperparams = extract_hyperparameters(jobs, visualize_keys=["backend.dummy"])

    assert "backend.dummy" in hyperparams
    assert set(hyperparams["backend.dummy"]) == {"1", "2"}


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


def test_visualize_plan_output(tmp_path):
    """Test that visualize_plan produces expected output structure."""
    # Create a simple sweep config
    sweep_cfg = SweepConfig(
        type="product",
        groups=[
            {
                "type": "product",
                "params": {
                    "backend.dummy": [1, 2],
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

    template_path = tmp_path / "template.sbatch"
    template_path.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    config_dict = {
        "project": {
            "name": "visualize_demo",
            "base_output_dir": str(output_dir),
            "log_path": str(log_dir / "slurm-%j.out"),
            "log_path_current": str(log_dir / "current.log"),
        },
        "sweep": asdict(sweep_cfg),
        "slurm": {
            "template_path": str(template_path),
            "script_dir": str(tmp_path / "scripts"),
            "log_dir": str(log_dir),
            "client": {"class_name": "FakeSlurmClient"},
        },
        "monitoring": {
            "class_name": "SlurmLogMonitor",
            "log_path": str(log_dir / "current.log"),
        },
        "backend": {"class_name": "NullBackend", "base_command": ["echo", "0"]},
        "stage": "",
        "index": 0,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict, sort_keys=False))

    root = load_config(config_path)
    points = expand_sweep(root.sweep)
    points_dict = {p.index: p for p in points}
    resolved_jobs = resolve_sweep_with_dag(
        root,
        points_dict,
        ConfigSetup(
            pwd=str(tmp_path),
            config_ref=str(config_path),
            config_dir=str(tmp_path),
            override=[],
        ),
    )

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        deps = build_stage_dependencies(points_dict)
        visualize_plan(
            resolved_jobs,
            config_name="test_visualization",
            max_jobs_per_stage=10,
            dependencies=deps,
        )
        output = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    # Verify output structure
    assert "Multi-Stage Experiment Plan" in output
    assert "test_visualization" in output
    assert "Total: 4 jobs" in output  # 2 dummy combos × 2 stages
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
            config=None,
            name="dummy1_stable",
            parameters=["stage=stable", "backend.dummy=1"],
            output_dir="/outputs/dummy1_stable",
            log_path="/logs/dummy1_stable/slurm-%j.out",
            log_path_current="/logs/dummy1_stable/current.log",
            output_paths=[],
            stage_name="stable",
        ),
        # Cooldown stage with conditions
        JobPlan(
            config=None,
            name="dummy1_cooldown",
            parameters=["stage=cooldown", "backend.dummy=2"],
            output_dir="/outputs/dummy1_cooldown",
            log_path="/logs/dummy1_cooldown/slurm-%j.out",
            log_path_current="/logs/dummy1_cooldown/current.log",
            output_paths=[],
            stage_name="cooldown",
            start_conditions=[
                {
                    "class_name": "FileExistsCondition",
                    "path": "/outputs/dummy1_stable/checkpoint/done.txt",
                    "blocking": True,
                }
            ],
            cancel_conditions=[
                {
                    "class_name": "SlurmStateCondition",
                    "job_name": "dummy1_stable",
                    "state": "FAILED",
                },
                {
                    "class_name": "LogPatternCondition",
                    "log_path": "/logs/dummy1_stable/current.log",
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
            config=None,
            name=f"job_{i}_stable",
            parameters=["stage=stable", f"idx={i}"],
            output_dir=f"/outputs/job_{i}_stable",
            log_path=f"/logs/job_{i}.out",
            log_path_current=f"/logs/job_{i}_stable/current.log",
            stage_name="stable",
        )
        for i in range(20)
    ]

    # Capture output with limit of 5 jobs per stage
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        visualize_plan(
            jobs,
            config_name="many_jobs_test",
            max_jobs_per_stage=5,
            visualize_keys=["backend.dummy", "stage"],
        )
        output = captured_output.getvalue()
    finally:
        sys.stdout = old_stdout

    # Should show "... and N more"
    assert "and 15 more" in output
    assert "job_0_stable" in output
    assert "job_4_stable" in output
    # Jobs beyond the limit should not appear individually
    assert "job_19_stable" not in output
