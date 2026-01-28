"""Integration tests for the visualization module.

Tests basic visualization functionality using mock job objects that
match the expected interface.
"""

from dataclasses import dataclass, field

# Import visualization functions
from scripts.visualize_plan import (
    group_jobs_by_stage,
    extract_hyperparameters,
    format_condition,
)


@dataclass
class MockJobPlan:
    """Mock JobPlan for testing visualization functions."""

    config: object = None
    parameters: list[str] = field(default_factory=list)
    stage_name: str | None = None
    start_conditions: list = field(default_factory=list)
    cancel_conditions: list = field(default_factory=list)
    output_paths: list = field(default_factory=list)


def test_group_jobs_by_stage():
    """Test grouping jobs by stage parameter."""
    jobs = [
        MockJobPlan(parameters=["stage=stable"], stage_name="stable"),
        MockJobPlan(parameters=["stage=stable"], stage_name="stable"),
        MockJobPlan(parameters=["stage=cooldown"], stage_name="cooldown"),
    ]

    stages = group_jobs_by_stage(jobs)

    assert len(stages) == 2
    assert len(stages["stable"]) == 2
    assert len(stages["cooldown"]) == 1


def test_extract_hyperparameters():
    """Test extraction of hyperparameter values."""
    jobs = [
        MockJobPlan(parameters=["backend.dummy=1", "stage=stable"]),
        MockJobPlan(parameters=["backend.dummy=2", "stage=stable"]),
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
