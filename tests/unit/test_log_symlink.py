"""Tests for log_path_current symlink functionality."""

from oellm_autoexp.sweep.planner import JobPlan
from oellm_autoexp.sweep.sibling_resolver import resolve_sibling_references


def test_log_symlink_with_sibling_resolver():
    """Test that sibling resolver properly sets log_path_current."""
    jobs = [
        JobPlan(
            name="lr1e-4_stable",
            parameters={"stage": "stable"},
            output_dir="/outputs/lr1e-4_stable",
            log_path="/logs/lr1e-4_stable/slurm-%j.out",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    # Verify log_path_current was set
    assert resolved_jobs[0].log_path_current == "/logs/lr1e-4_stable/current.log"


def test_log_symlink_path_generation():
    """Test that log_path_current is computed correctly from log_path."""
    jobs = [
        JobPlan(
            name="test_job",
            parameters={},
            output_dir="/outputs/test",
            log_path="/logs/test/slurm-%j.out",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    # Should replace slurm-%j.out with current.log
    assert resolved_jobs[0].log_path_current == "/logs/test/current.log"


def test_log_symlink_multiple_jobs():
    """Test that each job gets its own current.log symlink path."""
    jobs = [
        JobPlan(
            name="job1",
            parameters={},
            output_dir="/outputs/job1",
            log_path="/logs/job1/slurm-%j.out",
        ),
        JobPlan(
            name="job2",
            parameters={},
            output_dir="/outputs/job2",
            log_path="/logs/job2/slurm-%j.out",
        ),
    ]

    resolved_jobs = resolve_sibling_references(jobs)

    assert resolved_jobs[0].log_path_current == "/logs/job1/current.log"
    assert resolved_jobs[1].log_path_current == "/logs/job2/current.log"
