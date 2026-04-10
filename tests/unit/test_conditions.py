"""Test oellm-specific custom monitor conditions."""

from __future__ import annotations

from pathlib import Path

from oellm_autoexp.monitor.conditions import ConditionContext
from oellm_autoexp.monitor.actions import EventRecord
from oellm_autoexp.config.conditions import (
    BlockingFileExistsCondition,
    BlockingFileExistsConditionConfig,
    SlurmStateCondition,
    SlurmStateConditionConfig,
    LogPatternCondition,
    LogPatternConditionConfig,
)


def _context(tmp_path: Path, extra: dict | None = None) -> ConditionContext:
    """Create a test condition context."""
    event = EventRecord(event_id="evt1", name="test", source="test")
    return ConditionContext(
        event=event,
        job_metadata={"output_dir": str(tmp_path)},
        attempts=0,
        extra=extra or {},
    )


def test_blocking_file_exists_condition(tmp_path: Path):
    """Test BlockingFileExistsCondition checks for file existence."""
    target = tmp_path / "checkpoint.txt"
    ctx = _context(tmp_path)

    cond = BlockingFileExistsCondition(
        BlockingFileExistsConditionConfig(
            path=str(target),
            blocking=False,
        )
    )

    # File doesn't exist yet
    result = cond.check(ctx)
    assert not result.passed
    assert "missing" in result.message

    # Create the file
    target.write_text("checkpoint data")

    # Now it should pass
    result = cond.check(ctx)
    assert result.passed


def test_slurm_state_condition_job_found(tmp_path: Path):
    """Test SlurmStateCondition when job is found with matching state."""
    ctx = _context(tmp_path, extra={"job_states_by_name": {"parent_job": "COMPLETED"}})

    cond = SlurmStateCondition(
        SlurmStateConditionConfig(
            job_name="parent_job",
            state="COMPLETED",
        )
    )

    result = cond.check(ctx)
    assert result.passed
    assert "parent_job" in result.message
    assert "COMPLETED" in result.message


def test_slurm_state_condition_job_not_found(tmp_path: Path):
    """Test SlurmStateCondition when job is not found."""
    ctx = _context(tmp_path, extra={"job_states_by_name": {}})

    cond = SlurmStateCondition(
        SlurmStateConditionConfig(
            job_name="missing_job",
            state="COMPLETED",
        )
    )

    result = cond.check(ctx)
    assert not result.passed
    assert "not found" in result.message


def test_slurm_state_condition_wrong_state(tmp_path: Path):
    """Test SlurmStateCondition when job is in wrong state."""
    ctx = _context(tmp_path, extra={"job_states_by_name": {"parent_job": "RUNNING"}})

    cond = SlurmStateCondition(
        SlurmStateConditionConfig(
            job_name="parent_job",
            state="COMPLETED",
        )
    )

    result = cond.check(ctx)
    assert not result.passed
    assert "RUNNING" in result.message
    assert "COMPLETED" in result.message


def test_log_pattern_condition_pattern_found(tmp_path: Path):
    """Test LogPatternCondition when pattern is found in log."""
    log_file = tmp_path / "train.log"
    log_file.write_text("Starting training...\nCheckpoint saved\nTraining complete\n")

    ctx = _context(tmp_path)

    cond = LogPatternCondition(
        LogPatternConditionConfig(
            log_path=str(log_file),
            pattern="Checkpoint saved",
            tail_lines=1000,
        )
    )

    result = cond.check(ctx)
    assert result.passed
    assert "pattern" in result.message.lower()
    assert "found" in result.message.lower()


def test_log_pattern_condition_pattern_not_found(tmp_path: Path):
    """Test LogPatternCondition when pattern is not in log."""
    log_file = tmp_path / "train.log"
    log_file.write_text("Starting training...\nTraining in progress...\n")

    ctx = _context(tmp_path)

    cond = LogPatternCondition(
        LogPatternConditionConfig(
            log_path=str(log_file),
            pattern="Checkpoint saved",
            tail_lines=1000,
        )
    )

    result = cond.check(ctx)
    assert not result.passed
    assert "not found" in result.message


def test_log_pattern_condition_log_missing(tmp_path: Path):
    """Test LogPatternCondition when log file doesn't exist."""
    log_file = tmp_path / "missing.log"
    ctx = _context(tmp_path)

    cond = LogPatternCondition(
        LogPatternConditionConfig(
            log_path=str(log_file),
            pattern="anything",
            tail_lines=1000,
        )
    )

    result = cond.check(ctx)
    assert not result.passed
    assert "does not exist" in result.message


def test_log_pattern_condition_regex_pattern(tmp_path: Path):
    """Test LogPatternCondition with regex pattern."""
    log_file = tmp_path / "train.log"
    log_file.write_text("Epoch 1 loss: 2.5\nEpoch 2 loss: 1.8\nEpoch 3 loss: 1.2\n")

    ctx = _context(tmp_path)

    cond = LogPatternCondition(
        LogPatternConditionConfig(
            log_path=str(log_file),
            pattern=r"Epoch \d+ loss: [0-9.]+",
            tail_lines=1000,
        )
    )

    result = cond.check(ctx)
    assert result.passed
