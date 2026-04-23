"""Test oellm-specific custom monitor actions."""

from __future__ import annotations

from pathlib import Path

from oellm_autoexp.monitor.actions import ActionContext, EventRecord
from oellm_autoexp.config.actions import RunAutoexpAction, RunAutoexpActionConfig


def _context(tmp_path: Path) -> ActionContext:
    """Create a test action context."""
    event = EventRecord(event_id="evt1", name="test", source="test")
    return ActionContext(
        event=event,
        job_metadata={"session_id": "test_session"},
        workspace=tmp_path,
    )


def test_run_autoexp_action_config():
    """Test RunAutoexpActionConfig can be instantiated."""
    config = RunAutoexpActionConfig(
        script="scripts/run_autoexp.py",
        config_path="/tmp/test.yaml",
        overrides=["backend=test"],
        no_monitor=True,
    )
    assert config.script == "scripts/run_autoexp.py"
    assert config.no_monitor is True
    assert len(config.overrides) == 1


def test_run_autoexp_action_creates_command(tmp_path: Path):
    """Test that RunAutoexpAction builds the correct command structure."""
    config = RunAutoexpActionConfig(
        script="scripts/run_autoexp.py",
        config_path="{output_dir}/config.yaml",
        overrides=["stage={stage}", "job.retry_limit=5"],
        no_monitor=True,
    )
    action = RunAutoexpAction(config)

    # The action should be instantiable
    assert action.config == config

    # Note: Full execution testing requires actual script files and would be
    # better suited for integration tests. Unit tests verify config handling.


def test_run_autoexp_action_execution_dry_run(tmp_path: Path):
    """Test RunAutoexpAction returns appropriate result on execution
    failure."""
    ctx = _context(tmp_path)
    ctx.job_metadata["output_dir"] = str(tmp_path)

    # Create a simple failing script
    script = tmp_path / "fail_script.py"
    script.write_text("import sys; sys.exit(1)")

    config = RunAutoexpActionConfig(
        script=str(script),
        config_path=None,
        overrides=[],
        no_monitor=True,
    )
    action = RunAutoexpAction(config)
    result = action.execute(ctx)

    # Should return failed status
    assert result.status == "failed"
    assert "exited 1" in result.message
