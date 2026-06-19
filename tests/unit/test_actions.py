"""Test oellm-specific custom monitor actions."""

from __future__ import annotations

from pathlib import Path

from oellm_autoexp.monitor.actions import (
    ActionContext,
    AppendToFileAction,
    AppendToFileActionConfig,
    EventRecord,
    LogEvent,
    LogEventConfig,
)
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


def _append_context(node: str) -> ActionContext:
    """Context emulating a LogEvent that extracted a failing node name."""
    event = EventRecord(
        event_id="evt-node",
        name="comm_failure_exclude_node",
        source="log",
        payload={"node": node, "match": f"failed on node {node}: Communication connection failure"},
    )
    return ActionContext(event=event)


def test_append_to_file_action_appends_templated_node(tmp_path: Path):
    target = tmp_path / "nested" / "exclude.txt"
    config = AppendToFileActionConfig(path=str(target), content="{node}")
    result = AppendToFileAction(config).execute(_append_context("lrdn0417"))

    assert result.status == "success"
    assert result.metadata["appended"] is True
    # parent dirs created on demand
    assert target.read_text().splitlines() == ["lrdn0417"]


def test_append_to_file_action_dedups(tmp_path: Path):
    target = tmp_path / "exclude.txt"
    target.write_text("lrdn0417\n")
    config = AppendToFileActionConfig(path=str(target), content="{node}")
    result = AppendToFileAction(config).execute(_append_context("lrdn0417"))

    assert result.metadata["appended"] is False
    # not duplicated
    assert target.read_text().splitlines() == ["lrdn0417"]


def test_append_to_file_action_appends_distinct_nodes(tmp_path: Path):
    target = tmp_path / "exclude.txt"
    config = AppendToFileActionConfig(path=str(target), content="{node}")
    action = AppendToFileAction(config)
    action.execute(_append_context("lrdn0417"))
    action.execute(_append_context("lrdn0001"))

    assert target.read_text().splitlines() == ["lrdn0417", "lrdn0001"]


def test_log_event_comm_failure_chains_into_exclude_append(tmp_path: Path):
    """A 'Communication connection failure' log line should extract the node
    and append it to the exclusion file (the LogEvent -> AppendToFileAction
    chain)."""
    target = tmp_path / "exclude.txt"
    event_cfg = LogEventConfig(
        name="comm_failure_exclude_node",
        pattern_type="regex",
        pattern=r"failed on node ([^:\s]+): Communication connection failure",
        extract_groups={"node": 1},
        match_once=False,
        action=AppendToFileActionConfig(path=str(target), content="{node}"),
    )

    log_text = (
        "some preamble\nfailed on node lrdn0417: Communication connection failure\ntrailing line\n"
    )
    triggers = LogEvent(event_cfg).check_triggers(log_text)
    assert len(triggers) == 1
    assert triggers[0]["node"] == "lrdn0417"

    action = AppendToFileAction(event_cfg.action)
    action.execute(
        ActionContext(
            event=EventRecord(event_id="e", name=event_cfg.name, source="log", payload=triggers[0])
        )
    )
    assert target.read_text().splitlines() == ["lrdn0417"]


def test_append_to_file_action_empty_content_is_noop(tmp_path: Path):
    target = tmp_path / "exclude.txt"
    config = AppendToFileActionConfig(path=str(target), content="")
    result = AppendToFileAction(config).execute(_append_context("lrdn0417"))

    assert result.status == "failed"
    assert not target.exists()
