from __future__ import annotations

import sys
from pathlib import Path

from oellm_autoexp.monitor.actions import (
    ActionContext,
    LogAction,
    LogActionConfig,
    LogMessageAction,
    LogMessageActionConfig,
    PublishEventAction,
    PublishEventActionConfig,
    RestartAction,
    RestartActionConfig,
    RunCommandAction,
    RunCommandActionConfig,
)
from oellm_autoexp.monitor.events import EventRecord


def _context(tmp_path: Path) -> ActionContext:
    event = EventRecord(event_id="evt1", name="demo", source="test")
    return ActionContext(event=event, job_metadata={}, workspace=tmp_path)


def test_run_command_action(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    action = RunCommandAction(
        RunCommandActionConfig(command=[sys.executable, "-c", "raise SystemExit(0)"])
    )
    assert action.execute(ctx).status == "success"


def test_restart_action_returns_retry(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    action = RestartAction(RestartActionConfig(reason="retry please"))
    result = action.execute(ctx)
    assert result.status == "retry"
    assert "retry" in result.message


def test_publish_event_action(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    action = PublishEventAction(
        PublishEventActionConfig(event_name="checkpoint_ready", metadata={"foo": "bar"})
    )
    result = action.execute(ctx)
    assert result.metadata["publish_event"]["name"] == "checkpoint_ready"


def test_log_action_renders_template(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.job_metadata["name"] = "demo"
    action = LogAction(LogActionConfig(message="Job {name} ok"))
    result = action.execute(ctx)
    assert "demo" in result.message


def test_log_message_action_alias(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    action = LogMessageAction(LogMessageActionConfig(message="legacy ok"))
    result = action.execute(ctx)
    assert result.status == "success"
