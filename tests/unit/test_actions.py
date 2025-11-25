from __future__ import annotations

import sys
from pathlib import Path

from oellm_autoexp.monitor.action_queue import ActionQueue
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
    event = EventRecord(event_id="evt1", name="demo")
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


def test_action_queue_roundtrip(tmp_path: Path) -> None:
    queue_dir = tmp_path / "actions"
    queue = ActionQueue(queue_dir)
    record = queue.enqueue("LogAction", {"message": "hi"}, event_id="evt", metadata={"k": "v"})
    event_dir = queue_dir / "evt"
    assert (event_dir / f"{record.queue_id}.json").exists()
    listed = queue.list()
    assert len(listed) == 1
    assert listed[0].queue_id == record.queue_id
    claimed = queue.claim_next()
    assert claimed is not None
    queue.mark_done(claimed.queue_id, status="done", result={"status": "ok"})
    assert not event_dir.exists() or not any(event_dir.glob("*.json"))
    assert not queue.list()


def test_action_queue_retry(tmp_path: Path) -> None:
    queue = ActionQueue(tmp_path)
    record = queue.enqueue("LogAction", {"message": "hi"}, event_id="evt")
    claimed = queue.claim_next()
    assert claimed is not None
    assert claimed.status == "running"
    assert queue.retry(record.queue_id)
    reloaded = queue.load(record.queue_id)
    assert reloaded is not None
    assert reloaded.status == "pending"
