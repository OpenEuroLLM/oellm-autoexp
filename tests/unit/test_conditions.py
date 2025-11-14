from __future__ import annotations

import sys
import time
from pathlib import Path

from oellm_autoexp.monitor.conditions import (
    AlwaysTrueCondition,
    AlwaysTrueConditionConfig,
    CommandCondition,
    CommandConditionConfig,
    CompositeCondition,
    CompositeConditionConfig,
    ConditionContext,
    CooldownCondition,
    CooldownConditionConfig,
    FileExistsCondition,
    FileExistsConditionConfig,
    GlobExistsCondition,
    GlobExistsConditionConfig,
    MetadataCondition,
    MetadataConditionConfig,
    MaxAttemptsCondition,
    MaxAttemptsConditionConfig,
)
from oellm_autoexp.monitor.events import EventRecord


def _context(tmp_path: Path) -> ConditionContext:
    event = EventRecord(event_id="evt1", name="demo")
    return ConditionContext(event=event, job_metadata={"output_dir": str(tmp_path)}, attempts=0)


def test_always_true(tmp_path: Path) -> None:
    cond = AlwaysTrueCondition(AlwaysTrueConditionConfig(message="ok"))
    result = cond.check(_context(tmp_path))
    assert result.passed
    assert result.message == "ok"


def test_max_attempts(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.attempts = 1
    cond = MaxAttemptsCondition(MaxAttemptsConditionConfig(max_attempts=2))
    assert cond.check(ctx).passed
    ctx.attempts = 3
    assert not cond.check(ctx).passed


def test_cooldown(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.event.metadata["last_action_ts"] = time.time()
    cond = CooldownCondition(CooldownConditionConfig(cooldown_seconds=5))
    result = cond.check(ctx)
    assert result.waiting
    ctx.event.metadata["last_action_ts"] = time.time() - 10
    assert cond.check(ctx).passed


def test_file_exists(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    ctx = _context(tmp_path)
    cond = FileExistsCondition(
        FileExistsConditionConfig(
            path=str(target),
            blocking=False,
        )
    )
    waiting = cond.check(ctx)
    assert waiting.waiting
    target.write_text("ok", encoding="utf-8")
    assert cond.check(ctx).passed


def test_glob_exists(tmp_path: Path) -> None:
    pattern = tmp_path / "ckpt*.pt"
    ctx = _context(tmp_path)
    cond = GlobExistsCondition(
        GlobExistsConditionConfig(pattern=str(pattern), blocking=False, min_matches=1)
    )
    assert cond.check(ctx).waiting
    (tmp_path / "ckpt1.pt").write_text("1")
    assert cond.check(ctx).passed


def test_command_condition(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    cond = CommandCondition(
        CommandConditionConfig(command=[sys.executable, "-c", "raise SystemExit(0)"])
    )
    assert cond.check(ctx).passed
    bad = CommandCondition(
        CommandConditionConfig(command=[sys.executable, "-c", "raise SystemExit(1)"])
    )
    assert not bad.check(ctx).passed


def test_composite_condition(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    cond = CompositeCondition(
        CompositeConditionConfig(
            mode="all",
            conditions=[
                AlwaysTrueConditionConfig(),
                MaxAttemptsConditionConfig(max_attempts=1),
            ],
        )
    )
    assert cond.check(ctx).passed
    ctx.attempts = 5
    result = cond.check(ctx)
    assert result.status == "fail"


def test_metadata_condition(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.event.metadata["trigger"] = "checkpoint_evaluation"
    cond = MetadataCondition(MetadataConditionConfig(key="trigger", equals="checkpoint_evaluation"))
    assert cond.check(ctx).passed
    cond_fail = MetadataCondition(MetadataConditionConfig(key="trigger", equals="other"))
    assert cond_fail.check(ctx).status == "fail"
    cond_missing = MetadataCondition(MetadataConditionConfig(key="missing"))
    assert cond_missing.check(ctx).status == "fail"
