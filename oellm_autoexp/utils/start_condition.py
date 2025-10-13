"""Utilities for evaluating start-condition commands."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from oellm_autoexp.utils.run import run_with_tee


@dataclass(frozen=True)
class StartConditionResult:
    """Represents the outcome of a start condition command execution."""

    success: bool
    stdout: str
    stderr: str
    returncode: int

    def summary(self) -> str:
        if self.stdout:
            return self.stdout
        if self.stderr:
            return self.stderr
        return str(self.returncode)


def check_start_condition(command: str) -> StartConditionResult:
    """Execute ``command`` and return its success metadata."""

    proc = run_with_tee(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    success = False
    if stdout:
        try:
            success = int(stdout) == 1
        except ValueError:
            success = False
    if not success:
        success = proc.returncode == 0 and not stdout

    return StartConditionResult(
        success=success,
        stdout=stdout,
        stderr=stderr,
        returncode=proc.returncode,
    )


def wait_for_start_condition(
    command: str,
    *,
    interval_seconds: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> StartConditionResult:
    """Block until ``command`` succeeds according to ``check_start_condition``."""

    if not command:
        return StartConditionResult(True, "", "", 0)

    interval = max(1, int(interval_seconds or 60))
    log = logger or logging.getLogger(__name__)

    while True:
        result = check_start_condition(command)
        if result.success:
            return result
        log.info(
            "Start condition not met (command=%r, result=%s); retrying in %ss",
            command,
            result.summary(),
            interval,
        )
        sleep_fn(interval)


def resolve_start_condition_interval(
    job_interval_seconds: Optional[int],
    monitor_config: Any,
) -> int:
    """Determine an appropriate interval between start-condition retries."""

    if job_interval_seconds is not None:
        try:
            return max(1, int(job_interval_seconds))
        except (TypeError, ValueError):
            pass

    for attr in (
        "start_condition_interval_seconds",
        "check_interval_seconds",
        "poll_interval_seconds",
    ):
        value = getattr(monitor_config, attr, None)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return 60


__all__ = [
    "StartConditionResult",
    "check_start_condition",
    "wait_for_start_condition",
    "resolve_start_condition_interval",
]
