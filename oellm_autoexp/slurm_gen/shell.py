"""Shell utilities for running SLURM commands."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)


def run_command(
    argv: Sequence[str],
    *,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and return the result.

    Args:
        argv: Command and arguments to run.
        check: Raise CalledProcessError on non-zero exit.
        capture_output: Capture stdout and stderr.
        text: Return text instead of bytes.
        timeout: Timeout in seconds.

    Returns:
        CompletedProcess with stdout, stderr, and returncode.
    """
    LOGGER.debug("Running command: %s", " ".join(argv))
    result = subprocess.run(
        argv,
        check=check,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
    )
    LOGGER.debug("Command returned: %d", result.returncode)
    return result


__all__ = ["run_command"]
