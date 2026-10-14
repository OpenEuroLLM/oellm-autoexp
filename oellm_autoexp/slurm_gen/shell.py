"""Shell utilities for running SLURM commands."""

from __future__ import annotations

import errno
import logging
import subprocess
import time
from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

# Errnos that mean the process could not be *spawned* right now but might succeed
# on retry: a busy/shared SLURM login node at its per-user process limit
# (RLIMIT_NPROC) makes fork() fail with EAGAIN ("Resource temporarily
# unavailable"); ENOMEM is the transient out-of-memory-at-fork variant. Permanent
# errors (e.g. ENOENT "command not found") are NOT in this set and propagate
# immediately. Retrying is safe because these failures happen at fork(), before
# exec(), so the command never ran (no half-submitted sbatch to duplicate).
_RETRYABLE_SPAWN_ERRNOS = frozenset(
    {errno.EAGAIN, errno.ENOMEM, getattr(errno, "EWOULDBLOCK", errno.EAGAIN)}
)

# Never block forever on a SLURM client command. A wedged slurmctld makes
# `squeue` hang indefinitely, and with no timeout that hangs the whole monitor
# in a way nothing can see: the process is alive, the poll never returns, and no
# log line is written. Generous enough for a loaded controller, short enough
# that the loop's own error handling gets a turn.
DEFAULT_COMMAND_TIMEOUT_S = 120.0


def run_command(
    argv: Sequence[str],
    *,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = DEFAULT_COMMAND_TIMEOUT_S,
    max_spawn_retries: int = 5,
    spawn_retry_backoff: float = 1.0,
) -> subprocess.CompletedProcess[str]:
    """Run a command and return the result.

    Retries only on transient failures to *spawn* the child process (fork()
    failing with EAGAIN/ENOMEM on a saturated login node), with exponential
    backoff. A non-zero exit (``check``), timeout, or permanent OSError (e.g.
    command not found) is not retried and propagates normally.

    Args:
        argv: Command and arguments to run.
        check: Raise CalledProcessError on non-zero exit.
        capture_output: Capture stdout and stderr.
        text: Return text instead of bytes.
        timeout: Timeout in seconds; ``None`` waits forever (rarely what you
            want -- see DEFAULT_COMMAND_TIMEOUT_S).
        max_spawn_retries: Max retries when fork() fails transiently.
        spawn_retry_backoff: Base seconds for exponential backoff between retries.

    Returns:
        CompletedProcess with stdout, stderr, and returncode.
    """
    LOGGER.debug("Running command: %s", " ".join(argv))
    attempt = 0
    while True:
        try:
            result = subprocess.run(
                argv,
                check=check,
                capture_output=capture_output,
                text=text,
                timeout=timeout,
            )
            break
        except OSError as exc:
            if exc.errno not in _RETRYABLE_SPAWN_ERRNOS or attempt >= max_spawn_retries:
                raise
            delay = spawn_retry_backoff * (2**attempt)
            attempt += 1
            LOGGER.warning(
                "Failed to spawn '%s' (%s); retry %d/%d in %.1fs",
                " ".join(argv),
                exc,
                attempt,
                max_spawn_retries,
                delay,
            )
            time.sleep(delay)
    LOGGER.debug("Command returned: %d", result.returncode)
    return result


__all__ = ["DEFAULT_COMMAND_TIMEOUT_S", "run_command"]
