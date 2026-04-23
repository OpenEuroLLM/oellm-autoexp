"""Shell helper utilities."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence

from oellm_autoexp.utils.run import run_with_tee

LOGGER = logging.getLogger(__name__)


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return run_with_tee(argv, check=False, capture_output=True, text=True)


__all__ = ["run_command"]
