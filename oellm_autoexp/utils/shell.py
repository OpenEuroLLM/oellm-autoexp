"""Shell helper utilities."""

from __future__ import annotations

import subprocess
from typing import Sequence

import oellm_autoexp.utils.run


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return oellm_autoexp.utils.run.run_with_tee(argv, check=False, capture_output=True, text=True)


__all__ = ["run_command"]
