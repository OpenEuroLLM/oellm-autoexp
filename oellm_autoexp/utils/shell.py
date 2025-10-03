"""Shell helper utilities."""

from __future__ import annotations

import subprocess
from typing import Sequence


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)


__all__ = ["run_command"]

