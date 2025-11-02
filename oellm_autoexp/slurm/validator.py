"""Validation helpers for SBATCH scripts."""

from __future__ import annotations

import re
from collections.abc import Iterable


class SlurmValidationError(RuntimeError):
    pass


def validate_job_script(
    rendered: str, job_name: str, required_tokens: Iterable[str] | None = None
) -> None:
    if f"#SBATCH --job-name={job_name}" not in rendered:
        raise SlurmValidationError("Rendered script missing job name directive")
    if re.search(r"\{[A-Za-z0-9_]+\}", rendered):
        raise SlurmValidationError("Unreplaced template placeholder detected")
    if required_tokens:
        for token in required_tokens:
            if token not in rendered:
                raise SlurmValidationError(f"Required token '{token}' missing from script")


__all__ = ["validate_job_script", "SlurmValidationError"]
