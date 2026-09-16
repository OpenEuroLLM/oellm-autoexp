"""Validation helpers for SBATCH scripts."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from collections.abc import Iterable

LOGGER = logging.getLogger(__name__)


class SlurmValidationError(RuntimeError):
    """Raised when script validation fails."""

    pass


def validate_job_script(
    rendered_path: str,
    job_name: str,
    required_tokens: Iterable[str] | None = None,
) -> None:
    """Validate a rendered SBATCH script.

    Args:
        rendered: The rendered script content.
        job_name: Expected job name in the script.
        required_tokens: Optional list of tokens that must appear in the script.

    Raises:
        SlurmValidationError: If validation fails.
    """
    rendered = Path(rendered_path).read_text()

    # Check for job name directive
    if f"--job-name={job_name}" not in rendered:
        raise SlurmValidationError(
            f"Script is missing job name directive: #SBATCH --job-name={job_name}"
        )

    # Check for unreplaced template placeholders
    if re.search(r"[^\$]\{[A-Za-z0-9_]+\}", rendered):
        raise SlurmValidationError("Unreplaced template placeholder detected")

    # Check for required tokens
    if required_tokens:
        for token in required_tokens:
            if token not in rendered:
                raise SlurmValidationError(f"Required token '{token}' missing from script")

    LOGGER.debug(f"Validated script for job: {job_name}")


__all__ = ["validate_job_script", "SlurmValidationError"]
