"""Build execution plans from sweep outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class JobPlan:
    """Generic job plan containing the resolved configuration and metadata."""

    config: Any
    parameters: list[str] = field(default_factory=list)
    sibling_pattern: str | None = None
    stage_name: str | None = None


__all__ = ["JobPlan"]
