"""Normalization helpers shared by config loaders."""

from __future__ import annotations

from pathlib import Path

from . import schema


def ensure_monitoring_state_dir(root: schema.RootConfig) -> None:
    """Assign a deterministic monitoring_state_dir when unspecified."""

    if root.project.monitoring_state_dir is not None:
        return

    base = Path(root.project.base_output_dir)
    stable_base = (
        base.parent
        if base.name.split("_")[-1].isdigit() and len(base.name.split("_")[-1]) >= 8
        else base
    )
    root.project.monitoring_state_dir = stable_base / "monitoring_state"


__all__ = ["ensure_monitoring_state_dir"]
