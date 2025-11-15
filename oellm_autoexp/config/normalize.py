"""Normalization helpers shared by config loaders."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from . import schema


def normalize_config_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Apply legacy transforms and flatten sections before compoconf
    parsing."""

    normalized: dict[str, Any] = deepcopy(dict(data))

    _normalize_legacy_sections(normalized)
    return normalized


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


def _normalize_legacy_sections(data: dict[str, Any]) -> None:
    """Flatten legacy `implementation` wrappers for backend and monitoring."""

    for key in ("backend", "monitoring"):
        section = data.get(key)
        if not isinstance(section, dict):
            continue
        impl = section.get("implementation")
        if isinstance(impl, dict):
            merged = {**impl}
            for sub_key, value in section.items():
                if sub_key != "implementation":
                    merged[sub_key] = value
            section = merged
            data[key] = section

        if key == "backend":
            namespace = section.get("megatron")
            if isinstance(namespace, dict):
                for sub_key, value in namespace.items():
                    section.setdefault(sub_key, value)
                section.pop("megatron", None)

    slurm_section = data.get("slurm")
    if isinstance(slurm_section, dict):
        client = slurm_section.get("client")
        if isinstance(client, dict):
            name = client.get("class_name")
            if name == "FakeSlurm":
                client["class_name"] = "FakeSlurmClient"
            elif name == "RealSlurm":
                client["class_name"] = "SlurmClient"


__all__ = ["normalize_config_data", "ensure_monitoring_state_dir"]
