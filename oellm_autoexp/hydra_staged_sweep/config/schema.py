"""Configuration dataclasses for hydra_staged_sweep.

Defines the core structures required for sweep expansion and staging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from compoconf import ConfigInterface


@dataclass(kw_only=True)
class SweepConfig(ConfigInterface):
    """Sweep expansion settings.

    Supports composable groups format (product/list).
    """

    class_name: str = "Sweep"
    type: Literal["product", "list"] | None = None  # "product" or "list"
    groups: list[dict[str, Any]] | None = None
    base_values: dict[str, Any] = field(default_factory=dict)
    store_sweep_json: bool = True
    filter: bool | str | None = True
    list_composition: list[str] = field(default_factory=list)  # Parameters to accumulate as lists


@dataclass(kw_only=True)
class StagedSweepRoot(ConfigInterface):
    """Minimal interface expected by hydra_staged_sweep.

    External applications should inherit from this or define a
    compatible structure that includes these fields.
    """

    sweep: SweepConfig = field(default_factory=SweepConfig)
    stage: str = ""
    # These are set internally during resolution
    index: int | tuple[int] = 0
    sibling: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ConfigSetup:
    pwd: str = "."
    config_name: str | None = None
    config_path: str | None = None
    config_dir: str | None = None
    overrides: list[str] = field(default_factory=list)

    def __post_init__(self):
        assert self.config_path is not None or (
            self.config_name is not None and self.config_dir is not None
        )
