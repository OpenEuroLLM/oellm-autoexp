"""Sweep expansion utilities."""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from itertools import product
from typing import Any
from collections.abc import Mapping


from omegaconf import OmegaConf

from oellm_autoexp.config.schema import SweepConfig


@dataclass(frozen=True)
class SweepPoint:
    """Represents a single expanded sweep entry."""

    index: int = field(default_factory=MISSING)
    parameters: Mapping[str, Any]


def expand_sweep(config: SweepConfig) -> list[SweepPoint]:
    """Expand the sweep axes into concrete parameter sets."""

    base_values = dict(config.base_values)

    points: list[SweepPoint] = []
    for grid in config.grids:
        cfg = OmegaConf.create(grid or {})
        for combination in _product_dict(cfg):
            flat: dict[str, Any] = dict(base_values)
            for key, value in combination.items():
                flat[key] = value
            # Apply optional filter expression: only keep combinations satisfying the filter
            if config.filter:
                try:
                    if not eval(config.filter, {}, flat):
                        continue
                except Exception as e:
                    raise ValueError(f"Error evaluating sweep filter '{config.filter}': {e}")
            points.append(SweepPoint(index=len(points), parameters=flat))
    if not points:
        points.append(SweepPoint(index=0, parameters=base_values))
    return points


def _product_dict(cfg: dict[str, list]):
    combinations = [dict(zip(cfg.keys(), comb)) for comb in product(*cfg.values())]
    print(combinations)
    return combinations


__all__ = ["SweepPoint", "expand_sweep"]
