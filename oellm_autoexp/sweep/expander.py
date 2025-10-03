"""Sweep expansion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf

from oellm_autoexp.config.schema import SweepConfig


@dataclass(frozen=True)
class SweepPoint:
    """Represents a single expanded sweep entry."""

    index: int
    parameters: Mapping[str, Any]


def expand_sweep(config: SweepConfig) -> List[SweepPoint]:
    """Expand the sweep axes into concrete parameter sets."""

    cfg = OmegaConf.create(config.axes or {})
    base_values = dict(config.base_values)

    points: List[SweepPoint] = []
    for idx, combination in enumerate(_product_recursive(cfg)):
        flat: Dict[str, Any] = dict(base_values)
        for key_path, value in combination.items():
            key = _flatten_key(key_path)
            flat[key] = value
        points.append(SweepPoint(index=idx, parameters=flat))
    if not points:
        points.append(SweepPoint(index=0, parameters=base_values))
    return points


def _flatten_key(path: Tuple[str, ...]) -> str:
    return ".".join(path)


def _product_recursive(cfg: Any) -> List[Dict[Tuple[str, ...], Any]]:
    if isinstance(cfg, (str, int, float, bool)):
        return [{tuple(): cfg}]
    if isinstance(cfg, ListConfig):
        if all(isinstance(item, DictConfig) and len(item) == 1 for item in cfg):
            results: List[Dict[Tuple[str, ...], Any]] = []
            for element in cfg:
                key = _first_key(element)
                value = _first_value(element)
                sub_results = [_add_key(key, sub) for sub in _product_recursive(value)]
                results.extend(sub_results)
            return results
        if all(isinstance(item, (str, int, float, bool)) for item in cfg):
            return [{tuple(): item} for item in cfg]
        raise ValueError("List entries must be scalars or single-key mappings")
    if isinstance(cfg, DictConfig):
        accum: List[List[Dict[Tuple[str, ...], Any]]] = []
        for key, value in cfg.items():
            sub = [_add_key(key, sub) for sub in _product_recursive(value)]
            accum.append(sub)
        merged = [_merge(combo) for combo in product(*accum)]
        return merged
    if cfg is None:
        return [{}]
    raise ValueError(f"Unsupported sweep node type: {type(cfg)!r}")


def _add_key(key: str, combination: Dict[Tuple[str, ...], Any]) -> Dict[Tuple[str, ...], Any]:
    return {(key,) + path: value for path, value in combination.items()}


def _first_key(cfg: DictConfig) -> str:
    return next(iter(cfg.keys()))


def _first_value(cfg: DictConfig) -> Any:
    return next(iter(cfg.values()))


def _merge(nodes: Iterable[Dict[Tuple[str, ...], Any]]) -> Dict[Tuple[str, ...], Any]:
    merged: Dict[Tuple[str, ...], Any] = {}
    for node in nodes:
        merged.update(node)
    return merged


__all__ = ["SweepPoint", "expand_sweep"]
