"Sweep expansion utilities."

from __future__ import annotations

import logging
from dataclasses import dataclass, field, MISSING
from itertools import product
from typing import Any
from collections.abc import Mapping


from omegaconf import OmegaConf

from oellm_autoexp.hydra_staged_sweep.config.schema import SweepConfig

LOGGER = logging.getLogger(__file__)


@dataclass(frozen=True)
class SweepPoint:
    """Represents a single expanded sweep entry."""

    index: int = field(default_factory=MISSING)
    parameters: Mapping[str, Any]
    group_path: tuple[int, ...] = field(default_factory=tuple)  # Track group hierarchy
    stage_path: tuple[bool, ...] = field(default_factory=tuple)  # Track group stage sweeps


def expand_sweep(config: SweepConfig) -> list[SweepPoint]:
    """Expand the sweep axes into concrete parameter sets.

    Supports composable groups format.
    """
    LOGGER.debug("Starting sweep expansion")

    base_values = dict(config.base_values)

    if config.groups is not None and config.type is not None:
        LOGGER.info("Using composable sweep format")
        points = _expand_composable_sweep(config, base_values)
    else:
        LOGGER.info("No sweep groups defined, returning base config")
        points = [SweepPoint(index=0, parameters=base_values)]

    LOGGER.info(f"Sweep expansion complete: {len(points)} points")
    return points


def _has_stage_key(values: Mapping[str, Any], stage_str: str) -> bool:
    return stage_str in values


def _nested_has_stage_key(config_dict: Mapping[str, Any], stage_str: str) -> bool:
    for key in ("groups", "params", "configs"):
        value = config_dict.get(key)
        if isinstance(value, Mapping) and stage_str in value:
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and stage_str in item:
                    return True
    return False


def _expand_composable_sweep(config: SweepConfig, base_values: dict[str, Any]) -> list[SweepPoint]:
    """New composable groups expansion with product/list modes."""
    groups = config.groups or []
    if not groups:
        return [SweepPoint(index=0, parameters=base_values)]

    # Expand all groups recursively
    group_combinations = _expand_group(
        group_type=config.type or "product",
        groups=groups,
        base_values=base_values,
        group_path=(),
        stage_path=(),
    )

    # Create SweepPoints with indices
    points = [
        SweepPoint(index=idx, parameters=params, group_path=path, stage_path=stage_path)
        for idx, (params, path, stage_path) in enumerate(group_combinations)
    ]

    if not points:
        points.append(SweepPoint(index=0, parameters=base_values, group_path=(), stage_path=()))

    return points


def _expand_group(
    group_type: str,
    groups: list[dict[str, Any]],
    base_values: dict[str, Any],
    group_path: tuple[int, ...],
    stage_path: tuple[bool, ...],
    stage_str: str = "stage",
) -> list[tuple[dict[str, Any], tuple[int, ...], tuple[bool, ...]]]:
    """Recursively expand a group with product or list composition.

    Supports 'defaults' field at group level that gets merged into all configs.

    Returns:
        List of (parameters, group_path, stage_path) tuples
    """
    if not groups:
        return [
            (dict(base_values), group_path + (0,), stage_path + ("stage" in list(base_values),))
        ]

    all_combinations: list[tuple[dict[str, Any], tuple[int, ...], tuple[bool, ...]]] = []

    for group_idx, group in enumerate(groups):
        current_path = group_path + (group_idx,)
        group_type_str = group.get("type", "product")

        # Extract defaults if present - these will be merged into all configs in this group
        group_defaults = group.get("defaults", {})

        # Merge defaults into base values for this group
        # Order: base_values < defaults < config-specific values
        group_base_values = dict(base_values)
        group_base_values.update(group_defaults)

        # Check if this is a nested group or a leaf group
        if "groups" in group:
            # Nested group - recursively expand
            nested_combinations = _expand_group(
                group_type=group_type_str,
                groups=group["groups"],
                base_values=group_base_values,
                group_path=current_path,
                stage_path=stage_path + (False,),
                stage_str=stage_str,
            )
            all_combinations.append(nested_combinations)
        elif "params" in group:
            # Leaf product group - expand parameters as cartesian product
            params = group["params"]
            cfg = OmegaConf.create(params or {})
            param_combinations = _product_dict(cfg)

            combinations = []
            for combo_idx, combo in enumerate(param_combinations):
                full_params = dict(group_base_values)
                full_params.update(combo)
                # Include combo_idx in group_path to distinguish different parameter combinations
                combinations.append((full_params, current_path + (combo_idx,)))
            if any(_has_stage_key(comb[0], stage_str) for comb in combinations):
                combinations = [
                    (
                        *comb,
                        stage_path + (False, True),
                    )
                    for comb in combinations
                ]
            else:
                combinations = [
                    (
                        *comb,
                        stage_path + (False, False),
                    )
                    for comb in combinations
                ]
            all_combinations.append(combinations)
        elif "configs" in group:
            # Leaf list group - each config is a separate point (no cross-product)
            configs = group["configs"]
            combinations = []
            for config_idx, config_dict in enumerate(configs):
                # Check if this config is itself a nested group
                if isinstance(config_dict, dict) and (
                    "groups" in config_dict or "params" in config_dict or "configs" in config_dict
                ):
                    stage_idx = _nested_has_stage_key(config_dict, stage_str)
                    # Nested group - recursively expand
                    # Wrap as a single-element group list to process correctly
                    nested_combos = _expand_group(
                        group_type="list",  # Always use list mode for single nested group
                        groups=[config_dict],  # Wrap the config_dict as a group
                        base_values=group_base_values,
                        group_path=current_path + (config_idx,),
                        stage_path=stage_path
                        + (
                            False,
                            stage_idx,
                        ),
                        stage_str=stage_str,
                    )
                    combinations.extend(nested_combos)
                else:
                    # Simple config dict
                    full_params = dict(group_base_values)
                    full_params.update(config_dict)
                    stage_idx = _has_stage_key(full_params, stage_str)
                    combinations.append(
                        (
                            full_params,
                            current_path + (config_idx,),
                            stage_path
                            + (
                                False,
                                stage_idx,
                            ),
                        )
                    )
            # if any(stage_str in list(comb[0]) for comb in combinations):
            #     combinations = [(*comb, stage_path + (True,)) for comb in combinations]
            # else:
            #     combinations = [(*comb, stage_path + (False,)) for comb in combinations]
            all_combinations.append(combinations)
        else:
            raise ValueError(f"Group must have 'groups', 'params', or 'configs': {group}")

    # Combine all group results based on composition mode
    if group_type == "product":
        # Cartesian product of all groups
        return _cartesian_product_groups(all_combinations)
    elif group_type == "list":
        # No cross-product - just concatenate
        result = []
        for group_combos in all_combinations:
            result.extend(group_combos)
        return result
    else:
        raise ValueError(f"Unknown group type: {group_type}. Must be 'product' or 'list'.")


def _cartesian_product_groups(
    groups: list[list[tuple[dict[str, Any], tuple[int, ...], tuple[bool, ...]]]],
) -> list[tuple[dict[str, Any], tuple[int, ...], tuple[bool, ...]]]:
    """Compute cartesian product of parameter groups.

    Merges parameters from each group and combines group paths.
    """
    if not groups:
        return []
    if len(groups) == 1:
        return groups[0]

    result = []
    for combination in product(*groups):
        # Merge all parameters
        merged_params = {}
        merged_path = ()
        merged_stage_path = ()
        for params, path, stage_path in combination:
            merged_params.update(params)
            merged_path = merged_path + path
            merged_stage_path = merged_stage_path + stage_path
        result.append((merged_params, merged_path, merged_stage_path))

    return result


def _product_dict(cfg: dict[str, list]):
    """Generate cartesian product of dict values."""
    combinations = [dict(zip(cfg.keys(), comb)) for comb in product(*cfg.values())]
    LOGGER.debug(f"Generated {len(combinations)} combinations from {list(cfg.keys())}")
    return combinations


__all__ = ["SweepPoint", "expand_sweep"]
