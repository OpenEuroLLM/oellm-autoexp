"""Pure OmegaConf DAG-based sweep resolution.

This is the simplified v2 implementation that uses OmegaConf for ALL interpolations,
including sibling references. No custom template resolution - just pure OmegaConf.

Key insight: Use `${sibling.stable.output_dir}` syntax in YAML and add sibling data
to the OmegaConf namespace during resolution. OmegaConf handles everything.

See docs/sweep_resolution_ordering.md for design rationale (Option 5).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any

import networkx as nx
from compoconf import asdict
from omegaconf import DictConfig, ListConfig, OmegaConf

from oellm_autoexp.hydra_staged_sweep.config.schema import StagedSweepRoot, ConfigSetup
from oellm_autoexp.hydra_staged_sweep.config.loader import load_config_reference
from oellm_autoexp.hydra_staged_sweep.expander import SweepPoint
from oellm_autoexp.hydra_staged_sweep.planner import JobPlan

LOGGER = logging.getLogger(__file__)


@dataclass(frozen=True)
class SiblingIndex:
    patterns_by_idx: dict[int, set[str]]
    match_key_by_idx: dict[int, tuple[int, ...]]
    index_by_key: dict[tuple[int, ...], list[int]]
    stage_mask: tuple[bool, ...]


def extract_sibling_patterns(parameters: dict[str, Any]) -> set[str]:
    """Extract stage patterns from escaped sibling references."""
    patterns = set()
    sibling_regex = re.compile(r"\$\{sibling\.([^.}]+)\.")

    def scan_value(value: Any) -> None:
        if isinstance(value, str):
            for match in sibling_regex.finditer(value):
                patterns.add(match.group(1))
        elif isinstance(value, dict):
            for v in value.values():
                scan_value(v)
        elif isinstance(value, list):
            for item in value:
                scan_value(item)

    scan_value(parameters)
    LOGGER.debug(f"Extracted sibling patterns: {patterns}")
    return patterns


def _match_key(point: SweepPoint, stage_mask: tuple[bool, ...]) -> tuple[int, ...]:
    """Build a matching key that ignores globally stage-flagged path
    segments."""
    return tuple(
        group_idx
        for group_idx, is_stage in zip_longest(point.group_path, stage_mask, fillvalue=False)
        if not is_stage
    )


def _build_sibling_index(points: Mapping[int, SweepPoint]) -> SiblingIndex:
    patterns_by_idx: dict[int, set[str]] = {}
    match_key_by_idx: dict[int, tuple[int, ...]] = {}
    index_by_key: dict[tuple[int, ...], list[int]] = defaultdict(list)
    max_depth = 0

    for point in points.values():
        max_depth = max(max_depth, len(point.group_path), len(point.stage_path))

    stage_mask = [False] * max_depth
    for point in points.values():
        for idx, is_stage in enumerate(point.stage_path):
            if is_stage:
                stage_mask[idx] = True
    stage_mask_tuple = tuple(stage_mask)

    for idx, point in points.items():
        patterns_by_idx[idx] = extract_sibling_patterns(point.parameters)
        match_key = _match_key(point, stage_mask_tuple)
        match_key_by_idx[idx] = match_key
        index_by_key[match_key].append(idx)

    return SiblingIndex(
        patterns_by_idx=patterns_by_idx,
        match_key_by_idx=match_key_by_idx,
        index_by_key=dict(index_by_key),
        stage_mask=stage_mask_tuple,
    )


def _resolve_filter_from_context(filter_expr: Any, context: Mapping[str, Any]) -> bool:
    if filter_expr is None:
        return True
    if isinstance(filter_expr, bool):
        return filter_expr
    if not isinstance(filter_expr, str):
        raise ValueError("sweep.filter must resolve to a bool.")
    cfg = OmegaConf.create({**context, "sweep": {"filter": filter_expr}})
    try:
        resolved = OmegaConf.to_container(cfg, resolve=True)
    except Exception as exc:
        raise ValueError(f"sweep.filter must resolve to a bool: {exc}") from exc
    if not isinstance(resolved, dict):
        raise ValueError("sweep.filter must resolve to a bool.")
    result = resolved.get("sweep", {}).get("filter")
    if not isinstance(result, bool):
        raise ValueError("sweep.filter must resolve to a bool.")
    return result


def _collect_group_filters(
    groups: list[dict[str, Any]] | None, group_path: tuple[int, ...]
) -> list[Any]:
    if not groups:
        return []

    filters: list[Any] = []
    cursor = 0

    def walk(group_list: list[dict[str, Any]]) -> None:
        nonlocal cursor, filters  # noqa: F824
        for group_idx, group in enumerate(group_list):
            if cursor >= len(group_path):
                raise ValueError("Group path does not match sweep groups.")
            if group_path[cursor] != group_idx:
                raise ValueError("Group path does not match sweep groups.")
            cursor += 1

            group_type = group.get("type", "product")
            if group_type == "product" and "filter" in group:
                filters.append(group.get("filter"))

            if "groups" in group:
                walk(group["groups"])
            elif "params" in group:
                if cursor >= len(group_path):
                    raise ValueError("Group path does not match sweep groups.")
                cursor += 1
            elif "configs" in group:
                if cursor >= len(group_path):
                    raise ValueError("Group path does not match sweep groups.")
                config_idx = group_path[cursor]
                cursor += 1
                configs = group["configs"]
                if not isinstance(configs, list) or config_idx >= len(configs):
                    raise ValueError("Group path does not match sweep groups.")
                config_dict = configs[config_idx]
                if isinstance(config_dict, dict) and (
                    "groups" in config_dict or "params" in config_dict or "configs" in config_dict
                ):
                    walk([config_dict])
            else:
                raise ValueError("Group must have 'groups', 'params', or 'configs'.")

    walk(groups)
    if cursor != len(group_path):
        raise ValueError("Group path does not match sweep groups.")
    return filters


def find_sibling_by_group_path(
    point: SweepPoint,
    all_points: Mapping[int, SweepPoint] | list[SweepPoint],
    stage_pattern: str,
    sibling_index: SiblingIndex | None = None,
) -> SweepPoint | None:
    """Find sibling with matching hyperparameters."""
    points_dict = (
        all_points if isinstance(all_points, Mapping) else {p.index: p for p in all_points}
    )
    index = sibling_index or _build_sibling_index(points_dict)

    if not index.patterns_by_idx.get(point.index):
        return None

    siblings = []
    point_key = index.match_key_by_idx.get(point.index, ())

    for candidate_idx in index.index_by_key.get(point_key, []):
        if candidate_idx == point.index:
            continue
        siblings.append(points_dict[candidate_idx])

    LOGGER.debug(
        "Got siblings for point %s: %s",
        point.index,
        [s.parameters.get("stage", "") for s in siblings],
    )
    matched_sibling = [
        sibling
        for sibling in siblings
        if re.match(stage_pattern, sibling.parameters.get("stage", ""))
    ]
    if matched_sibling:
        if len(matched_sibling) > 1:
            LOGGER.warning(f"Multiple matched siblings for {point}, {stage_pattern}")
        return matched_sibling[0]
    return None


def build_dependency_dag_from_points(
    points: dict[int, SweepPoint],
    sibling_index: SiblingIndex | None = None,
) -> nx.DiGraph:
    """Build dependency DAG from sweep points."""
    LOGGER.debug(f"Building dependency DAG from {len(points)} points")
    dag = nx.DiGraph()
    index = sibling_index or _build_sibling_index(points)

    for point in points.values():
        dag.add_node(point.index)

    edges_added = 0
    for point in points.values():
        sibling_deps = index.patterns_by_idx.get(point.index, set())

        for stage_pattern in sibling_deps:
            try:
                sibling = find_sibling_by_group_path(
                    point, points, stage_pattern, sibling_index=index
                )
                if sibling:
                    dag.add_edge(sibling.index, point.index)
                else:
                    LOGGER.warning(
                        f"No sibling found for requested stage_pattern: {stage_pattern} of point {point}"
                    )
                edges_added += 1
            except ValueError:
                pass

    LOGGER.info(f"Built DAG with {len(points)} nodes and {edges_added} edges")
    return dag


def config_to_cmdline(
    cfg_dict: dict,
    override: str = "",
    prefix="",
) -> list[str]:
    cmdline_opts = []

    def dict_to_cmdlines(dct: dict | list | str | int | float, prefix: str = ""):
        cmdlines = []

        if isinstance(dct, (dict, DictConfig, Mapping)):
            for sub_cfg in dct:
                newprefix = (prefix + "." if prefix else "") + sub_cfg
                cmdlines += dict_to_cmdlines(dct[sub_cfg], prefix=newprefix)

        elif isinstance(dct, (list, ListConfig, Sequence)) and not isinstance(dct, (str, bytes)):
            cmdlines.append(override + prefix + "=[" + ",".join(map(str, range(len(dct)))) + "]")
            for n, sub_cfg in enumerate(dct):
                cmdlines += dict_to_cmdlines(
                    sub_cfg,
                    prefix=(prefix + "." if prefix else "") + str(n),
                )
        elif dct is None:
            cmdlines.append(override + prefix + "=null")
        else:
            if isinstance(dct, str):
                if not re.match(r"\[[A-Za-z][A-Za-z0-9,]*\]", dct):
                    dct = dct.replace('"', '\\"')
                    dct = f'"{dct}"'
            cmdlines.append(override + prefix + "=" + str(dct))
        return cmdlines

    cmdline_opts = dict_to_cmdlines(cfg_dict, prefix=prefix)
    LOGGER.debug("GENERATED CMDLINE OPTS {config_yaml}, {cmdline_opts}")
    return cmdline_opts


def param_to_cmdlines(key: str, val: Any, prefix: str = ""):
    if isinstance(val, str):
        if re.match(r"\[[A-Za-z][A-Za-z0-9,]*\]", val):
            return [f"{prefix}{key}={val}"]
        val = val.replace('"', '\\"')
        return [f'{prefix}{key}="{val}"']
    else:
        return config_to_cmdline(
            val,
            override="++",
            prefix=key,
        )


def resolve_sweep_with_dag(
    config: StagedSweepRoot,
    points: list[SweepPoint] | dict[int, SweepPoint],
    config_setup: ConfigSetup,
    config_class: type = StagedSweepRoot,
) -> list[JobPlan]:
    """Pure OmegaConf resolution with DAG ordering."""
    LOGGER.info(f"Starting DAG resolution for {len(points)} sweep points")

    if isinstance(points, list):
        points_dict = {p.index: p for p in points}
    else:
        points_dict = points

    sibling_index = _build_sibling_index(points_dict)
    dag = build_dependency_dag_from_points(points_dict, sibling_index=sibling_index)

    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        LOGGER.error(f"Circular dependencies detected: {cycles}")
        raise ValueError(f"Circular dependencies detected: {cycles}")

    ordered_indices = list(nx.topological_sort(dag))
    LOGGER.debug(f"Topological order: {ordered_indices}")

    resolved_jobs = {}
    filtered_jobs = {}
    base_context = asdict(config)
    base_context = {k: v for k, v in base_context.items() if k not in ("sweep", "sibling")}
    sweep_filter_expr = config.sweep.filter

    for point_idx in ordered_indices:
        point = points_dict[point_idx]
        try:
            group_filters = _collect_group_filters(config.sweep.groups, point.group_path)
        except ValueError as exc:
            raise ValueError(f"Unable to match group_path for filtering: {exc}") from exc

        filter_exprs = [sweep_filter_expr, *group_filters]

        sibling_patterns = sibling_index.patterns_by_idx.get(point_idx, set())
        sibling_jobs = {}
        for pattern in sibling_patterns:
            sibling_point = find_sibling_by_group_path(
                point, points_dict, pattern, sibling_index=sibling_index
            )
            if sibling_point and sibling_point.index in resolved_jobs:
                sibling_jobs[pattern] = resolved_jobs[sibling_point.index]

        sibling_job_configs = {
            sibling_pattern: asdict(
                load_config_reference(
                    config_dir=config_setup.config_dir,
                    config_path=config_setup.config_path,
                    config_name=config_setup.config_name,
                    overrides=list(config_setup.overrides) + sibling_job.parameters,
                    config_class=config_class,
                )
            )
            for sibling_pattern, sibling_job in sibling_jobs.items()
        }

        for sibling_pattern in sibling_job_configs:
            if "sweep" in sibling_job_configs[sibling_pattern]:
                del sibling_job_configs[sibling_pattern]["sweep"]

        cmdline_overrides_siblings = config_to_cmdline(
            {
                "sibling": {
                    sibling_job.get("stage", "unknown"): sibling_job
                    for sibling_job in sibling_job_configs.values()
                }
            },
            override="++",
        )

        job_parameters = (
            list(config_setup.overrides)
            + cmdline_overrides_siblings
            + [f"++index={point_idx}"]
            + sum(
                [
                    param_to_cmdlines(key, value, prefix="++")
                    for key, value in point.parameters.items()
                ],
                start=[],
            )
        )

        resolved = load_config_reference(
            config_dir=config_setup.config_dir,
            config_path=config_setup.config_path,
            config_name=config_setup.config_name,
            overrides=job_parameters,
            config_class=config_class,
        )

        resolved_dict = asdict(resolved)
        context = {k: v for k, v in resolved_dict.items() if k not in ("sweep")}
        skip_point = False
        for expr in filter_exprs:
            if not _resolve_filter_from_context(expr, context):
                LOGGER.info("Skipping point %s due to sweep.filter", point_idx)
                skip_point = True
                break
        filtered_jobs[point_idx] = skip_point

        stage_name = getattr(resolved, "stage", None)

        job = JobPlan(
            config=resolved,
            parameters=job_parameters,
            sibling_pattern=None,
            stage_name=stage_name,
        )

        resolved_jobs[point_idx] = job

    return list(
        resolved_jobs[point_idx] for point_idx in resolved_jobs if not filtered_jobs[point_idx]
    )


__all__ = [
    "extract_sibling_patterns",
    "find_sibling_by_group_path",
    "build_dependency_dag_from_points",
    "resolve_sweep_with_dag",
]
