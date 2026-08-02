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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

import networkx as nx
from compoconf import asdict
from omegaconf import DictConfig, ListConfig, OmegaConf

from .config.schema import StagedSweepRoot, SweepConfig, ConfigSetup
from .config.loader import load_config_reference
from .expander import SweepPoint
from .parallel import run_chunks, worker_count
from .planner import JobPlan

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


def _resolve_filter_from_context(filter_expr: Any, context: Mapping[str, Any] | Callable) -> bool:
    """Evaluate one filter expression.

    ``context`` may be a zero-argument callable, which is only invoked for
    filters that actually need it -- building the context means walking the
    whole resolved config, and most filters are already a plain bool.
    """
    if filter_expr is None:
        return True
    if isinstance(filter_expr, bool):
        return filter_expr
    if not isinstance(filter_expr, str):
        raise ValueError("sweep.filter must resolve to a bool.")
    if callable(context):
        context = context()
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


def drop_cmdline_invisible(value: Any) -> Any:
    """Strip what ``config_to_cmdline`` cannot express, so a direct merge matches it.

    An empty mapping flattens to zero overrides, so round-tripping a config
    through the command line silently drops it -- and a list element that is an
    empty mapping is left as the placeholder index ``config_to_cmdline`` emits
    for it. Applying the same pruning before merging keeps the two paths in
    exact agreement.
    """
    if isinstance(value, Mapping):
        pruned = {key: drop_cmdline_invisible(item) for key, item in value.items()}
        return {
            key: item
            for key, item in pruned.items()
            if not (isinstance(item, Mapping) and not item)
        }
    if isinstance(value, (list, ListConfig, Sequence)) and not isinstance(value, (str, bytes)):
        out = []
        for index, item in enumerate(value):
            pruned = drop_cmdline_invisible(item)
            out.append(index if isinstance(pruned, Mapping) and not pruned else pruned)
        return out
    return value


def is_config_group(key: str, config_dir: str | Path | None) -> bool:
    """Check if a parameter key refers to a Hydra config group.

    A parameter is considered a config group if:
    1. It contains '/' (e.g., "basic/subconfig") - explicit config group path
    2. A directory with that name exists in config_dir

    Args:
        key: Parameter name
        config_dir: Configuration directory path

    Returns:
        True if key refers to a config group, False otherwise
    """
    # If key contains '/', it's a config group path
    if "/" in key:
        return True

    # Check if directory exists
    if config_dir:
        config_group_path = Path(config_dir) / key
        return config_group_path.is_dir()

    return False


def param_to_cmdlines(key: str, val: Any, prefix: str = "", config_dir: str | Path | None = None):
    """Convert a parameter to command-line overrides.

    Args:
        key: Parameter name
        val: Parameter value
        prefix: Override prefix (e.g., "++" for force-add, "" for regular override)
        config_dir: Configuration directory for config group detection

    Returns:
        List of command-line override strings
    """
    if isinstance(val, str):
        if re.match(r"\[[A-Za-z][A-Za-z0-9,]*\]", val):
            return [f"{prefix}{key}={val}"]
        val = val.replace('"', '\\"')
        return [f'{prefix}{key}="{val}"']
    elif isinstance(val, list) and all(isinstance(item, str) for item in val):
        if any("$" in item for item in val):
            # Items contain OmegaConf interpolations; Hydra's [a,b] literal grammar
            # can't represent them. Fall through to the placeholder + dotted-path
            # pattern (key=[0,1,...], key.0=val0, ...) implemented in dict_to_cmdlines.
            return config_to_cmdline(val, override=prefix or "++", prefix=key)
        # Format as Hydra config group list: subconfig=[a,b]
        list_str = "[" + ",".join(val) + "]"
        return [f"{prefix}{key}={list_str}"]
    else:
        return config_to_cmdline(
            val,
            override="++",
            prefix=key,
        )


class _LazyFilterContext:
    """Build the filter context only when a filter actually reads it.

    Flattening a resolved config is not cheap, and a filter that is already a
    bool never looks at the context.
    """

    def __init__(self, resolved: Any) -> None:
        self._resolved = resolved
        self._context: dict[str, Any] | None = None

    def __call__(self) -> dict[str, Any]:
        if self._context is None:
            self._context = {
                key: value
                for key, value in asdict(self._resolved).items()
                if key not in ("sweep")
            }
        return self._context


# Chains of dependent points (a stable stage plus the cooldowns that branch off
# it) must be resolved in order, but separate chains share nothing. They are
# handed to a process pool, which is worth it because resolving a point is pure
# CPU: composing, resolving interpolations and building dataclasses.
_CHAIN_CONTEXT: tuple | None = None


def _resolve_chain(chain: list[int]) -> tuple[dict[int, JobPlan], dict[int, bool]]:
    """Resolve one dependency chain, in the order given."""
    assert _CHAIN_CONTEXT is not None, "chain context not initialised"
    (config, points_dict, config_setup, config_class, sibling_index, sweep_filter_expr) = _CHAIN_CONTEXT

    resolved_jobs: dict[int, JobPlan] = {}
    filtered_jobs: dict[int, bool] = {}
    sibling_context_cache: dict[tuple, tuple] = {}

    for point_idx in chain:
        point = points_dict[point_idx]
        try:
            group_filters = _collect_group_filters(config.sweep.groups, point.group_path)
        except ValueError as exc:
            raise ValueError(f"Unable to match group_path for filtering: {exc}") from exc

        filter_exprs = [sweep_filter_expr, *group_filters]

        sibling_patterns = sibling_index.patterns_by_idx.get(point_idx, set())
        sibling_jobs = {}
        sibling_ids = []
        for pattern in sibling_patterns:
            sibling_point = find_sibling_by_group_path(
                point, points_dict, pattern, sibling_index=sibling_index
            )
            if sibling_point and sibling_point.index in resolved_jobs:
                sibling_jobs[pattern] = resolved_jobs[sibling_point.index]
                sibling_ids.append((pattern, sibling_point.index))

        # Every point in a chain sees the same siblings, and flattening a
        # resolved sibling config is the most expensive step in this loop, so
        # build the context once per distinct set of siblings.
        context_key = tuple(sorted(sibling_ids))
        cached_context = sibling_context_cache.get(context_key)
        if cached_context is None:
            # The sibling was already resolved with exactly these overrides --
            # its JobPlan holds the result. Recomposing it here doubled the
            # amount of Hydra composition every staged sweep had to do.
            sibling_job_configs = {
                sibling_pattern: asdict(sibling_job.config)
                for sibling_pattern, sibling_job in sibling_jobs.items()
            }

            for sibling_pattern in sibling_job_configs:
                if "sweep" in sibling_job_configs[sibling_pattern]:
                    del sibling_job_configs[sibling_pattern]["sweep"]

            sibling_context = {
                "sibling": {
                    sibling_job.get("stage", "unknown"): sibling_job
                    for sibling_job in sibling_job_configs.values()
                }
            }
            # Flatten the context as-is so job_parameters is byte-identical to
            # what a command-line run would carry; merge the pruned form, which
            # is what those overrides actually produce once applied.
            cached_context = (
                config_to_cmdline(sibling_context, override="++"),
                OmegaConf.create(drop_cmdline_invisible(sibling_context))
                if sibling_job_configs
                else None,
            )
            sibling_context_cache[context_key] = cached_context
        cmdline_overrides_siblings, sibling_container = cached_context

        # Generate parameter overrides with smart prefix selection
        param_overrides = []
        for key, value in point.parameters.items():
            # Detect if this is a config group parameter
            if is_config_group(key, config_setup.config_dir):
                # Config group: use no prefix (regular override)
                param_overrides.extend(param_to_cmdlines(key, value, prefix="", config_dir=config_setup.config_dir))
            else:
                # Regular parameter: use ++ prefix (force-add)
                param_overrides.extend(param_to_cmdlines(key, value, prefix="++", config_dir=config_setup.config_dir))

        compose_overrides = list(config_setup.overrides) + [f"++index={point_idx}"] + param_overrides
        job_parameters = (
            list(config_setup.overrides)
            + cmdline_overrides_siblings
            + [f"++index={point_idx}"]
            + param_overrides
        )

        # job_parameters still records the sibling context as ++overrides so the
        # job stays reproducible from the command line, but composing with it is
        # far cheaper as a single merge than as several hundred parsed overrides.
        resolved = load_config_reference(
            config_dir=config_setup.config_dir,
            config_path=config_setup.config_path,
            config_name=config_setup.config_name,
            overrides=compose_overrides,
            extra_config=sibling_container,
            config_class=config_class,
        )

        filter_context = _LazyFilterContext(resolved)
        skip_point = False
        for expr in filter_exprs:
            if not _resolve_filter_from_context(expr, filter_context):
                LOGGER.info("Skipping point %s due to sweep.filter", point_idx)
                skip_point = True
                break
        filtered_jobs[point_idx] = skip_point

        resolved_jobs[point_idx] = JobPlan(
            config=resolved,
            parameters=job_parameters,
            sibling_pattern=None,
            stage_name=getattr(resolved, "stage", None),
        )

    return resolved_jobs, filtered_jobs


def _split_into_chains(dag: nx.DiGraph, ordered_indices: list[int]) -> list[list[int]]:
    """Group points into independent chains, each in topological order."""
    position = {index: order for order, index in enumerate(ordered_indices)}
    chains = [
        sorted(component, key=position.__getitem__)
        for component in nx.weakly_connected_components(dag)
    ]
    # Longest first, so the pool does not finish early workers and then wait on
    # one long chain started last.
    chains.sort(key=len, reverse=True)
    return chains


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
    sweep_filter_expr = config.sweep.filter if isinstance(config.sweep, SweepConfig) else True

    if not isinstance(config.sweep, SweepConfig):
        point = points_dict[list(points_dict)[0]]
        resolved = load_config_reference(
            config_dir=config_setup.config_dir,
            config_path=config_setup.config_path,
            config_name=config_setup.config_name,
            overrides=point.parameters,
            config_class=config_class,
        )

        stage_name = getattr(resolved, "stage", None)

        job = JobPlan(
            config=resolved,
            parameters=point,
            sibling_pattern=None,
            stage_name=stage_name,
        )

        return [job]

    chains = _split_into_chains(dag, ordered_indices)
    workers = worker_count(len(chains), len(ordered_indices))
    LOGGER.info("Resolving %d chains with %d worker(s)", len(chains), workers)

    global _CHAIN_CONTEXT
    _CHAIN_CONTEXT = (
        config,
        points_dict,
        config_setup,
        config_class,
        sibling_index,
        sweep_filter_expr,
    )
    try:
        chain_results = run_chunks(_resolve_chain, chains, workers)
    finally:
        _CHAIN_CONTEXT = None

    for chain_resolved, chain_filtered in chain_results:
        resolved_jobs.update(chain_resolved)
        filtered_jobs.update(chain_filtered)

    # Emit in topological order, matching the order a single-process run built.
    return [
        resolved_jobs[point_idx]
        for point_idx in ordered_indices
        if point_idx in resolved_jobs and not filtered_jobs[point_idx]
    ]


__all__ = [
    "extract_sibling_patterns",
    "find_sibling_by_group_path",
    "build_dependency_dag_from_points",
    "resolve_sweep_with_dag",
]
