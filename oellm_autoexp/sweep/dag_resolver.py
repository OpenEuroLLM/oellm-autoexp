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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
from compoconf import asdict, parse_config
from omegaconf import DictConfig, ListConfig

from oellm_autoexp.config.schema import RootConfig, ConfigSetup
from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.sweep.expander import SweepPoint
from oellm_autoexp.sweep.planner import JobPlan

LOGGER = logging.getLogger(__file__)


def _unescape_config_placeholders(obj: Any) -> Any:
    """Recursively restore {{...}} and unescape \\$ markers from YAML.

    This is FULLY GENERIC - works for ANY escaped interpolation.
    """

    # First unescape OmegaConf $ markers (\\$ → $) from sweep defaults
    # NOTE: Do NOT unescape \( and \) - per OmegaConf grammar, these must stay escaped
    # for proper parsing of expressions in ${oc.eval:...} (they're OmegaConf escape sequences)
    def unescape_dollar_only(o):
        if isinstance(o, str):
            return o.replace("\\$", "$")
        elif isinstance(o, dict):
            return {k: unescape_dollar_only(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [unescape_dollar_only(v) for v in o]
        return o

    obj = unescape_dollar_only(obj)

    # Then unescape our custom placeholders (generic, no pattern matching!)
    if isinstance(obj, str):
        # Restore {{env_flags}}
        result = obj.replace("__PLACEHOLDER_ENV_FLAGS__", "{{env_flags}}").replace(
            "__PLACEHOLDER_ENV_EXPORTS__", "{{env_exports}}"
        )
        # Generic: Restore ALL escaped dollars
        # Works for ${sibling.*}, ${aux.*}, ${slurm.*}, or ANY other escaped interpolation
        # result = result.replace("__ESCAPED_DOLLAR__", "$")
        return result
    elif isinstance(obj, dict):
        return {k: _unescape_config_placeholders(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_unescape_config_placeholders(v) for v in obj]
    return obj


def extract_sibling_patterns(parameters: dict[str, Any]) -> set[str]:
    """Extract stage patterns from escaped sibling references.

    Args:
        parameters: Flattened parameter dict from a SweepPoint

    Returns:
        Set of stage patterns (e.g., {'stable', 'decay6B'})
    """
    patterns = set()

    sibling_regex = re.compile(r"\$\{sibling\.([^.}]+)\.")

    def scan_value(value: Any) -> None:
        """Recursively scan value for sibling patterns."""
        if isinstance(value, str):
            for match in sibling_regex.finditer(value):
                # Extract the stage name (first captured group)
                stage = match.group(1)
                patterns.add(stage)
        elif isinstance(value, dict):
            for v in value.values():
                scan_value(v)
        elif isinstance(value, list):
            for item in value:
                scan_value(item)

    scan_value(parameters)
    LOGGER.debug(f"Extracted sibling patterns: {patterns}")
    return patterns


def find_sibling_by_group_path(
    point: SweepPoint, all_points: list[SweepPoint], stage_pattern: str
) -> SweepPoint | None:
    """Find sibling with matching hyperparameters."""

    own_stage: str | None = point.parameters.get("stage")
    if not extract_sibling_patterns(point.parameters):
        return []

    siblings = []
    point_filtered = list(zip(point.group_path, point.stage_path))

    for potential_sibling in all_points.values():
        sibling_filtered = list(zip(potential_sibling.group_path, potential_sibling.stage_path))
        if (
            all(
                (gp == gs) or sp or ss
                for ((gp, sp), (gs, ss)) in zip(point_filtered, sibling_filtered)
            )
            and point.group_path != potential_sibling.group_path
        ):
            # matching all non-sibling stages
            siblings.append(potential_sibling)
    LOGGER.debug(
        f"Got siblings for own stage {own_stage}: {[s.parameters.get('stage', '') for s in siblings]}"
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


def build_dependency_dag_from_points(points: dict[int, SweepPoint]) -> nx.DiGraph:
    """Build dependency DAG from sweep points."""
    LOGGER.debug(f"Building dependency DAG from {len(points)} points")
    dag = nx.DiGraph()

    for point in points.values():
        dag.add_node(point.index)

    edges_added = 0
    for point in points.values():
        sibling_deps = extract_sibling_patterns(point.parameters)

        for stage_pattern in sibling_deps:
            try:
                sibling = find_sibling_by_group_path(point, points, stage_pattern)
                if sibling:
                    dag.add_edge(sibling.index, point.index)
                else:
                    LOGGER.warning(
                        f"No sibling found for requested stage_pattern: {stage_pattern} of point {point}"
                    )
                edges_added += 1
            except ValueError:
                pass  # Root node, no dependencies

    LOGGER.info(f"Built DAG with {len(points)} nodes and {edges_added} edges")
    return dag


def build_stage_dependencies(points: dict[int, SweepPoint]) -> dict[str, set[str]]:
    """Build stage-level dependency map from sweep points."""
    dag = build_dependency_dag_from_points(points)
    stage_by_index = {
        idx: (point.parameters.get("stage") or "unknown") for idx, point in points.items()
    }
    dependencies: dict[str, set[str]] = {}
    for idx, stage in stage_by_index.items():
        dependencies.setdefault(stage, set())
        for parent in dag.predecessors(idx):
            parent_stage = stage_by_index.get(parent, "unknown")
            if parent_stage != stage:
                dependencies[stage].add(parent_stage)
    return dependencies


def _unflatten_dict(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert flattened dict with dotted keys to nested dict."""
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                break
            current = current[part]
        else:
            if isinstance(current, dict):
                current[parts[-1]] = value
    return nested


def escape_braces_except_dollar(s: str) -> str:
    """
    >>> escape_braces_except_dollar('{}')
    '\\\\{\\\\}'
    >>> escape_braces_except_dollar('123{}456')
    '123\\\\{\\\\}456'
    >>> escape_braces_except_dollar('{a}')
    '\\\\{a\\\\}'
    >>> escape_braces_except_dollar('${a}')
    '${a}'
    >>> escape_braces_except_dollar('${{}}')
    '${\\\\{\\\\}}'
    >>> escape_braces_except_dollar('{${}}')
    '\\\\{${}\\\\}'
    """
    stack = []
    res = ""
    pos = 0

    for mat in re.finditer(r"(\$\{|\{|\})", s):
        if mat.group() == "${":
            res += s[pos : mat.end()]
            stack.append(False)
        elif mat.group() == "{":
            stack.append(True)
            res += s[pos : mat.start()] + "\\{"
        elif mat.group() == "}":
            if stack[-1]:
                res += s[pos : mat.start()] + "\\}"
            else:
                res += s[pos : mat.end()]
            stack.pop(-1)
        pos = mat.end()
    res += s[pos:]
    return res


def config_to_cmdline(
    cfg_dict: dict, override: str = "", prefix="", unescape_interpolations: bool = False
) -> list[str]:
    # override either "", "+", "++" see hydra
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
                # unescape interpolations
                if unescape_interpolations:
                    dct = dct.replace("\\${", "${")
                # escape {}
                dct = re.sub(r"\$(?=[^\{])", r"\\$", dct)
                dct = re.sub(r"\(", r"\\(", dct)
                dct = re.sub(r"\)", r"\\)", dct)
                dct = escape_braces_except_dollar(dct)

                if " " in dct or "," in dct:
                    dct = dct.replace('"', '\\"')
                    dct = f'"{dct}"'
            cmdlines.append(override + prefix + "=" + str(dct))
        return cmdlines

    cmdline_opts = dict_to_cmdlines(cfg_dict, prefix=prefix)
    LOGGER.debug("GENERATED CMDLINE OPTS {config_yaml}, {cmdline_opts}")
    return cmdline_opts


def param_to_cmdlines(
    key: str, val: Any, prefix: str = "", unescape_interpolations: bool = True
) -> list[str]:
    if isinstance(val, str):
        if unescape_interpolations:
            val = val.replace("\\${", "${")
        if " " in val or "," in val:
            val = val.replace('"', '\\"')
            return [f'{prefix}{key}="{val}"']
        else:
            return [f"{prefix}{key}={val}"]
    else:
        return config_to_cmdline(
            val, override="++", prefix=key, unescape_interpolations=unescape_interpolations
        )


def resolve_sweep_with_dag(
    config: RootConfig, points: list[SweepPoint], config_setup: ConfigSetup
) -> list[JobPlan]:
    """Pure OmegaConf resolution with DAG ordering.

    This is the main entry point. It:
    1. Builds dependency DAG
    2. Topologically sorts jobs
    3. For each job, creates OmegaConf config with sibling data
    4. Lets OmegaConf resolve ALL interpolations (including ${sibling.*})
    5. Creates JobPlan from resolved config

    Args:
        config: Root configuration
        points: SweepPoints from expand_sweep()

    Returns:
        List of fully resolved JobPlans
    """
    LOGGER.info(f"Starting DAG resolution for {len(points)} sweep points")

    # Build DAG and get resolution order
    dag = build_dependency_dag_from_points(points)

    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        LOGGER.error(f"Circular dependencies detected: {cycles}")
        raise ValueError(f"Circular dependencies detected: {cycles}")

    ordered_indices = list(nx.topological_sort(dag))
    LOGGER.debug(f"Topological order: {ordered_indices}")

    # Resolve in topological order
    resolved_jobs = {}

    for point_idx in ordered_indices:
        point = points[point_idx]

        # Find sibling dependencies
        sibling_patterns = extract_sibling_patterns(point.parameters)

        # Get already-resolved sibling jobs
        sibling_jobs = {}
        for pattern in sibling_patterns:
            try:
                sibling_point = find_sibling_by_group_path(point, points, pattern)
                if sibling_point and sibling_point.index in resolved_jobs:
                    sibling_jobs[pattern] = resolved_jobs[sibling_point.index]
            except ValueError:
                pass  # No sibling found

        # Load sibling configs with their respective overrides
        sibling_job_configs = {
            sibling_pattern: asdict(
                load_config_reference(
                    config_dir=config_setup.config_dir,
                    ref=config_setup.config_ref,
                    overrides=list(config_setup.override) + sibling_job.parameters,
                )
            )
            for sibling_pattern, sibling_job in sibling_jobs.items()
        }

        for sibling_pattern in sibling_job_configs:
            # remove sweep for nested setup
            del sibling_job_configs[sibling_pattern]["sweep"]

        # Add sibling metadata for ${sibling.*} interpolations

        cmdline_overrides_siblings = config_to_cmdline(
            {
                "sibling": {
                    sibling_job["stage"]: sibling_job
                    for sibling_job in sibling_job_configs.values()
                }
            },
            override="++",
        )

        job_parameters = (
            list(config_setup.override)
            + cmdline_overrides_siblings
            + [f"++index={point_idx}"]
            + sum(
                [
                    param_to_cmdlines(key, value, prefix="")
                    for key, value in point.parameters.items()
                ],
                start=[],
            )
        )
        resolved = load_config_reference(
            config_dir=config_setup.config_dir,
            ref=config_setup.config_ref,
            overrides=job_parameters,
        )
        resolved = asdict(
            parse_config(
                RootConfig,
                {
                    **asdict(resolved),
                    "project_name": config.project.name,
                },
            )
        )

        # these are fixed from the config and not dependent on downstream data, exe
        job_name = resolved["project"]["name"]
        output_dir = str(Path(resolved["project"]["base_output_dir"]))
        # this should not contain slurm template arguments
        log_path_current = str(Path(resolved["project"]["log_path_current"]))

        # this can include template arguments for slurm
        log_path = str(Path(resolved["project"]["log_path"]))

        job_config = resolved.get("job", {})

        # Extract start/cancel conditions from resolved config
        start_conditions = job_config.get("start_conditions", [])
        cancel_conditions = job_config.get("cancel_conditions", [])

        start_condition_cmd = job_config.get("start_condition_cmd")
        start_condition_interval = job_config.get("start_condition_interval_seconds")
        termination_string = resolved.get("monitoring", {}).get("termination_string")
        termination_command = resolved.get("monitoring", {}).get("termination_command")
        inactivity_threshold = job_config.get("inactivity_threshold_seconds")
        if inactivity_threshold is None:
            inactivity_threshold = resolved.get("monitoring", {}).get(
                "inactivity_threshold_seconds"
            )

        # Create JobPlan
        job = JobPlan(
            name=job_name,
            parameters=job_parameters,
            output_dir=output_dir,
            log_path=log_path,
            log_path_current=log_path_current,
            output_paths=[],
            start_condition_cmd=start_condition_cmd,
            start_condition_interval_seconds=start_condition_interval,
            start_conditions=start_conditions,
            cancel_conditions=cancel_conditions,
            termination_string=termination_string,
            termination_command=termination_command,
            inactivity_threshold_seconds=inactivity_threshold,
            sibling_pattern=None,
            stage_name=resolved.get("stage"),
        )

        resolved_jobs[point_idx] = job

    return list(resolved_jobs.values())


def _flatten_dict(
    nested_dict: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Convert nested dict to flattened dict with dotted keys."""
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


__all__ = [
    "extract_sibling_patterns",
    "find_sibling_by_group_path",
    "build_dependency_dag_from_points",
    "resolve_sweep_with_dag",
]


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
