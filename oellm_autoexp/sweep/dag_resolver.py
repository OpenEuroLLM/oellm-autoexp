"""Pure OmegaConf DAG-based sweep resolution.

This is the simplified v2 implementation that uses OmegaConf for ALL interpolations,
including sibling references. No custom template resolution - just pure OmegaConf.

Key insight: Use `${sibling.stable.output_dir}` syntax in YAML and add sibling data
to the OmegaConf namespace during resolution. OmegaConf handles everything.

See docs/sweep_resolution_ordering.md for design rationale (Option 5).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import networkx as nx
from compoconf import asdict
from omegaconf import OmegaConf

from oellm_autoexp.config.schema import RootConfig
from oellm_autoexp.sweep.expander import SweepPoint
from oellm_autoexp.sweep.planner import JobPlan


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
        result = result.replace("__ESCAPED_DOLLAR__", "$")
        return result
    elif isinstance(obj, dict):
        return {k: _unescape_config_placeholders(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_unescape_config_placeholders(v) for v in obj]
    return obj


def _format_with_nested_dict(template: str, data: dict[str, Any]) -> str:
    """Format a template string supporting dotted key lookups in nested dicts.

    Handles templates like "{backend.megatron.lr}" with nested dict
    data.
    """
    import string

    class DottedDictFormatter(string.Formatter):
        def get_field(self, field_name, args, kwargs):
            # Split on dots and traverse the dict
            parts = field_name.split(".")
            obj = kwargs
            for part in parts:
                obj = obj[part]
            return obj, field_name

    formatter = DottedDictFormatter()
    return formatter.format(template, **data)


def extract_sibling_patterns(parameters: dict[str, Any]) -> set[str]:
    """Extract stage patterns from escaped sibling references.

    Args:
        parameters: Flattened parameter dict from a SweepPoint

    Returns:
        Set of stage patterns (e.g., {'stable', 'decay6B'})
    """
    patterns = set()

    # Pattern to match escaped sibling references: __ESCAPED_DOLLAR__{sibling.STAGE.FIELD}
    # These use the generic escaped dollar placeholder
    sibling_regex = re.compile(r"__ESCAPED_DOLLAR__\{sibling\.([^.}]+)\.")

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
    return patterns


def find_sibling_by_group_path(
    point: SweepPoint, all_points: list[SweepPoint], stage_pattern: str
) -> SweepPoint:
    """Find sibling with matching hyperparameters.

    Matches based on core hyperparameter values (lr, batch_size) that
    are common across all stages, ignoring stage-specific parameters
    like decay_fraction, load, etc.
    """
    # Core hyperparameters that should match across siblings (excluding stage-specific params)
    # These are the parameters that define the hyperparameter grid
    core_hp_keys = {"backend.megatron.lr", "backend.megatron.global_batch_size"}

    # Extract core hyperparameters from current point
    current_hps = {k: point.parameters.get(k) for k in core_hp_keys if k in point.parameters}

    candidates = []
    for p in all_points:
        # Must match stage pattern
        if p.parameters.get("stage") != stage_pattern:
            continue

        # Don't match self
        if p.index == point.index:
            continue

        # Check if core hyperparameters match
        p_hps = {k: p.parameters.get(k) for k in core_hp_keys if k in p.parameters}
        if p_hps != current_hps:
            continue

        candidates.append(p)

    if len(candidates) == 0:
        raise ValueError(
            f"No sibling found for point {point.index} (stage={point.parameters.get('stage')}) "
            f"with stage pattern '{stage_pattern}'. Core HPs: {current_hps}"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple siblings found for point {point.index}: {[c.index for c in candidates]}"
        )

    return candidates[0]


def build_dependency_dag_from_points(points: list[SweepPoint]) -> nx.DiGraph:
    """Build dependency DAG from sweep points."""
    dag = nx.DiGraph()

    for point in points:
        dag.add_node(point.index)

    for point in points:
        sibling_deps = extract_sibling_patterns(point.parameters)

        for stage_pattern in sibling_deps:
            try:
                sibling = find_sibling_by_group_path(point, points, stage_pattern)
                dag.add_edge(sibling.index, point.index)
            except ValueError:
                pass  # Root node, no dependencies

    return dag


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


def resolve_sweep_with_dag(config: RootConfig, points: list[SweepPoint]) -> list[JobPlan]:
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
    # Build DAG and get resolution order
    dag = build_dependency_dag_from_points(points)

    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        raise ValueError(f"Circular dependencies detected: {cycles}")

    ordered_indices = list(nx.topological_sort(dag))

    # Resolve in topological order
    resolved_jobs = {}
    base_output = Path(config.project.base_output_dir)

    for point_idx in ordered_indices:
        point = points[point_idx]

        # Find sibling dependencies
        sibling_patterns = extract_sibling_patterns(point.parameters)

        # Get already-resolved sibling jobs
        sibling_jobs = {}
        for pattern in sibling_patterns:
            try:
                sibling_point = find_sibling_by_group_path(point, points, pattern)
                if sibling_point.index in resolved_jobs:
                    sibling_jobs[pattern] = resolved_jobs[sibling_point.index]
            except ValueError:
                pass  # No sibling found

        # Build nested config dict (as plain dicts, not OmegaConf)
        # NOTE: Don't include 'sweep' - it contains group defaults with sibling refs that would conflict
        config_dict = {
            "backend": asdict(config.backend),
            "slurm": asdict(config.slurm),
            "project": asdict(config.project),
            "monitoring": asdict(config.monitoring),
            # Sibling data (so ${sibling.stable.output_dir} resolves)
            "sibling": {
                stage: {
                    "name": job.name,
                    "output_dir": job.output_dir,
                    "log_path": job.log_path,
                    "log_path_current": job.log_path_current or "",
                }
                for stage, job in sibling_jobs.items()
            },
        }

        # Merge point parameters (unflatten first) - use plain dict merge
        point_nested = _unflatten_dict(point.parameters)

        # Deep merge nested dicts
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value

        deep_merge(config_dict, point_nested)

        # Unescape placeholders (${sibling.*} and ${backend.megatron.aux.*})
        # These were escaped during config loading to prevent premature resolution
        config_dict = _unescape_config_placeholders(config_dict)

        # Create OmegaConf config (NOW it has unescaped ${...} patterns)
        cfg = OmegaConf.create(config_dict)

        # Resolve ALL interpolations (pure OmegaConf!)
        # All derived parameters should be defined using ${oc.eval:...} in the YAML config
        resolved = OmegaConf.to_container(cfg, resolve=True)

        # Format job name using nested dict structure (supports {backend.megatron.lr} syntax)
        job_name = _format_with_nested_dict(config.sweep.name_template, resolved)

        # Build paths
        output_dir = str(base_output / job_name)
        # Add output_dir and name to resolved for log path formatting
        resolved_with_paths = {**resolved, "output_dir": output_dir, "name": job_name}
        log_path = _format_with_nested_dict(
            config.monitoring.log_path_template, resolved_with_paths
        )

        # Extract flattened parameters for JobPlan
        backend_flat = _flatten_dict(resolved.get("backend", {}))

        # Extract start/cancel conditions
        start_conditions = resolved.get("job", {}).get("start_conditions", [])
        cancel_conditions = resolved.get("job", {}).get("cancel_conditions", [])

        # Filter parameters (remove job.* and monitoring.* keys)
        filtered_params = {}
        for key, value in backend_flat.items():
            if not key.startswith("job.") and not key.startswith("monitoring."):
                filtered_params[f"backend.{key}"] = str(value) if value is not None else None

        # Create JobPlan
        job = JobPlan(
            name=job_name,
            parameters=filtered_params,
            output_dir=output_dir,
            log_path=log_path,
            log_path_current=None,
            output_paths=[],
            start_conditions=start_conditions,
            cancel_conditions=cancel_conditions,
            termination_string=getattr(config.monitoring, "termination_string", None),
            termination_command=getattr(config.monitoring, "termination_command", None),
            inactivity_threshold_seconds=getattr(
                config.monitoring, "inactivity_threshold_seconds", None
            ),
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
