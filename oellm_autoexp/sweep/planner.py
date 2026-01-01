"""Build execution plans from sweep outputs."""

from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from pathlib import Path

from compoconf import asdict
from omegaconf import OmegaConf
from typing import Any
from oellm_autoexp.config.schema import RootConfig
from .expander import SweepPoint


@dataclass(kw_only=True)
class JobPlan:
    """Normalized job description used by downstream modules."""

    name: str = field(default_factory=MISSING)
    parameters: dict[str, str] = field(default_factory=MISSING)
    output_dir: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    log_path_current: str | None = None  # Symlink path for current.log
    output_paths: list[str] = field(default_factory=list)

    # Start conditions (old blocking approach - DEPRECATED)
    start_condition_cmd: str | None = None
    start_condition_interval_seconds: int | None = None

    # Start conditions (new async approach with monitor-driven submission)
    start_conditions: list[dict[str, Any]] = field(default_factory=list)

    # Termination and monitoring
    termination_string: str | None = None
    termination_command: str | None = None
    inactivity_threshold_seconds: int | None = None

    # Cancel conditions (proactive failure detection)
    cancel_conditions: list[dict[str, Any]] = field(default_factory=list)

    # Multi-stage metadata
    sibling_pattern: str | None = None  # Pattern to identify siblings (without stage)
    stage_name: str | None = None  # Stage identifier (e.g., "stable", "cooldown")


def flatten_config(config: RootConfig | dict[str, Any], connector: str = "."):
    """
    >>> flatten({"key": {"subkey": "val"}})
    {"key.subkey": "val"}
    """
    if not isinstance(config, dict):
        cfg_dict = asdict(config)
    else:
        cfg_dict = config

    def _flat(d: tuple | list | dict | Any, prefix: str = ""):
        res = {}
        if isinstance(d, dict):
            for key, val in d.items():
                res.update(_flat(val, prefix=prefix + connector + key if prefix else key))
        elif isinstance(d, (list, tuple)):
            for idx, val in enumerate(cfg_dict):
                res.update(_flat(val, prefix=prefix + connector + str(idx) if prefix else key))
        else:
            res = {prefix: d}
        return res

    return _flat(cfg_dict, prefix="")


def simple_format(template_str: str, args: dict):
    for key, val in args.items():
        template_str = template_str.replace("{" + key + "}", str(val))
    return template_str


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
                # Key exists as a value but we need it as a container
                # Skip this key to avoid overwriting
                break
            else:
                current = current[part]
        else:
            # Only set if we didn't break
            if isinstance(current, dict):
                current[parts[-1]] = value
    return nested


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


def _resolve_omegaconf_strings(context: dict[str, Any]) -> dict[str, Any]:
    """Resolve OmegaConf interpolations that were escaped during config
    loading.

    This handles interpolations in sweep defaults that reference both
    base config values and stage-specific parameter overrides.
    """
    from oellm_autoexp.sweep.expander import _unescape_omegaconf_markers
    import sys

    # Unescape ${...} markers
    unescaped = _unescape_omegaconf_markers(context)

    # Unflatten to nested structure for OmegaConf resolution
    nested = _unflatten_dict(unescaped)

    print("DEBUG: Attempting OmegaConf resolution...", file=sys.stderr)
    print(
        f"DEBUG: Sample nested keys: {list(nested.get('backend', {}).get('megatron', {}).get('aux', {}).keys())}",
        file=sys.stderr,
    )

    # Check what train_iters looks like
    train_iters = nested.get("backend", {}).get("megatron", {}).get("train_iters")
    print(f"DEBUG: train_iters before resolution = {train_iters}", file=sys.stderr)

    # Resolve with OmegaConf
    try:
        cfg = OmegaConf.create(nested)
        print("DEBUG: OmegaConf.create succeeded", file=sys.stderr)
        resolved_nested = OmegaConf.to_container(cfg, resolve=True)
        print("DEBUG: OmegaConf.to_container succeeded", file=sys.stderr)
        # Flatten back
        return _flatten_dict(resolved_nested) if isinstance(resolved_nested, dict) else unescaped
    except Exception as e:
        # If resolution fails, return unescaped (better than crashing)
        print(f"ERROR: OmegaConf resolution failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return unescaped


def _resolve_conditions(
    conditions: list[dict[str, Any]], context: dict[str, Any]
) -> list[dict[str, Any]]:
    """Resolve OmegaConf interpolations in start/cancel conditions using the
    merged context."""
    from oellm_autoexp.sweep.expander import _unescape_omegaconf_markers
    import sys

    # Unescape and resolve each condition
    resolved_conditions = []
    for cond in conditions:
        # Unescape ${...} markers
        unescaped_cond = _unescape_omegaconf_markers(cond)
        print(f"DEBUG _resolve_conditions: unescaped_cond = {unescaped_cond}", file=sys.stderr)

        # Create an OmegaConf object with the condition and context for resolution
        # The context provides values for ${backend.megatron.aux.start_iter_round} etc.
        nested_context = _unflatten_dict(context)
        try:
            # Merge condition into a temporary config with context
            temp_cfg = OmegaConf.create({"_cond": unescaped_cond, "_ctx": nested_context})
            resolved_temp = OmegaConf.to_container(temp_cfg, resolve=True)
            resolved_conditions.append(resolved_temp["_cond"])
            print(
                f"DEBUG _resolve_conditions: resolved_cond = {resolved_temp['_cond']}",
                file=sys.stderr,
            )
        except Exception as e:
            # If resolution fails, use unescaped
            print(f"ERROR _resolve_conditions: {e}", file=sys.stderr)
            resolved_conditions.append(unescaped_cond)

    return resolved_conditions


def build_job_plans(config: RootConfig, points: list[SweepPoint]) -> list[JobPlan]:
    base_output = Path(config.project.base_output_dir)
    project_name = config.project.name

    plans: list[JobPlan] = []
    for point in points:
        context: dict[str, str] = {
            "project_name": project_name,
            "index": str(point.index),
        }
        context.update(flatten_config(config))
        context.update(point.parameters)
        # Resolve any remaining OmegaConf interpolations (from escaped sweep defaults)
        # This ensures interpolations use the merged config+parameter values
        context = _resolve_omegaconf_strings(context)
        job_name = simple_format(config.sweep.name_template, context)

        output_dir = str(base_output / job_name)
        log_template = config.monitoring.log_path_template
        format_context = {**context, "output_dir": str(output_dir), "name": job_name}
        log_path = log_template.format(**format_context)
        monitoring_config = config.monitoring
        output_templates = getattr(monitoring_config, "output_paths", [])
        resolved_outputs: list[str] = []
        for template in output_templates:
            try:
                resolved_outputs.append(template.format(**format_context))
            except KeyError:
                resolved_outputs.append(template)

        start_condition_cmd = getattr(monitoring_config, "start_condition_cmd", None)
        start_condition_interval = getattr(
            monitoring_config,
            "start_condition_interval_seconds",
            None,
        )
        termination_string = getattr(monitoring_config, "termination_string", None)
        termination_command = getattr(monitoring_config, "termination_command", None)
        inactivity_threshold = getattr(
            monitoring_config,
            "inactivity_threshold_seconds",
            None,
        )

        start_conditions = []
        cancel_conditions = []

        filtered_params: dict[str, str] = {}
        for key, value in point.parameters.items():
            if value is None:
                continue
            normalized = key.lower()
            if normalized in {
                "job.start_condition",
                "job.start_condition_cmd",
                "start_condition",
                "start_condition_cmd",
                "monitoring.start_condition_cmd",
            }:
                start_condition_cmd = str(value)
                continue
            if normalized in {
                "job.start_condition_interval_seconds",
                "start_condition_interval_seconds",
                "monitoring.start_condition_interval_seconds",
            }:
                try:
                    start_condition_interval = int(value)
                except (TypeError, ValueError):
                    start_condition_interval = None
                continue
            if normalized in {
                "job.start_conditions",
                "start_conditions",
                "monitoring.start_conditions",
            }:
                # Extract start_conditions (expected to be a list of dicts)
                # Don't resolve yet - will resolve after context is built
                if OmegaConf.is_list(value) or isinstance(value, list):
                    start_conditions = (
                        OmegaConf.to_container(value, resolve=False)
                        if OmegaConf.is_list(value)
                        else value
                    )
                continue
            if normalized in {
                "job.termination_string",
                "termination_string",
                "monitoring.termination_string",
            }:
                termination_string = str(value)
                continue
            if normalized in {
                "job.termination_command",
                "termination_command",
                "monitoring.termination_command",
            }:
                termination_command = str(value)
                continue
            if normalized in {
                "job.inactivity_threshold_seconds",
                "inactivity_threshold_seconds",
                "monitoring.inactivity_threshold_seconds",
            }:
                try:
                    inactivity_threshold = int(value)
                except (TypeError, ValueError):
                    inactivity_threshold = None
                continue
            if normalized in {
                "job.cancel_conditions",
                "cancel_conditions",
                "monitoring.cancel_conditions",
            }:
                # Extract cancel_conditions (expected to be a list of dicts)
                # Don't resolve yet - will resolve after context is built
                if OmegaConf.is_list(value) or isinstance(value, list):
                    cancel_conditions = (
                        OmegaConf.to_container(value, resolve=False)
                        if OmegaConf.is_list(value)
                        else value
                    )
                continue
            filtered_params[key] = str(value)

        # Resolve start_conditions and cancel_conditions using the merged context
        # This allows them to reference computed values like start_iter_round
        if start_conditions:
            start_conditions = _resolve_conditions(start_conditions, context)
        if cancel_conditions:
            cancel_conditions = _resolve_conditions(cancel_conditions, context)

        plans.append(
            JobPlan(
                name=job_name,
                parameters=filtered_params,
                output_dir=output_dir,
                log_path=log_path,
                output_paths=resolved_outputs,
                start_condition_cmd=start_condition_cmd,
                start_condition_interval_seconds=start_condition_interval,
                start_conditions=start_conditions,
                termination_string=termination_string,
                termination_command=termination_command,
                inactivity_threshold_seconds=inactivity_threshold,
                cancel_conditions=cancel_conditions,
            )
        )
    return plans


__all__ = ["JobPlan", "build_job_plans"]


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
