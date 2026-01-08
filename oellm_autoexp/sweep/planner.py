"""Build execution plans from sweep outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, MISSING

from compoconf import asdict
from omegaconf import OmegaConf
from typing import Any
from oellm_autoexp.config.schema import RootConfig

LOGGER = logging.getLogger(__name__)


@dataclass(kw_only=True)
class JobPlan:
    """Normalized job description used by downstream modules."""

    name: str = field(default_factory=MISSING)
    parameters: list[str] = field(default_factory=MISSING)  # hydra / omegaconf cmdline overrides
    output_dir: str = field(default_factory=MISSING)
    log_path: str = field(default_factory=MISSING)
    log_path_current: str = field(default_factory=MISSING)
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
    """Format template string with dotted key support (e.g.,
    {backend.megatron.lr}).

    Args:
        template_str: Template with placeholders like {key} or {nested.key}
        args: Flat dict with dotted keys (e.g., {'backend.megatron.lr': '1e-4'})

    Returns:
        Formatted string with all placeholders replaced
    """
    # First try direct key replacement (flat keys)
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


__all__ = ["JobPlan"]


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
