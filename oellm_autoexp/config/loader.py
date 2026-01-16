"""Helpers for reading user configuration into typed dataclasses."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from compoconf import parse_config
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from . import schema
from .resolvers import register_default_resolvers

LOGGER = logging.getLogger(__file__)


class ConfigLoaderError(RuntimeError):
    """Raised when the configuration file cannot be parsed."""


_REGISTRY_SENTINEL = {"loaded": False}
_DEPRECATED_MONITORING_KEYS = ("log_signals", "policies")


def _escape_placeholders(obj):
    """Recursively escape {{...}} and \\$ that should not resolve during config
    loading.

    This is FULLY GENERIC - works for ANY escaped interpolation, not just specific patterns.
    """
    if isinstance(obj, str):
        # Replace {{env_flags}} with a placeholder (for Python's str.format())
        result = obj.replace("{{env_flags}}", "__PLACEHOLDER_ENV_FLAGS__").replace(
            "{{env_exports}}", "__PLACEHOLDER_ENV_EXPORTS__"
        )
        # Generic: Replace ALL \\$ with placeholder
        # This works for ANY escaped interpolation: \\${sibling.*}, \\${aux.*}, \\${slurm.*}, etc.
        # No need to know what's being escaped!
        # result = result.replace("\\$", "__ESCAPED_DOLLAR__")
        return result
    elif isinstance(obj, dict):
        return {k: _escape_placeholders(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_escape_placeholders(v) for v in obj]
    return obj


def _ensure_monitoring_state_dir(root: schema.RootConfig) -> None:
    """Assign a deterministic monitoring_state_dir when unspecified."""
    if root.project.monitoring_state_dir is not None:
        return

    base = Path(root.project.base_output_dir)
    stable_base = (
        base.parent
        if base.name.split("_")[-1].isdigit() and len(base.name.split("_")[-1]) >= 8
        else base
    )
    root.project.monitoring_state_dir = stable_base / "monitoring_state"
    LOGGER.debug(f"Auto-configured monitoring_state_dir: {root.project.monitoring_state_dir}")


def _ensure_no_deprecated_monitoring_keys(data: Mapping[str, Any], source: str) -> None:
    """Raise a friendly error when deprecated monitoring keys are present."""

    def _check(section: Mapping[str, Any], prefix: str = "monitoring") -> None:
        for key in _DEPRECATED_MONITORING_KEYS:
            if key in section:
                raise ConfigLoaderError(
                    f"{source}: `{prefix}.{key}` has been removed. "
                    "Attach actions directly to `log_events` / `state_events` instead."
                )

    monitoring_section = data.get("monitoring")
    if isinstance(monitoring_section, Mapping):
        _check(monitoring_section, prefix="monitoring")
    elif any(key in data for key in _DEPRECATED_MONITORING_KEYS):
        _check(data, prefix="monitoring")


def _ensure_registrations() -> None:
    if _REGISTRY_SENTINEL["loaded"]:
        return

    for module in (
        "oellm_autoexp.backends.base",
        "oellm_autoexp.backends.megatron_backend",
        "oellm_autoexp.monitor.actions",
        "oellm_autoexp.monitor.states",
        "oellm_autoexp.monitor.watcher",
        "oellm_autoexp.slurm.client",
    ):
        import_module(module)

    _REGISTRY_SENTINEL["loaded"] = True


def _load_inline_keys(config_path: Path) -> set[str] | None:
    if not config_path.exists():
        return None
    try:
        cfg = OmegaConf.load(str(config_path))
        data = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        return None
    if isinstance(data, Mapping):
        return set(data.keys())
    return None


def _resolve_base_config_path(config_dir: Path, config_name: str) -> Path:
    name_path = Path(config_name)
    if name_path.suffix in {".yaml", ".yml"}:
        return config_dir / name_path
    return config_dir / f"{config_name}.yaml"


def _apply_group_overrides(
    data: Mapping[str, Any],
    config_dir: Path,
    overrides: Iterable[str],
    base_config_path: Path | None = None,
) -> Mapping[str, Any]:
    """Apply config group overrides after Hydra composition.

    This ensures group selections like `backend=variant1` override inline config
    blocks from `_self_` when both are present.
    """
    merged = dict(data)
    inline_keys = _load_inline_keys(base_config_path) if base_config_path else None
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        group = key.lstrip("+")
        if not group or "." in group:
            continue
        if inline_keys is not None and group not in inline_keys:
            continue
        value_str = value.strip().strip("\"'")
        if value_str.startswith("[") and value_str.endswith("]"):
            parts = [p.strip() for p in value_str[1:-1].split(",") if p.strip()]
            if not parts:
                continue
            value_str = parts[-1]
        group_path = Path(config_dir) / group / f"{value_str}.yaml"
        if not group_path.exists():
            continue
        group_cfg = OmegaConf.load(str(group_path))
        group_data = OmegaConf.to_container(group_cfg, resolve=True)
        if not isinstance(group_data, Mapping):
            continue
        existing = merged.get(group, {})
        if isinstance(existing, Mapping):
            merged[group] = {**existing, **group_data}
        else:
            merged[group] = dict(group_data)
    return merged


def _load_yaml(path: str) -> Mapping[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def load_config(path: str | Path) -> schema.RootConfig:
    """Load and validate a configuration file into ``RootConfig``."""

    path = Path(path)
    if not path.exists():
        raise ConfigLoaderError(f"Configuration file not found: {path}")

    register_default_resolvers()
    _ensure_registrations()

    data = _load_yaml(path)
    if not isinstance(data, Mapping):
        raise ConfigLoaderError(f"Configuration root must be a mapping: {path}")
    _ensure_no_deprecated_monitoring_keys(data, source=str(path))
    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover - compoconf raises rich errors
        raise ConfigLoaderError(f"Unable to parse config {path}: {exc}") from exc

    _ensure_monitoring_state_dir(root)
    root.metadata.setdefault("config_ref", str(path))
    root.metadata.setdefault("config_dir", str(path.parent))

    return root


def load_hydra_config(
    config_name: str,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> schema.RootConfig:
    LOGGER.info(f"Loading Hydra config: {config_name} from {config_dir}")
    register_default_resolvers()

    overrides = list(overrides or [])
    if overrides:
        LOGGER.debug(f"Applying {len(overrides)} overrides")

    config_dir = Path(config_dir).resolve()
    if not config_dir.exists():
        raise ConfigLoaderError(f"Hydra config directory not found: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    # Escape placeholders that should not resolve during Hydra composition:
    # - {{...}} are for Python's str.format(), not OmegaConf
    # - \\${...} are escaped interpolations that resolve later (during DAG resolution)
    #   This includes ANY escaped interpolation: \\${sibling.*}, \\${aux.*}, \\${slurm.*}, etc.
    #   Generic approach: no need to know specific paths being escaped!

    # Get UNRESOLVED config, escape \\$, then resolve
    # This ensures escaped interpolations (\\${...}) stay as literal strings during Hydra composition
    # They will be unescaped later during sweep expansion/DAG resolution
    data_unresolved = OmegaConf.to_container(cfg, resolve=False)  # type: ignore[return-value]
    data_escaped = data_unresolved  # _escape_placeholders(data_unresolved)
    cfg_escaped = OmegaConf.create(data_escaped)
    data = OmegaConf.to_container(cfg_escaped, resolve=True)  # type: ignore[return-value]
    _ensure_registrations()
    if not isinstance(data, Mapping):
        raise ConfigLoaderError(f"Hydra config {config_name} did not produce a mapping")
    _ensure_no_deprecated_monitoring_keys(data, source=f"Hydra config {config_name}")

    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover
        raise ConfigLoaderError(f"Unable to parse Hydra config {config_name}: {exc}") from exc

    _ensure_monitoring_state_dir(root)
    root.metadata.setdefault("config_ref", str(config_name))
    root.metadata.setdefault("config_dir", str(config_dir))

    return root


def load_config_reference(
    ref: str | Path,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> schema.RootConfig:
    path = Path(ref)
    if path.exists():
        # If overrides are provided with a direct path, check for config_reference.json
        # to use Hydra compose with original overrides + new overrides
        if overrides:
            # Check for config_reference.json in the same directory
            config_reference_path = path.parent / "config_reference.json"
            if config_reference_path.exists():
                # Load config reference to get original overrides
                import json

                try:
                    reference_data = json.loads(config_reference_path.read_text(encoding="utf-8"))
                    original_config_ref = reference_data.get("config_ref")
                    original_config_dir = reference_data.get("config_dir")
                    original_overrides = reference_data.get("overrides", [])

                    # Combine original overrides + new overrides
                    combined_overrides = list(original_overrides) + list(overrides)

                    print(
                        f"[config] Using Hydra compose with original overrides + new overrides\n"
                        f"  Config: {original_config_ref}\n"
                        f"  Original overrides: {original_overrides}\n"
                        f"  New overrides: {list(overrides)}\n"
                        f"  Combined: {combined_overrides}"
                    )

                    # Use Hydra compose with combined overrides
                    return load_hydra_config(
                        original_config_ref, original_config_dir or config_dir, combined_overrides
                    )
                except Exception as exc:
                    print(
                        f"Warning: Could not load config_reference.json, falling back to OmegaConf: {exc}"
                    )

            # Fallback: use OmegaConf (won't handle Hydra group overrides properly)
            register_default_resolvers()
            _ensure_registrations()

            # Load YAML as OmegaConf
            # cfg = OmegaConf.load(str(path))

            # # Apply overrides (Hydra-style: key=value)
            # for override_str in overrides:
            #     if "=" not in override_str:
            #         continue
            #     key, value = override_str.split("=", 1)
            #     # Handle OmegaConf dot notation
            #     OmegaConf.update(cfg, key, value, merge=True)

            # # Convert to container and parse
            # data = OmegaConf.to_container(cfg, resolve=True)
            # if not isinstance(data, Mapping):
            #     raise ConfigLoaderError(f"Config file {path} did not produce a mapping")
            # _ensure_no_deprecated_monitoring_keys(data, source=str(path))
            with initialize_config_dir(version_base=None, config_dir=os.path.abspath(path.parent)):
                cfg = compose(config_name=path.name[:-5], overrides=overrides)

            # Escape placeholders, resolve, then unescape (same as load_hydra_config)
            data_unresolved = OmegaConf.to_container(cfg, resolve=False)
            data_escaped = data_unresolved  # _escape_placeholders(data_unresolved)
            cfg_escaped = OmegaConf.create(data_escaped)
            data = OmegaConf.to_container(cfg_escaped, resolve=True)
            if not isinstance(data, Mapping):
                raise ConfigLoaderError(f"Config file {path} did not produce a mapping")

            # Unescape placeholders before parsing
            _ensure_no_deprecated_monitoring_keys(data, source=str(path))

            data = _apply_group_overrides(data, path.parent, overrides, base_config_path=path)

            try:
                root = parse_config(schema.RootConfig, data)
            except Exception as exc:
                raise ConfigLoaderError(f"Unable to parse config {path}: {exc}") from exc

            _ensure_monitoring_state_dir(root)
            root.metadata.setdefault("config_ref", str(path))
            root.metadata.setdefault("config_dir", str(path.parent))
            return root
        else:
            # No overrides, use simple loader
            return load_config(path)
    return load_hydra_config(str(ref), config_dir, overrides)


def ensure_registrations() -> None:
    """Expose registry initialisation for consumers that only instantiate
    partial configs."""

    register_default_resolvers()
    _ensure_registrations()


def load_monitoring_reference(
    ref: str | Path,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> schema.MonitorInterface.cfgtype:
    path = Path(ref)
    register_default_resolvers()
    _ensure_registrations()
    if path.exists():
        data = _load_yaml(path)
        if not isinstance(data, Mapping):
            raise ConfigLoaderError(f"Monitor config {path} must be a mapping")
        _ensure_no_deprecated_monitoring_keys(data, source=str(path))
        try:
            return parse_config(schema.MonitorInterface.cfgtype, data)
        except Exception as exc:  # pragma: no cover
            raise ConfigLoaderError(f"Unable to parse monitor config {path}: {exc}") from exc

    root = load_hydra_config(str(ref), config_dir, overrides)
    return root.monitoring


__all__ = [
    "ConfigLoaderError",
    "load_config",
    "load_hydra_config",
    "load_config_reference",
    "load_monitoring_reference",
    "ensure_registrations",
]
