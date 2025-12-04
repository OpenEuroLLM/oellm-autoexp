"""Helpers for reading user configuration into typed dataclasses."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from compoconf import parse_config
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from . import schema
from .normalize import ensure_monitoring_state_dir, normalize_config_data
from .resolvers import register_default_resolvers


class ConfigLoaderError(RuntimeError):
    """Raised when the configuration file cannot be parsed."""


_REGISTRY_SENTINEL = {"loaded": False}
_DEPRECATED_MONITORING_KEYS = ("log_signals", "policies")


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
    data = normalize_config_data(data)
    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover - compoconf raises rich errors
        raise ConfigLoaderError(f"Unable to parse config {path}: {exc}") from exc

    ensure_monitoring_state_dir(root)

    return root


def load_hydra_config(
    config_name: str,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> schema.RootConfig:
    register_default_resolvers()

    overrides = list(overrides or [])
    overrides = [
        override.split("=")[0] + '="' + "=".join(override.split("=")[1:]) + '"'
        if "${" in override
        else override
        for override in overrides
    ]

    config_dir = Path(config_dir).resolve()
    if not config_dir.exists():
        raise ConfigLoaderError(f"Hydra config directory not found: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    # Temporarily escape {{...}} placeholders so Hydra doesn't try to parse them
    # These are meant for Python's str.format(), not Hydra interpolation
    def escape_double_braces(obj):
        """Recursively escape {{...}} in strings to prevent Hydra from parsing
        them."""
        if isinstance(obj, str):
            # Replace {{env_flags}} with a placeholder that Hydra won't touch
            return obj.replace("{{env_flags}}", "__PLACEHOLDER_ENV_FLAGS__").replace(
                "{{env_exports}}", "__PLACEHOLDER_ENV_EXPORTS__"
            )
        elif isinstance(obj, dict):
            return {k: escape_double_braces(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [escape_double_braces(v) for v in obj]
        return obj

    def unescape_double_braces(obj):
        """Recursively restore {{...}} placeholders."""
        if isinstance(obj, str):
            return obj.replace("__PLACEHOLDER_ENV_FLAGS__", "{{env_flags}}").replace(
                "__PLACEHOLDER_ENV_EXPORTS__", "{{env_exports}}"
            )
        elif isinstance(obj, dict):
            return {k: unescape_double_braces(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [unescape_double_braces(v) for v in obj]
        return obj

    # Get unresolved config, escape {{...}}, then resolve
    data_unresolved = OmegaConf.to_container(cfg, resolve=False)  # type: ignore[return-value]
    data_escaped = escape_double_braces(data_unresolved)
    cfg_escaped = OmegaConf.create(data_escaped)
    data = OmegaConf.to_container(cfg_escaped, resolve=True)  # type: ignore[return-value]
    data = unescape_double_braces(data)
    _ensure_registrations()
    if not isinstance(data, Mapping):
        raise ConfigLoaderError(f"Hydra config {config_name} did not produce a mapping")
    _ensure_no_deprecated_monitoring_keys(data, source=f"Hydra config {config_name}")
    data = normalize_config_data(data)
    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover
        raise ConfigLoaderError(f"Unable to parse Hydra config {config_name}: {exc}") from exc

    ensure_monitoring_state_dir(root)

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
            cfg = OmegaConf.load(str(path))

            # Apply overrides (Hydra-style: key=value)
            for override_str in overrides:
                if "=" not in override_str:
                    continue
                key, value = override_str.split("=", 1)
                # Handle OmegaConf dot notation
                OmegaConf.update(cfg, key, value, merge=True)

            # Convert to container and parse
            data = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(data, Mapping):
                raise ConfigLoaderError(f"Config file {path} did not produce a mapping")
            _ensure_no_deprecated_monitoring_keys(data, source=str(path))
            data = normalize_config_data(data)
            try:
                root = parse_config(schema.RootConfig, data)
            except Exception as exc:
                raise ConfigLoaderError(f"Unable to parse config {path}: {exc}") from exc

            ensure_monitoring_state_dir(root)
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
