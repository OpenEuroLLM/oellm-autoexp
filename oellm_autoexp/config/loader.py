"""Helpers for reading user configuration into typed dataclasses."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Mapping

from compoconf import parse_config
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from . import schema
from .resolvers import register_default_resolvers


class ConfigLoaderError(RuntimeError):
    """Raised when the configuration file cannot be parsed."""


_REGISTRY_SENTINEL = {"loaded": False}


def _ensure_registrations() -> None:
    if _REGISTRY_SENTINEL["loaded"]:
        return

    for module in (
        "oellm_autoexp.backends.base",
        "oellm_autoexp.backends.megatron_backend",
        "oellm_autoexp.monitor.watcher",
        "oellm_autoexp.monitor.policy",
        "oellm_autoexp.slurm.client",
    ):
        import_module(module)

    _REGISTRY_SENTINEL["loaded"] = True


def _load_yaml(path: Path) -> Mapping[str, Any]:
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
    if isinstance(data, dict):
        _normalize_legacy_sections(data)
    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover - compoconf raises rich errors
        raise ConfigLoaderError(f"Unable to parse config {path}: {exc}") from exc

    # Set default monitoring_state_dir (visible, stable location NOT dependent on timestamped runs)
    # This ensures monitoring sessions persist across runs and can be found for --monitor-all
    if root.project.monitoring_state_dir is None:
        # Use stable location: if OUTPUT_DIR is set, use that; otherwise use ./monitoring_state
        # This avoids timestamps in monitoring_state_dir (which would break cross-run monitoring)
        base = Path(root.project.base_output_dir)
        # Strip timestamp suffix if present (e.g., "output/run_20250101" -> "output")
        # by taking parent if base ends with timestamp-like pattern
        stable_base = base.parent if base.name.split("_")[-1].isdigit() and len(base.name.split("_")[-1]) >= 8 else base
        root.project.monitoring_state_dir = stable_base / "monitoring_state"

    # Keep state_dir for backward compatibility (deprecated)
    if root.project.state_dir is None:
        root.project.state_dir = root.project.monitoring_state_dir

    return root


def load_hydra_config(
    config_name: str,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> schema.RootConfig:
    register_default_resolvers()

    overrides = list(overrides or [])
    config_dir = Path(config_dir).resolve()
    if not config_dir.exists():
        raise ConfigLoaderError(f"Hydra config directory not found: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    # Temporarily escape {{...}} placeholders so Hydra doesn't try to parse them
    # These are meant for Python's str.format(), not Hydra interpolation
    def escape_double_braces(obj):
        """Recursively escape {{...}} in strings to prevent Hydra from parsing them."""
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
    if isinstance(data, dict):
        restart_cfg = data.pop("restart", None)
        if restart_cfg is not None and "restart_policies" not in data:
            if isinstance(restart_cfg, dict):
                data["restart_policies"] = restart_cfg.get("policies", [])
        _normalize_legacy_sections(data)
    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover
        raise ConfigLoaderError(f"Unable to parse Hydra config {config_name}: {exc}") from exc

    # Set default monitoring_state_dir (visible, stable location NOT dependent on timestamped runs)
    # This ensures monitoring sessions persist across runs and can be found for --monitor-all
    if root.project.monitoring_state_dir is None:
        # Use stable location: if OUTPUT_DIR is set, use that; otherwise use ./monitoring_state
        # This avoids timestamps in monitoring_state_dir (which would break cross-run monitoring)
        base = Path(root.project.base_output_dir)
        # Strip timestamp suffix if present (e.g., "output/run_20250101" -> "output")
        # by taking parent if base ends with timestamp-like pattern
        stable_base = base.parent if base.name.split("_")[-1].isdigit() and len(base.name.split("_")[-1]) >= 8 else base
        root.project.monitoring_state_dir = stable_base / "monitoring_state"

    # Keep state_dir for backward compatibility (deprecated)
    if root.project.state_dir is None:
        root.project.state_dir = root.project.monitoring_state_dir

    return root


def load_config_reference(
    ref: str | Path,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> schema.RootConfig:
    path = Path(ref)
    if path.exists():
        return load_config(path)
    return load_hydra_config(str(ref), config_dir, overrides)


def ensure_registrations() -> None:
    """Expose registry initialisation for consumers that only instantiate partial configs."""

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
        try:
            return parse_config(schema.MonitorInterface.cfgtype, data)
        except Exception as exc:  # pragma: no cover
            raise ConfigLoaderError(f"Unable to parse monitor config {path}: {exc}") from exc

    root = load_hydra_config(str(ref), config_dir, overrides)
    return root.monitoring


def _normalize_legacy_sections(data: dict[str, Any]) -> None:
    """Flatten legacy `implementation` wrappers for backend and monitoring sections."""

    for key in ("backend", "monitoring"):
        section = data.get(key)
        if not isinstance(section, dict):
            continue
        impl = section.get("implementation")
        if isinstance(impl, dict):
            merged = {**impl}
            for sub_key, value in section.items():
                if sub_key != "implementation":
                    merged[sub_key] = value
            section = merged
            data[key] = section

        if key == "backend":
            namespace = section.get("megatron")
            if isinstance(namespace, dict):
                for sub_key, value in namespace.items():
                    section.setdefault(sub_key, value)
                section.pop("megatron", None)

    slurm_section = data.get("slurm")
    if isinstance(slurm_section, dict):
        client = slurm_section.get("client")
        if isinstance(client, dict):
            name = client.get("class_name")
            if name == "FakeSlurm":
                client["class_name"] = "FakeSlurmClient"
            elif name == "RealSlurm":
                client["class_name"] = "SlurmClient"


__all__ = [
    "ConfigLoaderError",
    "load_config",
    "load_hydra_config",
    "load_config_reference",
    "load_monitoring_reference",
    "ensure_registrations",
]
