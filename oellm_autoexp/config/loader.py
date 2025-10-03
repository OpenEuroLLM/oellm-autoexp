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


def _load_yaml(path: Path) -> Mapping[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def load_config(path: str | Path) -> schema.RootConfig:
    """Load and validate a configuration file into ``RootConfig``."""

    path = Path(path)
    if not path.exists():
        raise ConfigLoaderError(f"Configuration file not found: {path}")

    register_default_resolvers()
    _import_side_effects()

    data = _load_yaml(path)
    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover - compoconf raises rich errors
        raise ConfigLoaderError(f"Unable to parse config {path}: {exc}") from exc

    if root.project.state_dir is None:
        root.project.state_dir = Path(root.project.base_output_dir) / ".oellm-autoexp"

    return root


def load_hydra_config(
    config_name: str,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> schema.RootConfig:
    register_default_resolvers()
    _import_side_effects()

    overrides = list(overrides or [])
    config_dir = Path(config_dir).resolve()
    if not config_dir.exists():
        raise ConfigLoaderError(f"Hydra config directory not found: {config_dir}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)

    data = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(data, dict):
        restart_cfg = data.pop("restart", None)
        if restart_cfg is not None and "restart_policies" not in data:
            if isinstance(restart_cfg, dict):
                data["restart_policies"] = restart_cfg.get("policies", [])
    try:
        root = parse_config(schema.RootConfig, data)
    except Exception as exc:  # pragma: no cover
        raise ConfigLoaderError(f"Unable to parse Hydra config {config_name}: {exc}") from exc

    if root.project.state_dir is None:
        root.project.state_dir = Path(root.project.base_output_dir) / ".oellm-autoexp"

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


def _import_side_effects() -> None:
    """Register built-in implementations with compoconf registries."""

    import_module("oellm_autoexp.backends.base")
    import_module("oellm_autoexp.backends.megatron")
    import_module("oellm_autoexp.monitor.policy")
    import_module("oellm_autoexp.monitor.watcher")


__all__ = [
    "ConfigLoaderError",
    "load_config",
    "load_hydra_config",
    "load_config_reference",
]
