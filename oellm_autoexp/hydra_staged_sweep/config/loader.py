"""Helpers for reading user configuration into typed dataclasses."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, TypeVar
from collections.abc import Iterable, Mapping

from compoconf import parse_config, ConfigInterface
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from . import schema
from .resolvers import register_default_resolvers

LOGGER = logging.getLogger(__file__)

T = TypeVar("T", bound=ConfigInterface)

register_default_resolvers()


class ConfigLoaderError(RuntimeError):
    """Raised when the configuration file cannot be parsed."""


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _set_metadata(root: ConfigInterface, config_ref: str | None, config_dir: str | None) -> None:
    if hasattr(root, "metadata") and isinstance(root.metadata, dict):
        if config_ref is not None:
            root.metadata.setdefault("config_ref", str(config_ref))
        if config_dir is not None:
            root.metadata.setdefault("config_dir", str(config_dir))


def _parse_root(
    data: Mapping[str, Any], config_class: type[T], config_ref: str | None, config_dir: str | None
) -> T:
    root = parse_config(config_class, data)

    _set_metadata(root, config_ref, config_dir)
    return root


def load_config(path: str | Path, config_class: type[T] = schema.StagedSweepRoot) -> T:
    """Load and validate a configuration file into ``config_class``."""

    path = Path(path)
    if not path.exists():
        raise ConfigLoaderError(f"Configuration file not found: {path}")

    data = _load_yaml(path)
    if not isinstance(data, Mapping):
        raise ConfigLoaderError(f"Configuration root must be a mapping: {path}")

    return _parse_root(data, config_class, str(path), str(path.parent))


def load_hydra_config(
    config_name: str,
    config_dir: str | Path,
    overrides: Iterable[str] | None = None,
    config_class: type[T] = schema.StagedSweepRoot,
) -> T:
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

    data = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if not isinstance(data, Mapping):
        raise ConfigLoaderError(f"Hydra config {config_name} did not produce a mapping")

    return _parse_root(data, config_class, str(config_name), str(config_dir))


def load_config_reference(
    config_name: str | None = None,
    config_path: str | Path | None = None,
    config_dir: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    config_class: type[T] = schema.StagedSweepRoot,
) -> T:
    if config_name is None and config_path:
        path = Path(config_path)
        if overrides:
            config_reference_path = path.parent / "config_reference.json"
            if config_reference_path.exists():
                import json

                try:
                    reference_data = json.loads(config_reference_path.read_text(encoding="utf-8"))
                    original_config_ref = reference_data.get("config_ref")
                    original_config_dir = reference_data.get("config_dir")
                    original_overrides = reference_data.get("overrides", [])

                    combined_overrides = list(original_overrides) + list(overrides)

                    return load_hydra_config(
                        original_config_ref,
                        original_config_dir or config_dir,
                        combined_overrides,
                        config_class=config_class,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Could not load config_reference.json, falling back to OmegaConf: %s",
                        exc,
                    )

            with initialize_config_dir(version_base=None, config_dir=os.path.abspath(path.parent)):
                cfg = compose(config_name=path.name[:-5], overrides=overrides)

            data = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(data, Mapping):
                raise ConfigLoaderError(f"Config file {path} did not produce a mapping")

            return _parse_root(data, config_class, str(path), str(path.parent))
        else:
            return load_config(path, config_class=config_class)
    return load_hydra_config(config_name, config_dir, overrides, config_class=config_class)


__all__ = [
    "ConfigLoaderError",
    "load_config",
    "load_hydra_config",
    "load_config_reference",
]
