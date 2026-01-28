"""Helpers for reading user configuration into typed dataclasses.

This module wraps hydra_staged_sweep's config loader and adds oellm-
specific registration and configuration handling.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TypeVar

# Import from oellm_autoexp.hydra_staged_sweep library
from oellm_autoexp.hydra_staged_sweep.config.loader import (
    ConfigLoaderError,
    load_config_reference as _load_config_reference_base,
)
from compoconf import ConfigInterface

from . import schema
from .resolvers import register_default_resolvers

LOGGER = logging.getLogger(__file__)

T = TypeVar("T", bound=ConfigInterface)

_REGISTRY_SENTINEL = {"loaded": False}


def _ensure_registrations() -> None:
    """Ensure oellm-specific types are registered with compoconf."""
    if _REGISTRY_SENTINEL["loaded"]:
        return

    # Import modules for their registration side effects (compoconf @register)
    for module in (
        "oellm_autoexp.backends.base",
        "oellm_autoexp.backends.megatron_backend",
        "oellm_autoexp.config.actions",
        "oellm_autoexp.config.conditions",
    ):
        import_module(module)

    _REGISTRY_SENTINEL["loaded"] = True


def load_config_reference(
    *,
    config_setup: schema.ConfigSetup,
    config_class: type[T] = schema.RootConfig,
) -> T:
    """Load config by name or path, with optional Hydra overrides.

    Args:
        config_name: Config name relative to config_dir (e.g., "experiments/dense_300M")
        config_path: Full path to a specific config file (e.g., "/path/to/config.yaml")
        config_dir: Base config directory (default: "config/")
        overrides: Hydra-style overrides (e.g., ["backend=megatron", "stage=stable"])
        config_class: Configuration dataclass type (default: RootConfig)

    Returns:
        Parsed configuration instance

    Note:
        Provide either config_name OR config_path, not both.
        - config_name: for configs in the config/ directory tree
        - config_path: for standalone YAML files anywhere
    """
    register_default_resolvers()
    _ensure_registrations()

    root = _load_config_reference_base(
        config_name=config_setup.config_name,
        config_path=config_setup.config_path,
        config_dir=config_setup.config_dir,
        overrides=config_setup.overrides,
        config_class=config_class,
    )

    return root


def ensure_registrations() -> None:
    """Expose registry initialisation for consumers that only instantiate
    partial configs."""
    register_default_resolvers()
    _ensure_registrations()


__all__ = [
    "ConfigLoaderError",
    "load_config_reference",
    "ensure_registrations",
]
