"""MaxText (JAX) backend adapter.

Generates a flat CLI launch command for MaxText training. MaxText uses JAX
(not PyTorch), so there is no torchrun — JAX distributed is driven by env
vars (JAX_COORDINATOR_ADDRESS, JAX_PROCESS_COUNT, JAX_PROCESS_ID) and a
single process per node launched via srun.

Design:
  Hydra config → NonStrictDataclass → flat key=value CLI args → python train.py base.yml <overrides>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from compoconf import NonStrictDataclass, asdict, register

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig
from oellm_autoexp.backends.titan_backend import _prune_internal_keys

LOGGER = logging.getLogger(__name__)

# Keys that are internal to the backend config and should NOT be emitted as
# MaxText CLI overrides.
_INTERNAL_KEYS = frozenset(
    {
        "class_name",
        "launcher_script",
        "base_config_path",
        "extra_cli_args",
        "env",
        "full_schema_validation",
    }
)


@dataclass(init=False)
class MaxTextJobConfig(NonStrictDataclass):
    """MaxText job configuration (non-strict to allow arbitrary MaxText
    flags)."""

    class_name: str = "MaxTextJobConfig"


@dataclass(init=False)
class MaxTextBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Configuration for the MaxText (JAX) backend."""

    class_name: str = "MaxTextBackend"
    env: dict[str, str] = field(default_factory=dict)

    # Job configuration (non-strict: preserves arbitrary MaxText keys)
    maxtext: MaxTextJobConfig = field(default_factory=MaxTextJobConfig)

    # Launch settings
    launcher_script: str = "./submodules/MaxText/MaxText/train.py"
    base_config_path: str = "./submodules/MaxText/MaxText/configs/base.yml"
    extra_cli_args: list[str] = field(default_factory=list)

    # Validation
    full_schema_validation: bool = False


@register
class MaxTextBackend(BaseBackend):
    config: MaxTextBackendConfig

    def __init__(self, config: MaxTextBackendConfig) -> None:
        super().__init__(config)
        self.config = config

    def validate(self) -> None:
        if not self.config.full_schema_validation:
            return
        LOGGER.info("Full MaxText validation not yet implemented; skipping.")

    def build_launch_command(self) -> str:
        # No torchrun — JAX uses env vars for distributed coordination.
        # MaxText entry: python train.py base.yml key=value key=value ...
        script = self.config.launcher_script
        base_config = self.config.base_config_path

        overrides = self._build_cli_overrides()
        cmd = f"python {script} {base_config}"
        if overrides:
            cmd = f"{cmd} {overrides}"
        if self.config.extra_cli_args:
            cmd = f"{cmd} {' '.join(self.config.extra_cli_args)}"
        return cmd

    def _build_cli_overrides(self) -> str:
        """Convert the maxtext config section to flat MaxText CLI format: key=value pairs."""
        data = asdict(self.config.maxtext)
        _prune_internal_keys(data)
        flat = self._flatten(data)
        return " ".join(f"{k}={v}" for k, v in flat.items())

    @staticmethod
    def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
        """Flatten nested dict to flat key=value pairs for MaxText CLI."""
        result: dict[str, str] = {}
        for key, value in data.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if value is None:
                continue
            if isinstance(value, dict):
                result.update(MaxTextBackend._flatten(value, full_key))
            elif isinstance(value, bool):
                result[full_key] = "true" if value else "false"
            elif isinstance(value, list):
                # MaxText uses space-separated lists or comma-separated
                result[full_key] = str(value)
            else:
                result[full_key] = str(value)
        return result


__all__ = [
    "MaxTextBackendConfig",
    "MaxTextBackend",
    "MaxTextJobConfig",
]
