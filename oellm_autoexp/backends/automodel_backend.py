"""NeMo Automodel backend adapter.

Generates a YAML recipe file for NeMo Automodel and produces a torchrun
launch command that invokes the pretrain/benchmark script.

The design follows the TitanBackend pattern:
  Hydra config  →  NonStrictDataclass  →  asdict()  →  YAML file  →  torchrun cmd
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from compoconf import NonStrictDataclass, asdict, register

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig
from oellm_autoexp.backends.titan_backend import _prune_internal_keys

LOGGER = logging.getLogger(__name__)


@dataclass(init=False)
class AutomodelJobConfig(NonStrictDataclass):
    """NeMo Automodel job configuration (non-strict to allow _target_
    fields)."""

    class_name: str = "AutomodelJobConfig"


@dataclass(init=False)
class AutomodelBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Configuration for the NeMo Automodel backend."""

    class_name: str = "AutomodelBackend"
    env: dict[str, str] = field(default_factory=dict)

    # Job configuration (non-strict: preserves _target_ keys, arbitrary nesting)
    automodel: AutomodelJobConfig = field(default_factory=AutomodelJobConfig)

    # Launch settings
    launcher_script: str = "./submodules/Automodel/examples/llm_pretrain/pretrain.py"
    torchrun_args: dict[str, Any] = field(default_factory=dict)
    extra_cli_args: list[str] = field(default_factory=list)

    # Generated YAML artifact
    yaml_output_path: str = ""

    # Validation
    full_schema_validation: bool = False


@register
class AutomodelBackend(BaseBackend):
    config: AutomodelBackendConfig

    def __init__(self, config: AutomodelBackendConfig) -> None:
        super().__init__(config)
        self.config = config

    def validate(self) -> None:
        if not self.config.full_schema_validation:
            return
        # Full validation would import nemo_automodel and parse the config.
        # For now, schema-only mode is sufficient (actual validation happens
        # inside the container at runtime).
        LOGGER.info("Full Automodel validation not yet implemented; skipping.")

    def build_launch_command(self) -> str:
        yaml_path = self._ensure_yaml()
        torchrun = self._build_torchrun_cmd()
        if self.config.launcher_script.startswith("-m "):
            # Module mode: positional config arg for nemo_automodel.cli.app
            cmd = f"{torchrun} {self.config.launcher_script} {yaml_path}"
        else:
            # Script mode (backwards compat): --config flag
            cmd = f"{torchrun} {self.config.launcher_script} --config {yaml_path}"
        if self.config.extra_cli_args:
            cmd = f"{cmd} {' '.join(self.config.extra_cli_args)}"
        return cmd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_torchrun_cmd(self) -> str:
        """Build the ``torchrun`` prefix from *torchrun_args*."""
        parts = ["torchrun"]
        for key, value in self.config.torchrun_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    parts.append(flag)
                continue
            parts.append(f"{flag}={value}")
        return " ".join(parts)

    def _ensure_yaml(self) -> str:
        """Serialise the Automodel job config to a YAML file and return its
        path."""
        output_path = self.config.yaml_output_path or "config.automodel.yaml"
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._build_yaml_dict()
        yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
        path.write_text(yaml_text, encoding="utf-8")
        return str(path)

    def _build_yaml_dict(self) -> dict[str, Any]:
        """Convert the Automodel config dataclass to a plain dict for YAML
        output."""
        data = asdict(self.config.automodel)
        _prune_internal_keys(data)
        return data


__all__ = [
    "AutomodelBackendConfig",
    "AutomodelBackend",
    "AutomodelJobConfig",
]
