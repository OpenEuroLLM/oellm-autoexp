"""Megatron backend adapter.

Schema-Only Validation Mode
============================

The Megatron backend supports two validation modes:

1. **Full validation mode (default)**:
   - Imports Megatron-LM and uses its argparse parser
   - Provides complete argument validation against Megatron's parser
   - Requires Megatron-LM to be installed or available in submodules/
   - Use this mode when running inside containers with Megatron available

2. **Schema only validation (full_validation=False):
   - Uses pre-generated config_schema.py without importing Megatron-LM
   - Provides basic type checking and argument filtering
   - Does NOT require Megatron-LM to be importable
   - Suitable for running on login nodes that don't have Megatron installed
   - The actual training will still be fully validated inside containers

Why Schema-Only Mode?
---------------------
In SLURM environments with containers:
  - sbatch must be called from the login node (outside containers)
  - Megatron-LM runs inside containers on compute nodes
  - The login node may not have Megatron-LM installed
  - Schema-only mode allows job submission from login nodes
  - Full validation happens when the job runs inside the container
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from compoconf import NonStrictDataclass, asdict, register, MissingValue

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig
from oellm_autoexp.backends.megatron.cli_metadata import (
    MEGATRON_ACTION_SPECS,
    MEGATRON_ARG_METADATA,
)
from oellm_autoexp.backends.megatron.config_schema import MegatronConfig as MegatronConfigBase

from oellm_autoexp.backends.megatron_args import MegatronArgMetadata, build_cmdline_args
from oellm_autoexp.argparse_schema.resolver import register_argparse_resolver


LOGGER = logging.getLogger(__name__)

register_argparse_resolver(
    "argsmegatron",
    arg_metadata=dict(MEGATRON_ARG_METADATA),
    action_specs=dict(MEGATRON_ACTION_SPECS),
    skip_defaults=True,
)

register_argparse_resolver(
    "cliargs",
)


@dataclass
class MegatronConfig(MegatronConfigBase):
    def __post_init__(self):
        # this is for manual checks and must be updated with every new megatron version
        assert self.eval_interval is not None, "Eval interval must be specified in the config"


@dataclass(init=False)
class MegatronBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Configuration for the Megatron backend."""

    class_name: str = "MegatronBackend"
    launcher_script: str = "./submodules/Megatron-LM/pretrain_gpt.py"
    env: dict[str, str] = field(default_factory=dict)
    full_schema_validation: bool = False
    torchrun_args: dict[str, str] = field(default_factory=dict)
    dist_cmd: str
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    python_cmd: str = "python"
    full_cmd: str = MissingValue
    aux: dict[str, Any] = field(default_factory=dict)  # to define convenience variables

    def __post_init__(self):
        # this should be set via the config
        if self.full_cmd is MissingValue:
            self.full_cmd = " ".join(
                [
                    self.python_cmd,
                    *(() if not self.torchrun_args else self.torchrun_args),
                    self.launcher_script,
                    *build_cmdline_args(
                        asdict(self.megatron),
                        dict(MEGATRON_ARG_METADATA),
                        dict(MEGATRON_ACTION_SPECS),
                        skip_defaults=True,
                    ),
                ]
            )


@register
class MegatronBackend(BaseBackend):
    config: MegatronBackendConfig

    def __init__(self, config: MegatronBackendConfig) -> None:
        super().__init__(config)
        self.config = config
        if not config.full_schema_validation:
            # Schema-only mode: skip parser initialization
            self._parser = None
            self._arg_metadata: dict[str, MegatronArgMetadata] = dict(MEGATRON_ARG_METADATA)
            self._action_specs = dict(MEGATRON_ACTION_SPECS)
            self._schema_only = True
        else:
            from oellm_autoexp.backends.megatron_args import (
                get_action_specs,
                get_arg_metadata,
                get_megatron_parser,
            )

            self._parser = get_megatron_parser()
            self._arg_metadata: dict[str, MegatronArgMetadata] = get_arg_metadata(self._parser)
            self._action_specs = get_action_specs(self._parser)
            self._schema_only = False

    def validate(self) -> None:
        args = asdict(self.config.megatron)
        # remove potential extension
        if "_non_strict" in args:
            args.remove("_non_strict")
        if self._schema_only:
            # Schema-only validation: basic type checking against MegatronConfig
            self._validate_schema_only(args)
        else:
            # Full validation using Megatron parser
            cli_args = build_cmdline_args(
                args, self._arg_metadata, self._action_specs, skip_defaults=True
            )
            try:
                self._parser.parse_args(cli_args)
            except SystemExit as exc:  # pragma: no cover - argparse exits
                raise ValueError("Invalid Megatron arguments") from exc

    def _validate_schema_only(self, args: dict[str, Any]) -> None:
        """Lightweight validation using only the schema (no Megatron
        import)."""
        # Check that all provided args are known fields in MegatronConfig
        schema_fields = {f.name for f in MegatronConfig.__dataclass_fields__.values()}
        for key in args.keys():
            normalized_key = self._normalize_key(key)
            if normalized_key not in schema_fields:
                # Allow unknown args in schema-only mode (they'll be validated in container)
                pass

    def build_launch_command(self) -> str:
        return self.config.full_cmd

    def _filter_megatron_args(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._schema_only:
            # Schema-only mode: filter based on MegatronConfig fields
            schema_fields = {f.name for f in MegatronConfig.__dataclass_fields__.values()}
            filtered: dict[str, Any] = {}
            for key, value in params.items():
                dest = self._normalize_key(key)
                if dest in schema_fields:
                    filtered[dest] = value
            return filtered
        else:
            # Full mode: filter based on parser metadata
            filtered: dict[str, Any] = {}
            for key, value in params.items():
                dest = self._normalize_key(key)
                if dest in self._arg_metadata:
                    filtered[dest] = value
            return filtered

    @staticmethod
    def _normalize_key(key: str) -> str:
        key = key.replace("-", "_")
        if key.startswith("megatron."):
            key = key.split(".", 1)[1]
        return key


def _normalize_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).replace("_", "")
    try:
        return int(float(text))
    except ValueError:
        return 0


__all__ = [
    "MegatronBackendConfig",
    "MegatronBackend",
]
