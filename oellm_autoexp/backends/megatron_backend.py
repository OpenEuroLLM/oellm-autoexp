"""Megatron backend adapter that mirrors megatron-train behaviour.

Schema-Only Validation Mode
============================

The Megatron backend supports two validation modes:

1. **Full validation mode (default)**:
   - Imports Megatron-LM and uses its argparse parser
   - Provides complete argument validation against Megatron's parser
   - Requires Megatron-LM to be installed or available in submodules/
   - Use this mode when running inside containers with Megatron available

2. **Schema-only mode (set OELLM_MEGATRON_SCHEMA_ONLY=1)**:
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

Usage:
------
    # Enable schema-only mode (e.g., on login nodes)
    export OELLM_MEGATRON_SCHEMA_ONLY=1
    python scripts/run_autoexp.py ...

    # Or set in Python before importing
    import os
    os.environ["OELLM_MEGATRON_SCHEMA_ONLY"] = "1"
    from oellm_autoexp import ...
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from compoconf import ConfigInterface, NonStrictDataclass, asdict, register

from oellm_autoexp.backends.base import BaseBackend, BackendJobSpec, LaunchCommand
from oellm_autoexp.backends.megatron.config_schema import MegatronConfig

# Schema-only mode: use pre-generated schema without importing Megatron-LM
# This allows validation on login nodes without containers
USE_SCHEMA_ONLY = os.environ.get("OELLM_MEGATRON_SCHEMA_ONLY", "0") == "1"

if not USE_SCHEMA_ONLY:
    from oellm_autoexp.backends.megatron_args import (
        MegatronArgMetadata,
        build_cmdline_args,
        get_arg_metadata,
        get_megatron_parser,
    )


@dataclass(init=False)
class MegatronBackendConfig(NonStrictDataclass, ConfigInterface):
    """Configuration for the Megatron backend."""

    class_name: str = "MegatronBackend"
    launcher_script: Path = Path("scripts/run_megatron.sh")
    env: Dict[str, str] = field(default_factory=dict)
    extra_cli_args: list[str] = field(default_factory=list)
    use_torchrun: bool = False
    torchrun_args: Dict[str, Any] = field(default_factory=dict)
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    differential_cmd: bool = True  # if to only pass non-default arguments

    def cli_arguments(self) -> Dict[str, Any]:
        """Return Megatron arguments from config and dynamic extras."""

        args: Dict[str, Any] = {}

        if isinstance(self.megatron, MegatronConfig):
            for key, value in asdict(self.megatron).items():
                if value is not None:
                    args[key] = value

        for key, value in self._extras.items():  # type: ignore[attr-defined]
            if value is not None:
                args[key] = value

        return args


@register
class MegatronBackend(BaseBackend):
    config: MegatronBackendConfig

    def __init__(self, config: MegatronBackendConfig) -> None:
        super().__init__(config)
        self.config = config
        if USE_SCHEMA_ONLY:
            # Schema-only mode: skip parser initialization
            self._parser = None
            self._arg_metadata = {}
            self._schema_only = True
        else:
            self._parser = get_megatron_parser()
            self._arg_metadata: Dict[str, MegatronArgMetadata] = get_arg_metadata(self._parser)
            self._schema_only = False

    def validate(self, spec: BackendJobSpec) -> None:
        if self._schema_only:
            # Schema-only validation: basic type checking against MegatronConfig
            merged_args = self._merge_args(spec)
            self._validate_schema_only(merged_args)
        else:
            # Full validation using Megatron parser
            merged_args = self._merge_args(spec)
            cli_args = build_cmdline_args(merged_args, self._parser, self._arg_metadata)
            try:
                self._parser.parse_args(cli_args)
            except SystemExit as exc:  # pragma: no cover - argparse exits
                raise ValueError("Invalid Megatron arguments") from exc

    def _validate_schema_only(self, args: Dict[str, Any]) -> None:
        """Lightweight validation using only the schema (no Megatron import)."""
        # Check that all provided args are known fields in MegatronConfig
        schema_fields = {f.name for f in MegatronConfig.__dataclass_fields__.values()}
        for key in args.keys():
            normalized_key = self._normalize_key(key)
            if normalized_key not in schema_fields:
                # Allow unknown args in schema-only mode (they'll be validated in container)
                pass

    def build_launch_command(self, spec: BackendJobSpec) -> LaunchCommand:
        merged_args = self._merge_args(spec)

        if self._schema_only:
            # Schema-only mode: build simple command line args
            cli_args = self._build_simple_cmdline_args(merged_args)
        else:
            # Full mode: use megatron_args builder
            cli_args = build_cmdline_args(merged_args, self._parser, self._arg_metadata)

        argv = [str(self.config.launcher_script), *cli_args, *self.config.extra_cli_args]

        if (
            not self.config.use_torchrun
            and self.config.launcher_script.suffix == ".py"
        ):
            argv = ["python", *argv]

        # Prepend torchrun if enabled
        if self.config.use_torchrun:
            torchrun_argv = self._build_torchrun_args()
            argv = [*torchrun_argv, *argv]

        env = {**self.config.env}
        return LaunchCommand(argv=argv, env=env)

    def _build_simple_cmdline_args(self, args: Dict[str, Any]) -> list[str]:
        """Build command line args without megatron_args (schema-only mode)."""
        base_cfg = MegatronConfig()
        cli_args = []
        for key, value in args.items():
            if value is None:
                continue
            if getattr(base_cfg, key) == value and self.config.differential_cmd:
                continue

            # Convert to CLI format
            cli_key = key.replace("_", "-")

            if isinstance(value, bool):
                if value:
                    cli_args.append(f"--{cli_key}")
            elif isinstance(value, list):
                cli_args.append(f"--{cli_key}")
                cli_args.append(",".join(str(v) for v in value))
            else:
                cli_args.append(f"--{cli_key}")
                cli_args.append(str(value))

        return cli_args

    def _build_torchrun_args(self) -> list[str]:
        """Build torchrun command line arguments."""
        args = ["python", "-u", "-m", "torch.distributed.run"]

        for key, value in self.config.torchrun_args.items():
            key_formatted = key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key_formatted}")
            elif value is not None:
                args.append(f"--{key_formatted}")
                args.append(str(value))

        return args

    def _merge_args(self, spec: BackendJobSpec) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        args.update(self._filter_megatron_args(self.config.cli_arguments()))
        args.update(self._filter_megatron_args(spec.parameters))
        return args

    def _filter_megatron_args(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._schema_only:
            # Schema-only mode: filter based on MegatronConfig fields
            schema_fields = {f.name for f in MegatronConfig.__dataclass_fields__.values()}
            filtered: Dict[str, Any] = {}
            for key, value in params.items():
                dest = self._normalize_key(key)
                if dest in schema_fields:
                    filtered[dest] = value
            return filtered
        else:
            # Full mode: filter based on parser metadata
            filtered: Dict[str, Any] = {}
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


@dataclass(init=False)
class AutoMegatronBackendConfig(MegatronBackendConfig):
    class_name: str = "AutoMegatronBackend"


@register
class AutoMegatronBackend(MegatronBackend):
    config: AutoMegatronBackendConfig

    def _merge_args(self, spec: BackendJobSpec) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}
        raw.update(self.config.cli_arguments())
        raw.update(spec.parameters)
        normalized = {self._normalize_key(k): v for k, v in raw.items()}
        converted = self._apply_convenience_arguments(normalized)
        return self._filter_megatron_args(converted)

    def _apply_convenience_arguments(self, args: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(args)

        if "train_tokens" in args:
            train_tokens = _normalize_int(args.pop("train_tokens"))
            seq_length = _normalize_int(args.get("seq_length"))
            global_batch = _normalize_int(args.get("global_batch_size"))
            if seq_length and global_batch:
                tokens_per_iter = seq_length * global_batch
                if tokens_per_iter > 0:
                    args["train_iters"] = math.ceil(train_tokens / tokens_per_iter)

        if "lr_decay_fraction" in args:
            fraction = float(args.pop("lr_decay_fraction"))
            if "train_samples" in args:
                args["lr_decay_samples"] = int(_normalize_int(args["train_samples"]) * fraction)
            elif "train_iters" in args:
                args["lr_decay_iters"] = int(_normalize_int(args["train_iters"]) * fraction)

        if "lr_decay_iters" in args and "lr_wsd_decay_iters" not in args:
            args["lr_wsd_decay_iters"] = args["lr_decay_iters"]
        if "lr_decay_samples" in args and "lr_wsd_decay_samples" not in args:
            args["lr_wsd_decay_samples"] = args["lr_decay_samples"]

        return args


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
    "AutoMegatronBackendConfig",
    "AutoMegatronBackend",
]
