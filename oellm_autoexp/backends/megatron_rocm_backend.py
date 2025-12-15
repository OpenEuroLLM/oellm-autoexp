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

import math
from dataclasses import dataclass, field
from typing import Any

from compoconf import asdict, register

from oellm_autoexp.backends.base import BackendJobSpec
from oellm_autoexp.backends.megatron_rocm.cli_metadata import (
    MEGATRON_ACTION_SPECS,
    MEGATRON_ARG_METADATA,
)
from oellm_autoexp.backends.megatron_backend import (
    MegatronBackendConfig,
    MegatronBackend,
)
from oellm_autoexp.backends.megatron_rocm.config_schema import MegatronConfig

from oellm_autoexp.backends.megatron_args import MegatronArgMetadata


@dataclass(init=False)
class MegatronROCmBackendConfig(MegatronBackendConfig):
    """Configuration for the Megatron backend."""

    class_name: str = "MegatronROCmBackend"
    megatron_rocm: MegatronConfig = field(default_factory=MegatronConfig)
    megatron: MegatronConfig | None = None

    def __post_init__(self):
        self.megatron = self.megatron_rocm  # megatron should be actually megatron_rocm config

    def cli_arguments(self) -> dict[str, Any]:
        """Return Megatron arguments from config and dynamic extras."""

        args: dict[str, Any] = {}

        if isinstance(self.megatron, MegatronConfig):
            for key, value in asdict(self.megatron).items():
                if value is not None:
                    args[key] = value

        for key, value in self._extras.items():  # type: ignore[attr-defined]
            if value is not None:
                args[key] = value

        return args


@register
class MegatronROCmBackend(MegatronBackend):
    config: MegatronROCmBackendConfig

    def __init__(self, config: MegatronROCmBackendConfig) -> None:
        self.config = config
        if not config.full_validation:
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


@dataclass(init=False)
class AutoMegatronROCmBackendConfig(MegatronROCmBackendConfig):
    class_name: str = "AutoMegatronROCmBackend"


@register
class AutoMegatronROCmBackend(MegatronROCmBackend):
    config: AutoMegatronROCmBackendConfig

    def _merge_args(self, spec: BackendJobSpec) -> dict[str, Any]:
        raw: dict[str, Any] = {}
        raw.update(self.config.cli_arguments())
        raw.update(spec.parameters)
        normalized = {self._normalize_key(k): v for k, v in raw.items()}
        converted = self._apply_convenience_arguments(normalized)
        return self._filter_megatron_args(converted)

    def _apply_convenience_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
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
    "MegatronROCmBackendConfig",
    "MegatronROCmBackend",
    "AutoMegatronROCmBackendConfig",
    "AutoMegatronROCmBackend",
]
