"""Megatron-Bridge checkpoint conversion backend.

Converts a Megatron-LM ``torch_dist`` (or ``torch``) checkpoint into a
HuggingFace-format checkpoint using
``submodules/Megatron-Bridge/examples/conversion/convert_checkpoints.py``.

The whole 4-step flow lives in
``oellm_autoexp.backends.megatron_bridge.run_export``; this backend just
emits the ``python -m oellm_autoexp.backends.megatron_bridge.run_export ...``
command into the sbatch.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field

from compoconf import MissingValue, NonStrictDataclass, register

from oellm_autoexp.backends.base import BaseBackend, BaseBackendConfig

LOGGER = logging.getLogger(__name__)


@dataclass(init=False)
class MegatronBridgeBackendConfig(NonStrictDataclass, BaseBackendConfig):
    """Configuration for the Megatron-Bridge conversion backend."""

    class_name: str = "MegatronBridgeBackend"
    env: dict[str, str] = field(default_factory=dict)

    # Python executable used inside the sbatch (the slurm launcher_cmd typically
    # activates a venv first; "python" then resolves to that venv).
    python_cmd: str = "python"

    # Paths
    bridge_root: str = "submodules/Megatron-Bridge"
    resources: str = "oellm_autoexp/postprocess/resources/megatron_bridge"

    # Required per run
    megatron_path: str = ""
    hf_path: str = ""
    hf_model: str = ""
    tokenizer: str = ""

    # Auto-generated HF reference config. When `derive_hf_arch` is set, the
    # reference `config.json` is synthesised from `megatron_config_path`
    # (typically the train stage's resolved config-<jobid>.yaml) instead of
    # using the vendored `resources/configs/<hf_model>/config.json`.
    derive_hf_arch: str | None = None
    megatron_config_path: str | None = None

    # Misc
    keep_staging: bool = False
    extra_cli_args: list[str] = field(default_factory=list)

    full_cmd: str = MissingValue

    def __post_init__(self) -> None:
        if self.full_cmd is MissingValue:
            self.full_cmd = _build_cmd(self)


def _build_cmd(cfg: MegatronBridgeBackendConfig) -> str:
    parts = [
        cfg.python_cmd,
        "-m",
        "oellm_autoexp.backends.megatron_bridge.run_export",
        f"--megatron-path {shlex.quote(cfg.megatron_path)}",
        f"--hf-path {shlex.quote(cfg.hf_path)}",
        f"--hf-model {shlex.quote(cfg.hf_model)}",
        f"--tokenizer {shlex.quote(cfg.tokenizer)}",
        f"--bridge-root {shlex.quote(cfg.bridge_root)}",
        f"--resources {shlex.quote(cfg.resources)}",
    ]
    if cfg.derive_hf_arch:
        parts.append(f"--derive-hf-arch {shlex.quote(cfg.derive_hf_arch)}")
    if cfg.megatron_config_path:
        parts.append(f"--megatron-config {shlex.quote(cfg.megatron_config_path)}")
    if cfg.keep_staging:
        parts.append("--keep-staging")
    if cfg.extra_cli_args:
        parts.extend(str(a) for a in cfg.extra_cli_args)
    return " ".join(parts)


@register
class MegatronBridgeBackend(BaseBackend):
    config: MegatronBridgeBackendConfig

    def __init__(self, config: MegatronBridgeBackendConfig) -> None:
        super().__init__(config)
        self.config = config

    def validate(self) -> None:
        cfg = self.config
        for field_name in ("megatron_path", "hf_path", "hf_model", "tokenizer"):
            if not getattr(cfg, field_name):
                raise ValueError(f"MegatronBridgeBackend: `{field_name}` is required")

    def build_launch_command(self) -> str:
        return self.config.full_cmd


__all__ = [
    "MegatronBridgeBackend",
    "MegatronBridgeBackendConfig",
]
