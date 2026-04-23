"""Megatron-to-HuggingFace checkpoint conversion step."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

from compoconf import ConfigInterface, register

from oellm_autoexp.postprocess.base import PostProcessStepInterface


@dataclass(kw_only=True)
class MegatronToHFStepConfig(ConfigInterface):
    """Config for Megatron-to-HuggingFace checkpoint conversion."""

    class_name: str = "MegatronToHFStep"

    architecture: str = "opensci"
    torch_ckpt_dir: str = ""
    hf_output_dir: str = ""

    target_tensor_parallel_size: int = 1
    target_pipeline_parallel_size: int = 1
    target_params_dtype: str = "bf16"
    num_key_value_heads: int | None = None

    model_type: str = "opensci"
    architecture_class: str = "OpensciForCausalLM"
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"


@register
class MegatronToHFStep(PostProcessStepInterface):
    config: MegatronToHFStepConfig

    def get_run_mode(self) -> Literal["same_job", "new_job"]:
        return "same_job"

    def build_command(self) -> str:
        cfg = self.config
        args = [
            "python -m oellm_autoexp.postprocess.converters.mcore_to_hf",
            "--convert_checkpoint_from_megatron_to_transformers",
            f"--prepare_resources {shlex.quote(cfg.architecture)}",
            f"--load_path {shlex.quote(cfg.torch_ckpt_dir)}",
            f"--save_path {shlex.quote(cfg.hf_output_dir)}",
            f"--source_model {shlex.quote(cfg.hf_output_dir)}",
            f"--target_tensor_model_parallel_size {cfg.target_tensor_parallel_size}",
            f"--target_pipeline_model_parallel_size {cfg.target_pipeline_parallel_size}",
            f"--target_params_dtype {shlex.quote(cfg.target_params_dtype)}",
            f"--world_size {cfg.target_tensor_parallel_size * cfg.target_pipeline_parallel_size}",
            f"--model_type {shlex.quote(cfg.model_type)}",
            f"--architecture {shlex.quote(cfg.architecture_class)}",
            f"--tokenizer_name {shlex.quote(cfg.tokenizer_name)}",
        ]
        if cfg.num_key_value_heads is not None:
            args.append(f"--num_key_value_heads {cfg.num_key_value_heads}")
        return " \\\n    ".join(args)


__all__ = ["MegatronToHFStep", "MegatronToHFStepConfig"]
