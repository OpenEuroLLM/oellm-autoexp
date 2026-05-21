"""Megatron-to-HuggingFace checkpoint conversion step."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

from compoconf import ConfigInterface, register

from oellm_autoexp.postprocess.base import PostProcessStepInterface, _all_ckpt_steps


@dataclass(kw_only=True)
class MegatronToHFStepConfig(ConfigInterface):
    """Config for Megatron-to-HuggingFace checkpoint conversion."""

    class_name: str = "MegatronToHFStep"

    architecture: str = "opensci"
    torch_ckpt_base_dir: str = ""
    hf_output_base_dir: str = ""

    target_tensor_parallel_size: int = 1
    target_pipeline_parallel_size: int = 1
    target_params_dtype: str = "bf16"
    num_key_value_heads: int | None = None

    model_type: str = "opensci"
    architecture_class: str = "OpensciForCausalLM"
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"

    ckpt_step: int | None = None
    save_interval: int | None = None
    train_iters: int | None = None


@register
class MegatronToHFStep(PostProcessStepInterface):
    config: MegatronToHFStepConfig

    def get_run_mode(self) -> Literal["same_job", "new_job"]:
        return "same_job"

    def build_commands(self) -> list[str]:
        cfg = self.config
        if cfg.save_interval is not None and cfg.train_iters is not None:
            steps = _all_ckpt_steps(cfg.save_interval, cfg.train_iters)
        elif cfg.ckpt_step is not None:
            steps = [cfg.ckpt_step]
        else:
            return [self._build_single_command(cfg.torch_ckpt_base_dir, cfg.hf_output_base_dir)]
        return [
            self._build_single_command(
                f"{cfg.torch_ckpt_base_dir}/iter_{step:07d}",
                f"{cfg.hf_output_base_dir}/iter_{step:07d}",
            )
            for step in steps
        ]

    def _build_single_command(self, torch_ckpt_dir: str, hf_output_dir: str) -> str:
        cfg = self.config
        args = [
            "python -m oellm_autoexp.postprocess.converters.mcore_to_hf",
            "--convert_checkpoint_from_megatron_to_transformers",
            f"--prepare_resources {shlex.quote(cfg.architecture)}",
            f"--load_path {shlex.quote(torch_ckpt_dir)}",
            f"--save_path {shlex.quote(hf_output_dir)}",
            f"--source_model {shlex.quote(hf_output_dir)}",
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
        # Note: rank-0 gating is enforced inside mcore_to_hf.main() via
        # SLURM_PROCID, NOT here — wrapping `if/then/fi` around this command
        # at the shell level breaks because the orchestrator prepends
        # `apptainer exec image.sif` and bash then mis-parses `if` as a
        # positional arg to apptainer.


__all__ = ["MegatronToHFStep", "MegatronToHFStepConfig"]
