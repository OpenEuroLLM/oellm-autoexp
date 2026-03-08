"""Tests for the postprocess step system."""

from __future__ import annotations

import pytest

from oellm_autoexp.postprocess.base import PostProcessStepInterface
from oellm_autoexp.postprocess.megatron_to_hf import MegatronToHFStep, MegatronToHFStepConfig
from oellm_autoexp.postprocess.oellm_eval import OELLMEvalStep, OELLMEvalStepConfig


# Test classes follow pipeline execution order: torch→HF → eval


class TestMegatronToHFStep:
    def test_run_mode_is_same_job(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_dir="/tmp/torch/iter_1000",
            hf_output_dir="/tmp/hf/iter_1000",
        )
        step = MegatronToHFStep(cfg)
        assert step.get_run_mode() == "same_job"

    def test_build_command_uses_python_m(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_dir="/tmp/torch/iter_1000",
            hf_output_dir="/tmp/hf/iter_1000",
            num_key_value_heads=8,
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_command()

        assert "python -m oellm_autoexp.postprocess.converters.mcore_to_hf" in cmd
        assert "--convert_checkpoint_from_megatron_to_transformers" in cmd
        assert "--load_path /tmp/torch/iter_1000" in cmd
        assert "--save_path /tmp/hf/iter_1000" in cmd
        assert "--num_key_value_heads 8" in cmd
        assert "--prepare_resources opensci" in cmd

    def test_build_command_without_num_key_value_heads(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_dir="/tmp/torch/iter_1000",
            hf_output_dir="/tmp/hf/iter_1000",
            num_key_value_heads=None,
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_command()
        assert "--num_key_value_heads" not in cmd

    def test_build_command_prepare_resources(self):
        cfg = MegatronToHFStepConfig(
            architecture="opensci",
            torch_ckpt_dir="/tmp/torch/iter_1000",
            hf_output_dir="/tmp/hf/iter_1000",
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_command()
        assert "--prepare_resources opensci" in cmd

    def test_build_command_custom_params_dtype(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_dir="/tmp/torch",
            hf_output_dir="/tmp/hf",
            target_params_dtype="fp32",
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_command()
        assert "--target_params_dtype fp32" in cmd

    def test_build_command_no_script_path(self):
        """Verify no file path dependency — uses python -m instead."""
        cfg = MegatronToHFStepConfig(
            torch_ckpt_dir="/tmp/torch",
            hf_output_dir="/tmp/hf",
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_command()
        assert "scripts/" not in cmd
        assert "python -m" in cmd

    def test_is_postprocess_step(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_dir="/tmp/torch",
            hf_output_dir="/tmp/hf",
        )
        step = MegatronToHFStep(cfg)
        assert isinstance(step, PostProcessStepInterface)


class TestOELLMEvalStep:
    def test_run_mode_is_new_job(self):
        cfg = OELLMEvalStepConfig(model_path="/tmp/hf/iter_1000")
        step = OELLMEvalStep(cfg)
        assert step.get_run_mode() == "new_job"

    def test_build_command_default_task_group(self):
        cfg = OELLMEvalStepConfig(model_path="/tmp/hf/iter_1000")
        step = OELLMEvalStep(cfg)
        cmd = step.build_command()

        assert "oellm schedule-eval" in cmd
        assert "--models /tmp/hf/iter_1000" in cmd
        assert "open-sci-0.01" in cmd
        assert "--dry_run" not in cmd

    def test_build_command_multiple_task_groups(self):
        cfg = OELLMEvalStepConfig(
            model_path="/tmp/hf",
            task_groups=["open-sci-0.01", "dclm-core-22"],
        )
        step = OELLMEvalStep(cfg)
        cmd = step.build_command()
        assert "open-sci-0.01 dclm-core-22" in cmd

    def test_build_command_dry_run(self):
        cfg = OELLMEvalStepConfig(
            model_path="/tmp/hf",
            dry_run=True,
        )
        step = OELLMEvalStep(cfg)
        cmd = step.build_command()
        assert "--dry_run" in cmd

    def test_is_postprocess_step(self):
        cfg = OELLMEvalStepConfig(model_path="/tmp/hf")
        step = OELLMEvalStep(cfg)
        assert isinstance(step, PostProcessStepInterface)


class TestPostProcessStepInterface:
    def test_interface_registered(self):
        assert hasattr(PostProcessStepInterface, "cfgtype")

    def test_instantiate_megatron_to_hf(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_dir="/tmp/torch",
            hf_output_dir="/tmp/hf",
        )
        step = cfg.instantiate(PostProcessStepInterface)
        assert isinstance(step, MegatronToHFStep)

    def test_instantiate_oellm_eval(self):
        cfg = OELLMEvalStepConfig(model_path="/tmp/hf")
        step = cfg.instantiate(PostProcessStepInterface)
        assert isinstance(step, OELLMEvalStep)
