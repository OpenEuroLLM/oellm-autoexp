"""Tests for the postprocess step system."""

from __future__ import annotations


from oellm_autoexp.postprocess.base import PostProcessStepInterface, _all_ckpt_steps
from oellm_autoexp.postprocess.megatron_dist_to_torch import MegatronDistToTorchStep, MegatronDistToTorchStepConfig
from oellm_autoexp.postprocess.megatron_to_hf import MegatronToHFStep, MegatronToHFStepConfig
from oellm_autoexp.postprocess.oellm_eval import OELLMEvalStep, OELLMEvalStepConfig


# Test classes follow pipeline execution order: torch→HF → eval


class TestAllCkptSteps:
    def test_exact_multiple(self):
        assert _all_ckpt_steps(2000, 6000) == [2000, 4000, 6000]

    def test_non_multiple_train_iters(self):
        steps = _all_ckpt_steps(2000, 71526)
        assert steps[0] == 2000
        assert steps[-1] == 71526
        assert 70000 in steps
        assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))

    def test_single_step(self):
        assert _all_ckpt_steps(1000, 1000) == [1000]


class TestMegatronDistToTorchStep:
    def test_run_mode_is_same_job(self):
        cfg = MegatronDistToTorchStepConfig(cmd="python train.py")
        step = MegatronDistToTorchStep(cfg)
        assert step.get_run_mode() == "same_job"

    def test_no_steps_configured_returns_cmd(self):
        cfg = MegatronDistToTorchStepConfig(cmd="python train.py")
        step = MegatronDistToTorchStep(cfg)
        assert step.build_commands() == ["python train.py"]

    def test_single_step_no_load_dir_returns_cmd(self):
        cfg = MegatronDistToTorchStepConfig(cmd="python train.py", ckpt_step=1000)
        step = MegatronDistToTorchStep(cfg)
        assert step.build_commands() == ["python train.py"]

    def test_single_step_with_load_dir(self):
        cfg = MegatronDistToTorchStepConfig(
            cmd="python train.py",
            load_dir="/tmp/ckpt",
            ckpt_step=1000,
        )
        step = MegatronDistToTorchStep(cfg)
        cmds = step.build_commands()
        assert len(cmds) == 2
        assert cmds[0] == "echo 1000 > /tmp/ckpt/latest_checkpointed_iteration.txt"
        assert cmds[1] == "python train.py"

    def test_multi_step_with_load_dir(self):
        cfg = MegatronDistToTorchStepConfig(
            cmd="python train.py",
            load_dir="/tmp/ckpt",
            save_interval=2000,
            train_iters=6000,
        )
        step = MegatronDistToTorchStep(cfg)
        cmds = step.build_commands()
        # 3 checkpoints × 2 commands each
        assert len(cmds) == 6
        tracker = "/tmp/ckpt/latest_checkpointed_iteration.txt"
        assert cmds[0] == f"echo 2000 > {tracker}"
        assert cmds[1] == "python train.py"
        assert cmds[2] == f"echo 4000 > {tracker}"
        assert cmds[3] == "python train.py"
        assert cmds[4] == f"echo 6000 > {tracker}"
        assert cmds[5] == "python train.py"

    def test_multi_step_no_load_dir_returns_single_cmd(self):
        cfg = MegatronDistToTorchStepConfig(
            cmd="python train.py",
            save_interval=2000,
            train_iters=6000,
        )
        step = MegatronDistToTorchStep(cfg)
        assert step.build_commands() == ["python train.py"]

    def test_non_multiple_train_iters(self):
        cfg = MegatronDistToTorchStepConfig(
            cmd="python train.py",
            load_dir="/tmp/ckpt",
            save_interval=2000,
            train_iters=5500,
        )
        step = MegatronDistToTorchStep(cfg)
        cmds = step.build_commands()
        # 3 checkpoints: 2000, 4000, 5500
        assert len(cmds) == 6
        tracker = "/tmp/ckpt/latest_checkpointed_iteration.txt"
        assert cmds[0] == f"echo 2000 > {tracker}"
        assert cmds[4] == f"echo 5500 > {tracker}"

    def test_is_postprocess_step(self):
        cfg = MegatronDistToTorchStepConfig(cmd="python train.py")
        step = MegatronDistToTorchStep(cfg)
        assert isinstance(step, PostProcessStepInterface)


class TestMegatronToHFStep:
    def test_run_mode_is_same_job(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            ckpt_step=1000,
        )
        step = MegatronToHFStep(cfg)
        assert step.get_run_mode() == "same_job"

    def test_build_commands_uses_python_m(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            num_key_value_heads=8,
            ckpt_step=1000,
        )
        step = MegatronToHFStep(cfg)
        cmds = step.build_commands()

        assert len(cmds) == 1
        cmd = cmds[0]
        assert "python -m oellm_autoexp.postprocess.converters.mcore_to_hf" in cmd
        assert "--convert_checkpoint_from_megatron_to_transformers" in cmd
        assert "--load_path /tmp/torch/iter_0001000" in cmd
        assert "--save_path /tmp/hf/iter_0001000" in cmd
        assert "--num_key_value_heads 8" in cmd
        assert "--prepare_resources opensci" in cmd

    def test_build_commands_without_num_key_value_heads(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            num_key_value_heads=None,
            ckpt_step=1000,
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_commands()[0]
        assert "--num_key_value_heads" not in cmd

    def test_build_commands_prepare_resources(self):
        cfg = MegatronToHFStepConfig(
            architecture="opensci",
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            ckpt_step=1000,
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_commands()[0]
        assert "--prepare_resources opensci" in cmd

    def test_build_commands_custom_params_dtype(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            target_params_dtype="fp32",
            ckpt_step=5,
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_commands()[0]
        assert "--target_params_dtype fp32" in cmd

    def test_build_commands_no_script_path(self):
        """Verify no file path dependency — uses python -m instead."""
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            ckpt_step=5,
        )
        step = MegatronToHFStep(cfg)
        cmd = step.build_commands()[0]
        assert "scripts/" not in cmd
        assert "python -m" in cmd

    def test_build_commands_all_steps(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            save_interval=2000,
            train_iters=6000,
        )
        step = MegatronToHFStep(cfg)
        cmds = step.build_commands()

        assert len(cmds) == 3
        assert "--load_path /tmp/torch/iter_0002000" in cmds[0]
        assert "--load_path /tmp/torch/iter_0004000" in cmds[1]
        assert "--load_path /tmp/torch/iter_0006000" in cmds[2]
        assert "--save_path /tmp/hf/iter_0002000" in cmds[0]

    def test_build_commands_non_multiple_train_iters(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
            save_interval=2000,
            train_iters=5500,
        )
        step = MegatronToHFStep(cfg)
        cmds = step.build_commands()

        assert len(cmds) == 3  # 2000, 4000, 5500
        assert "--load_path /tmp/torch/iter_0002000" in cmds[0]
        assert "--load_path /tmp/torch/iter_0004000" in cmds[1]
        assert "--load_path /tmp/torch/iter_0005500" in cmds[2]

    def test_is_postprocess_step(self):
        cfg = MegatronToHFStepConfig(
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
        )
        step = MegatronToHFStep(cfg)
        assert isinstance(step, PostProcessStepInterface)


class TestOELLMEvalStep:
    def test_run_mode_is_new_job(self):
        cfg = OELLMEvalStepConfig(model_path="/tmp/hf/iter_1000")
        step = OELLMEvalStep(cfg)
        assert step.get_run_mode() == "new_job"

    def test_build_commands_default_task_group(self):
        cfg = OELLMEvalStepConfig(model_path="/tmp/hf/iter_1000")
        step = OELLMEvalStep(cfg)
        cmds = step.build_commands()

        assert len(cmds) == 1
        cmd = cmds[0]
        assert "oellm schedule-eval" in cmd
        assert "--models /tmp/hf/iter_1000" in cmd
        assert "open-sci-0.01" in cmd
        assert "--dry_run" not in cmd

    def test_build_commands_multiple_task_groups(self):
        cfg = OELLMEvalStepConfig(
            model_path="/tmp/hf",
            task_groups=["open-sci-0.01", "dclm-core-22"],
        )
        step = OELLMEvalStep(cfg)
        cmd = step.build_commands()[0]
        assert "open-sci-0.01 dclm-core-22" in cmd

    def test_build_commands_dry_run(self):
        cfg = OELLMEvalStepConfig(
            model_path="/tmp/hf",
            dry_run=True,
        )
        step = OELLMEvalStep(cfg)
        cmd = step.build_commands()[0]
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
            torch_ckpt_base_dir="/tmp/torch",
            hf_output_base_dir="/tmp/hf",
        )
        step = cfg.instantiate(PostProcessStepInterface)
        assert isinstance(step, MegatronToHFStep)

    def test_instantiate_oellm_eval(self):
        cfg = OELLMEvalStepConfig(model_path="/tmp/hf")
        step = cfg.instantiate(PostProcessStepInterface)
        assert isinstance(step, OELLMEvalStep)
