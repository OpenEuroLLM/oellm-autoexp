"""The collaborator recipe must preserve the tested treatment and private outputs."""
from pathlib import Path

import pytest

from oellm_autoexp.backends.megatron_backend import MegatronBackend
from oellm_autoexp.config.loader import load_config_reference
from oellm_autoexp.config.schema import ConfigSetup


def test_ce_fp32_recipe(monkeypatch, tmp_path):
    for key, value in {
        "CE_RUN_ROOT": str(tmp_path / "run"),
        "CE_RUN_NAME": "ce-fp32-test",
        "CE_LOAD_PATH": str(tmp_path / "seed"),
        "CE_REMAINING_STEPS": "4000",
        "CE_END_STEP": "110000",
        "PROJECT_DIR": str(Path.cwd()),
        "SLURM_ACCOUNT": "test-account",
        "JUPITER_EXCLUDE_NODES": str(tmp_path / "exclude.txt"),
    }.items():
        monkeypatch.setenv(key, value)
    cfg = load_config_reference(config_setup=ConfigSetup(
        config_dir=Path("config"),
        config_name="experiments/oellm_32b_dense/oellm_32b_dense_revival_1_ce_fp32",
        overrides=["backend.megatron.data_args_path=null"],
    ))
    cmd = MegatronBackend(cfg.backend).build_launch_command()
    # The argument renderer omits native because it is Megatron's default.
    assert cfg.backend.megatron.cross_entropy_fusion_impl == "native"
    assert "--cross-entropy-fusion-impl te" not in cmd
    assert "--output-z-loss-coeff 0.0001" in cmd
    assert "--no-load-optim" not in cmd and "--no-load-rng" not in cmd
    assert "--exit-interval 110000" in cmd
    assert "--wandb-project oellm_32b_dense_loss-increase_debug" in cmd
    assert str(tmp_path / "seed") in cmd
    assert str(tmp_path / "run/training_ckpts/checkpoints") in cmd
    assert cfg.slurm.env["SLURM_MPI_TYPE"] == "none"
    assert "--mpi=none" in cfg.slurm.srun_opts
    assert "--cpus-per-task=288" in cfg.slurm.srun_opts
    assert cfg.job.checkpoint_hook_command == ""
    assert cfg.job.restart_pre_command == ""
    assert len(cfg.job.log_events) == 37


def test_ce_fp32_requires_explicit_run_settings(monkeypatch):
    for key in ["CE_RUN_ROOT", "CE_LOAD_PATH", "CE_REMAINING_STEPS", "CE_END_STEP"]:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(Exception, match="CE_RUN_ROOT|CE_LOAD_PATH|CE_REMAINING_STEPS|CE_END_STEP"):
        load_config_reference(config_setup=ConfigSetup(
            config_dir=Path("config"),
            config_name="experiments/oellm_32b_dense/oellm_32b_dense_revival_1_ce_fp32",
            overrides=["backend.megatron.data_args_path=null"],
        ))
