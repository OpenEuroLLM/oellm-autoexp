from pathlib import Path

from oellm_autoexp.backends.base import BackendJobSpec


def test_megatron_backend_builds_command():
    from oellm_autoexp.backends.megatron_backend import MegatronBackend, MegatronBackendConfig

    config = MegatronBackendConfig(
        launcher_script=Path("launch.sh"),
        env={"ENV": "1"},
        extra_cli_args=["--extra"],
        lr=0.1,
        use_gpu=True,
    )
    backend = MegatronBackend(config)

    spec = BackendJobSpec(parameters={"megatron.micro_batch_size": 4})

    backend.validate(spec)
    cmd = backend.build_launch_command(spec)

    assert cmd.argv[0] == "launch.sh"
    assert "--lr" in cmd.argv and "0.1" in cmd.argv
    assert "--micro-batch-size" in cmd.argv and "4" in cmd.argv
    assert "--use-gpu" in cmd.argv
    assert cmd.argv[-1] == "--extra"
    assert cmd.env["ENV"] == "1"


def test_auto_megatron_backend_converts_convenience_args():
    from oellm_autoexp.backends.megatron_backend import (
        AutoMegatronBackend,
        AutoMegatronBackendConfig,
    )

    config = AutoMegatronBackendConfig(launcher_script=Path("launch.sh"))
    backend = AutoMegatronBackend(config)

    normalized = {
        "train_tokens": "1_000",
        "seq_length": 10,
        "global_batch_size": 10,
        "lr_decay_fraction": 0.5,
    }
    converted = backend._apply_convenience_arguments(normalized)
    assert converted.get("train_iters") == 10
    assert "train_tokens" not in converted
    assert converted.get("lr_decay_iters") == 5
    assert converted.get("lr_wsd_decay_iters") == 5
