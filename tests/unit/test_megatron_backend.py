from pathlib import Path

from oellm_autoexp.backends.base import BackendJobSpec


def test_megatron_backend_builds_command():
    from oellm_autoexp.backends.megatron import MegatronBackend, MegatronBackendConfig

    config = MegatronBackendConfig(
        launcher_script=Path("launch.sh"),
        config_overrides={"megatron.lr": 0.1, "megatron.use_gpu": True},
        environment={"ENV": "1"},
        extra_cli_args=["--extra"],
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
    assert cmd.environment["ENV"] == "1"
