def test_megatron_backend_builds_command():
    from oellm_autoexp.backends.megatron_backend import (
        MegatronBackend,
        MegatronBackendConfig,
        MegatronConfig,
    )

    config = MegatronBackendConfig(
        launcher_script="launch.sh",
        env={"ENV": "1"},
        extra_cli_args=["--extra"],
        lr=0.1,
        use_gpu=True,
        dist_cmd="",
        megatron=MegatronConfig(micro_batch_size=4),
    )
    backend = MegatronBackend(config)

    backend.validate()
    cmd = backend.build_launch_command()

    assert "launch.sh" in cmd
    assert "--micro-batch-size 4" in cmd
