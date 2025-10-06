from pathlib import Path

import argparse

from scripts import run_megatron_container


def test_run_megatron_container_invokes_fake_submit(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(args, cmd_parts):
        calls.append(cmd_parts)
        return 0

    monkeypatch.setattr(run_megatron_container, "_run_in_container", fake_run)

    args = argparse.Namespace(
        image="image.sif",
        apptainer_cmd="apptainer",
        config_ref="autoexp",
        config_dir=str(tmp_path),
        override=["project=demo"],
        dry_run=False,
        fake_submit=True,
        show_command=False,
        env=[],
        bind=[],
        no_submit=False,
    )
    monkeypatch.setattr(run_megatron_container, "parse_args", lambda: args)

    run_megatron_container.main()

    assert len(calls) == 2
    assert calls[0][-1] == "--dry-run"
    assert calls[1][-1] == "--use-fake-slurm"


def test_build_container_command_includes_env_and_bind():
    args = argparse.Namespace(
        apptainer_cmd="singularity",
        image="container.sif",
        env=["FOO=1", "BAR=2"],
        bind=["/data:/data"],
    )
    cmd = run_megatron_container._build_container_command(args, ["python", "script.py"])
    assert "--bind" in cmd
    assert "--env" in cmd
    assert cmd[-1].endswith("script.py")
