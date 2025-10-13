from pathlib import Path

import argparse

from scripts import run_megatron_container


def test_run_megatron_container_with_fake_submit(monkeypatch, tmp_path: Path):
    """Test that fake submit runs inside container and doesn't execute sbatch on host."""

    def fake_subprocess_run(cmd, **kwargs):
        # Simulate container returning output without sbatch command (fake SLURM handles internally)
        result = argparse.Namespace()
        result.returncode = 0
        result.stdout = "Using fake SLURM client\nJob submitted internally"
        result.stderr = ""
        return result

    monkeypatch.setattr(run_megatron_container, "run_with_tee", fake_subprocess_run)

    args = argparse.Namespace(
        image="image.sif",
        apptainer_cmd="apptainer",
        config_ref="autoexp",
        config_dir=str(tmp_path),
        override=["project=demo"],
        no_run=False,
        fake_submit=True,
        env=[],
        bind=[],
        ihelp=False,
    )
    monkeypatch.setattr(run_megatron_container, "parse_args", lambda: args)

    # Should not raise, and should handle fake SLURM gracefully
    run_megatron_container.main()


def test_run_megatron_container_parses_sbatch_command(monkeypatch, tmp_path: Path):
    """Test that sbatch command is parsed from container output and executed on host."""

    calls = []

    def fake_subprocess_run(cmd, **kwargs):
        result = argparse.Namespace()
        result.returncode = 0

        # First call: container generates scripts and outputs sbatch command
        if "bash" in cmd and "-lc" in cmd:
            result.stdout = "Successful, to execute, run: sbatch /path/to/script.sbatch"
            result.stderr = ""
        # Second call: sbatch execution on host
        elif "sbatch" in cmd:
            calls.append(("sbatch", cmd))
            result.stdout = "Submitted batch job 12345"
            result.stderr = ""
        else:
            result.stdout = ""
            result.stderr = ""

        return result

    monkeypatch.setattr(run_megatron_container, "run_with_tee", fake_subprocess_run)

    args = argparse.Namespace(
        image="image.sif",
        apptainer_cmd="apptainer",
        config_ref="autoexp",
        config_dir=str(tmp_path),
        override=["project=demo"],
        no_run=False,
        fake_submit=False,
        env=[],
        bind=[],
        ihelp=False,
    )
    monkeypatch.setattr(run_megatron_container, "parse_args", lambda: args)

    run_megatron_container.main()

    # Verify sbatch was called on the host
    assert len(calls) == 1
    assert calls[0][0] == "sbatch"
    assert "sbatch" in calls[0][1]


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
