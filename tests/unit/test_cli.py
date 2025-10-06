from pathlib import Path

import yaml
from click.testing import CliRunner

from oellm_autoexp.cli import cli


def _write_config(tmp_path: Path) -> Path:
    template = tmp_path / "template.sbatch"
    template.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")

    config = {
        "project": {"name": "demo", "base_output_dir": str(tmp_path / "outputs")},
        "sweep": {"axes": {}},
        "slurm": {
            "template_path": str(template),
            "script_dir": str(tmp_path / "scripts"),
            "log_dir": str(tmp_path / "logs"),
            "launcher_cmd": "module load foo &&",
            "srun_opts": "--ntasks=1",
            "client": {"class_name": "FakeSlurmClient"},
        },
        "monitoring": {"implementation": {"class_name": "NullMonitor"}},
        "backend": {"implementation": {"class_name": "NullBackend"}},
        "restart_policies": [{"mode": "success", "implementation": {"class_name": "NoRestartPolicy"}}],
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_cli_plan(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["plan", str(config)])
    assert result.exit_code == 0
    assert '"project": "demo"' in result.output


def test_cli_submit_fake(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["submit", str(config), "--fake"])
    assert result.exit_code == 0
    assert "submitted" in result.output
    assert "log:" in result.output


def test_cli_submit_real_slurm_client(tmp_path: Path, monkeypatch) -> None:
    """Test that SlurmClient can be used (though we mock the actual sbatch call)."""
    config = _write_config(tmp_path)
    data = yaml.safe_load(config.read_text())
    data["slurm"]["client"]["class_name"] = "SlurmClient"
    config.write_text(yaml.safe_dump(data))

    # Mock subprocess.run to avoid needing real sbatch
    import subprocess

    class MockResult:
        returncode = 0
        stdout = "Submitted batch job 12345"
        stderr = ""

    def mock_subprocess_run(cmd, **kwargs):
        return MockResult()

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    runner = CliRunner()
    result = runner.invoke(cli, ["submit", str(config)])
    assert result.exit_code == 0
    assert "submitted" in result.output
    assert "log:" in result.output


def test_cli_monitor_emits_checkpoint_action(tmp_path: Path) -> None:
    monitor_cfg = tmp_path / "monitor.yaml"
    monitor_cfg.write_text("class_name: SlurmLogMonitor\ncheck_interval_seconds: 1\n")
    log_path = tmp_path / "job.log"
    log_path.write_text("Checkpoint saved: /tmp/ckpt.bin\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "monitor",
            "--monitor-config",
            str(monitor_cfg),
            "--log",
            f"run={log_path}",
            "--dry-run",
            "--action",
            "new_checkpoint=echo {checkpoint_path}",
        ],
    )
    assert result.exit_code == 0
    assert "new_checkpoint" in result.output
    assert "DRY-RUN command: echo /tmp/ckpt.bin" in result.output


def test_cli_monitor_reports_slurm_transitions(tmp_path: Path) -> None:
    monitor_cfg = tmp_path / "monitor.yaml"
    monitor_cfg.write_text("class_name: SlurmLogMonitor\ncheck_interval_seconds: 1\n")
    log_path = tmp_path / "job.log"
    log_path.write_text("initial\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "monitor",
            "--monitor-config",
            str(monitor_cfg),
            "--log",
            f"run={log_path}",
            "--slurm-state",
            "run=RUNNING",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "run_started" in result.output
