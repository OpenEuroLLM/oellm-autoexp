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
        },
        "monitoring": {"implementation": {"class_name": "NullMonitor"}},
        "backend": {"implementation": {"class_name": "NullBackend"}},
        "restart_policies": [
            {"mode": "success", "implementation": {"class_name": "NoRestartPolicy"}}
        ],
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_cli_plan(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["plan", str(config)])
    assert result.exit_code == 0
    assert "\"project\": \"demo\"" in result.output


def test_cli_submit_fake(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["submit", str(config), "--fake"])
    assert result.exit_code == 0
    assert "submitted" in result.output


def test_cli_submit_real_not_implemented(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["submit", str(config)])
    assert result.exit_code != 0
    assert "Real SLURM submission not implemented yet" in result.output
