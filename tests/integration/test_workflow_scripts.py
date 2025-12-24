from __future__ import annotations

from pathlib import Path

import pytest

from scripts import monitor_autoexp, plan_autoexp, submit_autoexp


CONFIG_TEMPLATE = """
project:
  name: integration
  base_output_dir: ./outputs
  monitoring_state_dir: ./monitor
sweep:
  grids:
    - backend.megatron.lr: [0.1]
slurm:
  template_path: {template}
  script_dir: ./scripts
  log_dir: ./logs
  launcher_cmd: ""
  srun_opts: ""
  client:
    class_name: FakeSlurmClient
monitoring:
  class_name: NullMonitor
backend:
  class_name: NullBackend
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    template = tmp_path / "template.sbatch"
    template.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(CONFIG_TEMPLATE.format(template=template))
    return cfg_path


def test_plan_submit_monitor_fake_slurm(
    tmp_path: Path, monkeypatch, config_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_ACCOUNT", "integration")

    plan_autoexp.main(["--config-ref", str(config_path)])

    manifest_dir = tmp_path / "monitor" / "manifests"
    manifests = sorted(manifest_dir.glob("plan_*.json"))
    assert len(manifests) == 1
    manifest = manifests[0]

    submit_autoexp.main(
        [
            "--manifest",
            str(manifest),
            "--use-fake-slurm",
            "--dry-run",
        ]
    )

    monitor_autoexp.main(["--manifest", str(manifest), "--use-fake-slurm"])
    captured = capsys.readouterr()
    assert "No jobs registered" in captured.out


def test_multiple_runs_store_manifests(tmp_path: Path, monkeypatch, config_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_ACCOUNT", "integration")

    plan_autoexp.main(["--config-ref", str(config_path)])
    plan_autoexp.main(["--config-ref", str(config_path)])

    manifest_dir = tmp_path / "monitor" / "manifests"
    manifests = sorted(manifest_dir.glob("plan_*.json"))
    assert len(manifests) == 2
    assert len({manifest.name for manifest in manifests}) == 2
