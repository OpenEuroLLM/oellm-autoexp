from __future__ import annotations

from pathlib import Path

import pytest

from scripts import plan_autoexp


MINIMAL_CONFIG = """
project:
  name: unit
  base_output_dir: ./outputs
  monitoring_state_dir: ./monitor
sweep:
  axes:
    only: [1]
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
def config_file(tmp_path: Path) -> Path:
    template = tmp_path / "template.sbatch"
    template.write_text("#!/bin/bash\n{sbatch_directives}\n\nsrun {srun_opts}{launcher_cmd}\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MINIMAL_CONFIG.format(template=template))
    return config_path


def test_plan_creates_timestamped_manifest(tmp_path: Path, monkeypatch, config_file: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_ACCOUNT", "unit")

    plan_autoexp.main(["--config-ref", str(config_file)])

    manifest_dir = tmp_path / "monitor" / "manifests"
    manifests = sorted(manifest_dir.glob("plan_*.json"))
    assert len(manifests) == 1
    assert manifests[0].exists()


def test_plan_multiple_runs_produce_unique_manifests(
    tmp_path: Path, monkeypatch, config_file: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_ACCOUNT", "unit")

    plan_autoexp.main(["--config-ref", str(config_file)])
    plan_autoexp.main(["--config-ref", str(config_file)])

    manifest_dir = tmp_path / "monitor" / "manifests"
    manifests = sorted(manifest_dir.glob("plan_*.json"))
    assert len(manifests) == 2
    names = {manifest.name for manifest in manifests}
    assert len(names) == 2
