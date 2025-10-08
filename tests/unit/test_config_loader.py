from pathlib import Path

import pytest

from oellm_autoexp.config.loader import load_config, load_config_reference
from oellm_autoexp.config.loader import ConfigLoaderError


def test_load_config(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "data" / "sample_config.yaml"
    cfg = load_config(config_path)

    assert cfg.project.name == "demo"
    assert cfg.project.base_output_dir == Path("./outputs")
    assert cfg.monitoring.class_name == "NullMonitor"
    assert cfg.backend.class_name == "NullBackend"
    assert cfg.slurm.launcher_cmd == ""
    assert cfg.slurm.srun_opts == ""
    assert cfg.slurm.srun_opts == ""
    assert cfg.slurm.client.class_name == "FakeSlurmClient"
    assert cfg.restart_policies[0].mode == "success"
    assert cfg.project.state_dir == Path("./outputs") / ".oellm-autoexp"


def test_load_hydra_config(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ACCOUNT", "debug")
    monkeypatch.setenv("CONTAINER_CACHE_DIR", "debug")
    cfg = load_config_reference("autoexp", Path("config"), overrides=["project=juwels", "slurm=juwels"])

    assert cfg.project.name == "juwels"
    assert "JUWELS" in cfg.slurm.env.get("MACHINE_NAME", "")
    assert str(cfg.slurm.template_path).endswith("juwels.sbatch")
    assert cfg.slurm.launcher_cmd.startswith("{{env_exports}}")
    assert cfg.slurm.srun_opts == ""


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoaderError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_reference_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoaderError):
        load_config_reference("autoexp", tmp_path / "missing")


def test_load_hydra_config_restart(tmp_path: Path) -> None:
    config_dir = tmp_path / "hydra"
    config_dir.mkdir()
    (config_dir / "autoexp.yaml").write_text(
        """
defaults:
  - _self_

project:
  name: hydra
  base_output_dir: ./outputs
sweep:
  axes: {}
slurm:
  template_path: ./templates/train.sbatch
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
restart:
  policies:
    - mode: success
      implementation:
        class_name: NoRestartPolicy
scheduler:
  max_jobs: null
  submit_rate_limit_seconds: null
metadata: {}
"""
    )

    cfg = load_config_reference("autoexp", config_dir)
    assert cfg.restart_policies[0].mode == "success"
