from __future__ import annotations

from pathlib import Path

import pytest

from oellm_autoexp.config.loader import (
    ConfigLoaderError,
    ensure_registrations,
    load_config,
    load_hydra_config,
    load_monitoring_reference,
)

CONFIG_DIR = Path("config")
MONITORING_DIR = CONFIG_DIR / "monitoring"


MONITOR_FILES = sorted(MONITORING_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", MONITOR_FILES, ids=[p.stem for p in MONITOR_FILES])
def test_monitoring_configs_load(path: Path) -> None:
    ensure_registrations()
    root = load_hydra_config(
        "autoexp",
        CONFIG_DIR,
        overrides=[f"monitoring={path.stem}"],
    )
    assert root.monitoring.class_name


def test_monitor_config_with_log_signals_rejected(tmp_path: Path) -> None:
    ensure_registrations()
    bad = tmp_path / "invalid_monitor.yaml"
    bad.write_text(
        """
class_name: SlurmLogMonitor
log_signals:
  - name: legacy
    pattern: ERROR
    pattern_type: substring
    metadata:
      note: legacy
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoaderError, match="log_signals"):
        load_monitoring_reference(bad, CONFIG_DIR)


def test_root_config_with_policies_rejected(tmp_path: Path) -> None:
    ensure_registrations()
    bad = tmp_path / "invalid_root.yaml"
    bad.write_text(
        """
project:
  name: reject
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
  policies: []
backend:
  class_name: NullBackend
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoaderError, match="monitoring\\.policies"):
        load_config(bad)
