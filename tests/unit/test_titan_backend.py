from __future__ import annotations

import types
import sys


def test_titan_backend_builds_command(tmp_path):
    from oellm_autoexp.backends.titan_backend import TitanBackend, TitanBackendConfig
    from oellm_autoexp.backends.titan.config_schema import JobConfig as TitanJobConfig

    config = TitanBackendConfig(
        titan=TitanJobConfig(),
        toml_output_path=str(tmp_path / "config.titan.toml"),
        torchrun_args={"nproc_per_node": 1},
    )

    backend = TitanBackend(config)
    cmd = backend.build_launch_command()

    assert "torchrun" in cmd
    assert "--job.config_file=" in cmd


def test_titan_backend_applies_cluster_args(tmp_path, monkeypatch):
    from oellm_autoexp.backends.titan_backend import TitanBackend, TitanBackendConfig
    from oellm_autoexp.backends.titan.config_schema import JobConfig as TitanJobConfig

    fake_cluster = types.SimpleNamespace(
        get_cli_args=lambda **kwargs: "--model.flavor=1B --training.steps=123",
    )
    monkeypatch.setitem(sys.modules, "titan_oellm.cluster_config", fake_cluster)

    config = TitanBackendConfig(
        titan=TitanJobConfig(),
        toml_output_path=str(tmp_path / "config.titan.toml"),
        cluster_args_source="titan_oellm",
    )

    backend = TitanBackend(config)
    backend.build_launch_command()

    text = (tmp_path / "config.titan.toml").read_text()
    assert "[model]" in text
    assert "flavor = '1B'" in text or "flavor = \"1B\"" in text
    assert "[training]" in text
    assert "steps = 123" in text
