import pytest
import json
import textwrap
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch
from hydra_staged_sweep.config.loader import (
    load_config,
    load_hydra_config,
    load_config_reference,
    ConfigLoaderError,
)
from hydra_staged_sweep.dag_resolver import param_to_cmdlines
from hydra_staged_sweep.config.schema import StagedSweepRoot


@dataclass(kw_only=True)
class MockConfig(StagedSweepRoot):
    backend: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def test_load_config_missing_file():
    with pytest.raises(ConfigLoaderError, match="Configuration file not found"):
        load_config("nonexistent.yaml")


def test_load_config_not_mapping(tmp_path):
    p = tmp_path / "test.yaml"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ConfigLoaderError, match="Configuration root must be a mapping"):
        load_config(p)


def test_load_hydra_config_missing_dir():
    with pytest.raises(ConfigLoaderError, match="Hydra config directory not found"):
        load_hydra_config("base", "/nonexistent")


def test_load_config_reference_with_json(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("stage: stable")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = run_dir / "config.yaml"
    config_path.write_text("stage: override")

    ref_path = run_dir / "config_reference.json"
    ref_path.write_text(
        json.dumps(
            {
                "config_ref": "base",
                "config_dir": str(config_dir),
                "overrides": ["stage=json_override"],
            }
        )
    )

    # Should use the JSON reference
    res = load_config_reference(
        config_path=config_path,
        config_dir=config_dir,
        overrides=["++index=1"],
        config_class=MockConfig,
    )
    assert res.stage == "json_override"
    assert res.index == 1


def test_load_config_reference_no_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("stage: direct")

    res = load_config_reference(
        config_path=config_path, config_dir=tmp_path, config_class=MockConfig
    )
    assert res.stage == "direct"


def test_load_hydra_config_with_interpolation_override(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("stage: ${oc.env:STAGE,unknown}")

    res = load_hydra_config(
        "base", config_dir, overrides=["stage=${oc.env:STAGE,val}"], config_class=MockConfig
    )
    assert res.stage == "val"


def test_load_hydra_config_multiple_defaults_merge(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        textwrap.dedent(
            """
            defaults:
              - setup: setup1
              - _self_
            """
        ).strip()
        + "\n"
    )
    setup_dir = config_dir / "setup"
    setup_dir.mkdir()
    (setup_dir / "setup1.yaml").write_text("mode: first\nalpha: 1\n")
    (setup_dir / "setup2.yaml").write_text("mode: second\nbeta: 2\n")

    @dataclass(kw_only=True)
    class DefaultsConfig(StagedSweepRoot):
        setup: dict[str, Any] = field(default_factory=dict)

    res = load_hydra_config(
        "base", config_dir, overrides=["setup=[setup1,setup2]"], config_class=DefaultsConfig
    )
    assert res.setup["mode"] == "second"
    assert res.setup["alpha"] == 1
    assert res.setup["beta"] == 2

    res = load_hydra_config(
        "base",
        config_dir,
        overrides=param_to_cmdlines("setup", "[setup1,setup2]"),
        config_class=DefaultsConfig,
    )
    assert res.setup["mode"] == "second"
    assert res.setup["alpha"] == 1
    assert res.setup["beta"] == 2


def test_load_config_reference_not_mapping(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_path = config_dir / "base.yaml"
    config_path.write_text("[1, 2]")

    from hydra.errors import ConfigCompositionException

    with pytest.raises((ConfigLoaderError, ConfigCompositionException)):
        load_config_reference(config_path=config_path, config_dir=tmp_path, overrides=["++a=b"])


def test_load_hydra_config_not_mapping_error(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("foo: bar")

    with patch("hydra.compose") as mock_compose:
        mock_compose.return_value = MagicMock()
        with patch("omegaconf.OmegaConf.to_container", return_value=[1, 2]):
            with pytest.raises(
                ConfigLoaderError, match="Hydra config base did not produce a mapping"
            ):
                load_hydra_config("base", config_dir)


def test_load_config_reference_not_mapping_v2(tmp_path):
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_path = config_dir / "ref.yaml"
    config_path.write_text("stage: ref")

    with patch("hydra.compose") as mock_compose:
        mock_compose.return_value = MagicMock()
        with patch("omegaconf.OmegaConf.to_container", return_value=[1, 2]):
            with pytest.raises(ConfigLoaderError, match="Config file .* did not produce a mapping"):
                load_config_reference(
                    config_path=config_path, config_dir=tmp_path, overrides=["++a=b"]
                )
