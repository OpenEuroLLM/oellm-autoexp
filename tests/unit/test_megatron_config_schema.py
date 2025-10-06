from pathlib import Path

import pytest
from omegaconf import OmegaConf

from compoconf import parse_config

from oellm_autoexp.backends.megatron.config_schema import MegatronConfig


def test_parse_megatron_defaults():
    cfg_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "backend"
        / "megatron"
        / "base_defaults.yaml"
    )
    data = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    cfg = parse_config(MegatronConfig, data)
    assert cfg.hidden_size is None
    assert cfg.attention_backend == 5


def test_invalid_literal_rejected():
    with pytest.raises(Exception):
        parse_config(MegatronConfig, {"attention_backend": 999})
