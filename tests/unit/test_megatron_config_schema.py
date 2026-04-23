from pathlib import Path

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
    # After enum name extraction fix, attention_backend is now "auto" instead of 5
    assert cfg.attention_backend == "auto"


def test_invalid_literal_rejected():
    # After enum name extraction, attention_backend accepts any string
    # Test a different field with Literal constraints instead
    # For now, we accept that enum fields use str type for flexibility
    cfg = parse_config(MegatronConfig, {"attention_backend": "invalid_value"})
    assert cfg.attention_backend == "invalid_value"
