from __future__ import annotations

import pytest

from oellm_autoexp.backends.megatron_bridge.hf_config_gen import (
    derive_hf_config,
    derive_qwen3_hf_config,
)


def _qwen3_125M_megatron() -> dict:
    return {
        "num_layers": 16,
        "hidden_size": 512,
        "ffn_hidden_size": 2048,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "kv_channels": 128,
        "seq_length": 4096,
        "norm_epsilon": 1e-6,
        "rotary_base": 1000000,
        "untie_embeddings_and_output_weights": False,
        "bf16": True,
        "add_qkv_bias": False,
    }


def test_qwen3_basic_mapping():
    cfg = derive_qwen3_hf_config(_qwen3_125M_megatron(), vocab_size=151936)
    assert cfg["model_type"] == "qwen3"
    assert cfg["architectures"] == ["Qwen3ForCausalLM"]
    assert cfg["num_hidden_layers"] == 16
    assert cfg["hidden_size"] == 512
    assert cfg["intermediate_size"] == 2048
    assert cfg["num_attention_heads"] == 4
    assert cfg["num_key_value_heads"] == 4
    assert cfg["head_dim"] == 128
    assert cfg["max_position_embeddings"] == 4096
    assert cfg["rope_theta"] == 1000000
    assert cfg["tie_word_embeddings"] is True
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["vocab_size"] == 151936


def test_qwen3_untie_flag_inverts_tie():
    m = _qwen3_125M_megatron()
    m["untie_embeddings_and_output_weights"] = True
    assert derive_qwen3_hf_config(m, 100)["tie_word_embeddings"] is False


def test_qwen3_head_dim_defaults_from_hidden_and_heads():
    m = _qwen3_125M_megatron()
    m["kv_channels"] = None
    cfg = derive_qwen3_hf_config(m, 100)
    assert cfg["head_dim"] == 512 // 4


def test_dispatch_via_derive_hf_config():
    cfg = derive_hf_config("qwen3", _qwen3_125M_megatron(), vocab_size=42)
    assert cfg["vocab_size"] == 42


def test_dispatch_unknown_arch():
    with pytest.raises(ValueError, match="No HF config deriver"):
        derive_hf_config("llama", {}, 0)
