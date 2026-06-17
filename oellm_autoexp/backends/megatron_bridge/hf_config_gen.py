"""Derive a HuggingFace ``config.json`` from a Megatron training config.

The Megatron-Bridge conversion path needs a *reference* HF config dir (passed
as ``--hf-model`` to ``convert_checkpoints.py``) so it can build a dummy model
and match weight shapes. When the Megatron training arch doesn't have a
matching upstream HF release (e.g. a custom Qwen3-sized variant), vendoring a
fixed ``config.json`` is fragile — any time someone tweaks the Megatron yaml,
the vendored HF config gets out of sync.

This module synthesises the HF config from the resolved Megatron config dict
at convert time. The mapping is architecture-specific (one ``_derive_*``
function per supported HF model family). Add more architectures by writing a
new ``_derive_*`` and registering it under ``DERIVERS``.

Used by ``oellm_autoexp.backends.megatron_bridge.run_export``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


def _g(megatron: dict, *keys, default=None):
    """Get the first non-None value among ``keys`` from ``megatron``."""
    for k in keys:
        if k in megatron and megatron[k] is not None:
            return megatron[k]
    return default


def _padded_vocab(raw_vocab: int, megatron: dict) -> int:
    """Apply Megatron's vocab-size padding to match the trained embedding row
    count."""
    pad = _g(megatron, "make_vocab_size_divisible_by")
    if pad and pad > 0:
        # Round UP to the next multiple of `pad`.
        return ((int(raw_vocab) + int(pad) - 1) // int(pad)) * int(pad)
    return int(raw_vocab)


def derive_qwen3_hf_config(megatron: dict, vocab_size: int) -> dict[str, Any]:
    """Build a Qwen3-shaped HF config.json from a Megatron config dict."""
    hidden_size = _g(megatron, "hidden_size")
    num_layers = _g(megatron, "num_layers")
    num_attention_heads = _g(megatron, "num_attention_heads")
    num_query_groups = _g(megatron, "num_query_groups", default=num_attention_heads)
    kv_channels = _g(megatron, "kv_channels")
    if kv_channels is None and hidden_size and num_attention_heads:
        kv_channels = hidden_size // num_attention_heads
    ffn_hidden_size = _g(megatron, "ffn_hidden_size")
    seq_length = _g(megatron, "max_position_embeddings", "seq_length")
    rms_norm_eps = _g(megatron, "norm_epsilon", "layernorm_epsilon", default=1e-6)
    rope_theta = _g(megatron, "rotary_base", "rope_theta", default=10000)
    tie_embeddings = not bool(_g(megatron, "untie_embeddings_and_output_weights", default=False))
    dtype = _g(megatron, "params_dtype", default=None)
    if dtype is None:
        dtype = (
            "bfloat16"
            if _g(megatron, "bf16")
            else ("float16" if _g(megatron, "fp16") else "float32")
        )

    config: dict[str, Any] = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "attention_bias": bool(_g(megatron, "add_qkv_bias", default=False)),
        "attention_dropout": float(_g(megatron, "attention_dropout", default=0.0)),
        "hidden_act": _g(megatron, "activation_func", "hidden_act", default="silu"),
        "hidden_size": hidden_size,
        "initializer_range": float(_g(megatron, "init_method_std", default=0.02)),
        "intermediate_size": ffn_hidden_size,
        "max_position_embeddings": seq_length,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": num_query_groups,
        "head_dim": kv_channels,
        "rms_norm_eps": float(rms_norm_eps),
        "rope_scaling": None,
        "rope_theta": int(rope_theta),
        "tie_word_embeddings": tie_embeddings,
        "torch_dtype": dtype,
        "use_cache": True,
        # Megatron pads the embedding rows up to a multiple of
        # `make_vocab_size_divisible_by` (default 128). The trained
        # checkpoint has the padded size, so the HF config must report it
        # to satisfy transformers' shape check on load.
        "vocab_size": _padded_vocab(vocab_size, megatron),
        "sliding_window": None,
        "use_sliding_window": False,
        "max_window_layers": num_layers,
    }
    return config


DERIVERS: dict[str, Callable[[dict, int], dict[str, Any]]] = {
    "qwen3": derive_qwen3_hf_config,
}


def derive_hf_config(arch: str, megatron: dict, vocab_size: int) -> dict[str, Any]:
    """Dispatch to the right per-architecture deriver."""
    deriver = DERIVERS.get(arch)
    if deriver is None:
        raise ValueError(
            f"No HF config deriver for arch={arch!r}. "
            f"Available: {sorted(DERIVERS)}. Add one in hf_config_gen.py."
        )
    return deriver(megatron, vocab_size)


def write_hf_config_dir(
    arch: str,
    megatron: dict,
    vocab_size: int,
    tokenizer_src: Path | str | None,
    outdir: Path,
) -> Path:
    """Write a directory shaped like a HF model snapshot (``config.json`` +
    tokenizer files copied from ``tokenizer_src`` if given).

    Returns the directory path so the caller can pass it as ``--hf-model``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = derive_hf_config(arch, megatron, vocab_size)
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    LOGGER.info("Wrote derived HF config to %s", outdir / "config.json")
    if tokenizer_src is not None:
        import shutil

        tokenizer_src = Path(tokenizer_src)
        if tokenizer_src.is_dir():
            for f in tokenizer_src.iterdir():
                if f.is_file():
                    shutil.copy(f, outdir / f.name)
            LOGGER.info("Copied tokenizer files from %s", tokenizer_src)
    return outdir


__all__ = [
    "derive_hf_config",
    "derive_qwen3_hf_config",
    "write_hf_config_dir",
    "DERIVERS",
]
