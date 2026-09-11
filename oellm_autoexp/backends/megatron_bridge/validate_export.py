"""Validate the structure of a Qwen3 Hugging Face safetensors export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _expected_qwen3_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    vocab = int(config["vocab_size"])

    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
    }
    if not config.get("tie_word_embeddings", False):
        expected["lm_head.weight"] = (vocab, hidden)

    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (heads * head_dim, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, heads * head_dim),
                f"{prefix}.self_attn.q_norm.weight": (head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return expected


def validate_qwen3_export(hf_path: Path) -> dict[str, int]:
    """Check required Qwen3 tensor names/shapes without materializing weights."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to validate the HF export") from exc

    hf_path = Path(hf_path)
    config_path = hf_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hugging Face config: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError(f"Expected model_type=qwen3, got {config.get('model_type')!r}")

    shard_paths = sorted(hf_path.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No safetensors files found under {hf_path}")

    actual: dict[str, tuple[int, ...]] = {}
    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in actual:
                    raise ValueError(f"Duplicate tensor {key!r} in HF export")
                actual[key] = tuple(shard.get_slice(key).get_shape())

    expected = _expected_qwen3_shapes(config)
    missing = sorted(set(expected) - set(actual))
    mismatched = {
        key: (expected[key], actual[key])
        for key in expected.keys() & actual.keys()
        if expected[key] != actual[key]
    }
    if missing or mismatched:
        details: list[str] = []
        if missing:
            details.append(f"missing tensors ({len(missing)}): {missing[:20]}")
        if mismatched:
            sample = list(sorted(mismatched.items()))[:20]
            details.append(f"shape mismatches ({len(mismatched)}): {sample}")
        raise ValueError("Invalid Qwen3 HF export: " + "; ".join(details))

    return {
        "safetensor_shards": len(shard_paths),
        "tensors": len(actual),
        "required_tensors": len(expected),
        "layers": int(config["num_hidden_layers"]),
        "vocab_size": int(config["vocab_size"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_path", type=Path)
    args = parser.parse_args()
    summary = validate_qwen3_export(args.hf_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
