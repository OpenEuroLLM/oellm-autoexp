"""HF ``config.json`` of an engram_scaling run, from the run's resolved ``config-<job>.yaml``.

One HF layer per Megatron pattern symbol (``L`` Gated DeltaNet, ``E`` MoE, ``N`` Engram, ``*`` gated
attention), so HF layer ``i`` is ``decoder.layers.i`` and Megatron's ``layer_number`` is ``i + 1``.

The config carries three kinds of fields: sizes copied from Megatron; Megatron options translated to
what the HF forward needs (layer types, RoPE, router, norms); and values that are part of the model
but not of the checkpoint, recomputed here: the Engram table sizes and the per-layer hash multipliers
(Megatron derives both from the config and fixed seeds). Megatron options that change the forward and
that the HF model does not implement stop the build (:func:`check_supported`), so an export never
silently drops part of the model.

    python -m oellm_autoexp.hf_export.engram_scaling_clariden.hf_config \\
        --config <run>/config-<job>.yaml --ckpt-dir <run>/ckpt/iter_0001000 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..export_ambitious import engram_hash_multipliers, engram_table_sizes
from ..export_q35 import megatron_args, write_config

LOGGER = logging.getLogger(__name__)

MODEL_TYPE = "oellm_engram"
LAYER_TYPES = {"L": "gdn", "E": "moe", "N": "engram", "*": "attention"}

# Options the HF model implements only at this value.
_REQUIRED = {
    "is_hybrid_model": True,
    "experimental_attention_variant": "gated_delta_net",  # what 'L' builds
    "normalization": "RMSNorm",
    "swiglu": True,
    "add_bias_linear": False,
    "add_qkv_bias": False,
    "position_embedding_type": "rope",
    "softmax_type": "vanilla",
    "attention_length_scaling": "none",
    "linear_hybrid_attention": "standard",
    "linear_beta_activation": "sigmoid",
    "linear_beta_max": 1.0,
    "linear_drop_qkv_silu": False,
    "moe_router_score_function": "sigmoid",
    "moe_router_enable_expert_bias": True,
    "moe_router_pre_softmax": False,
    "embedding_multiplier": 1.0,
    "glu_linear_offset": 0.0,
}

# Features that change the forward pass and are not ported: each must be off (None / False / 0 / []).
_OFF = (
    # residual / depth schemes
    "mhc", "depth_memory", "attn_res", "ngpt", "loop_block_layers", "post_sublayer_norm",
    "apply_residual_connection_post_layernorm",
    # embedding / output
    "use_mup", "embedding_norm", "value_embedding", "per_layer_embedding_dim", "mtp_num_layers",
    # attention
    "multi_latent_attention", "exclusive_self_attention", "window_size", "window_attn_skip_freq",
    "qk_l2_norm", "use_rope_scaling", "rotary_seq_len_interpolation_factor", "page_sparse_selector",
    "dsa_indexer_topk", "linear_xsa", "linear_attention_freq",
    # MLP / MoE
    "activation_func_clamp_value", "squared_relu", "quick_geglu", "mlp_post_act_norm",
    "moe_expert_post_act_norm", "moe_router_num_groups", "moe_router_group_topk", "moe_router_force_biased",
    "moe_zero_expert_num", "moe_latent_size", "moe_apply_probs_on_input", "moe_mlp_glu_interleave_size",
    "moe_shared_expert_glu_interleave_size",
    # Engram
    "engram_sign_hash", "engram_aux_ngram_orders",
)


def block_types(pattern: str) -> list[str]:
    """Megatron's pattern (``|`` = pipeline stage boundary, no layer) -> one HF layer type per symbol."""
    if "/" in pattern:
        raise ValueError(f"MTP layers in {pattern!r} have no HF export")
    symbols = [c for c in pattern if c != "|"]
    bad = sorted({c for c in symbols if c not in LAYER_TYPES})
    if bad:
        raise ValueError(f"pattern symbols {bad} in {pattern!r} are not ported (supported: {sorted(LAYER_TYPES)})")
    return [LAYER_TYPES[c] for c in symbols]


def check_supported(m: dict) -> None:
    """Raise on every Megatron option of ``m`` that the HF model would not reproduce (all at once)."""
    problems = [f"{k}={m.get(k)!r} (expected {v!r})" for k, v in _REQUIRED.items() if m.get(k) != v]
    problems += [f"{k}={m[k]!r} (expected off)" for k in _OFF if m.get(k) not in (None, False, 0, 0.0, [], "None")]
    if m.get("moe_router_topk_scaling_factor") not in (None, 1, 1.0):
        problems.append(f"moe_router_topk_scaling_factor={m['moe_router_topk_scaling_factor']!r} (expected None / 1)")
    lb = m.get("moe_router_load_balancing_type")
    if "quantile_balancing" in (lb if isinstance(lb, (list, tuple)) else [lb]):
        problems.append("moe_router_load_balancing_type includes quantile_balancing (routes differently)")
    rot = int(m["kv_channels"] * float(m.get("rotary_percent") or 1.0))
    if rot % 2:
        problems.append(f"rotary dim {rot} is odd")
    if problems:
        raise ValueError("HF export does not support:\n  " + "\n  ".join(problems))


def hf_config(m: dict, vocab_size: int) -> dict:
    """The HF config of the Megatron arguments ``m`` (``backend.megatron`` of the resolved config)."""
    check_supported(m)
    types = block_types(m["hybrid_layer_pattern"])
    heads = m["num_attention_heads"]
    head_dim = m["kv_channels"]
    rotary_percent = float(m.get("rotary_percent") or 1.0)
    engram_layers = [i for i, t in enumerate(types) if t == "engram"]
    cfg = {
        "architectures": ["OellmEngramForCausalLM"],
        "model_type": MODEL_TYPE,
        "auto_map": {
            "AutoConfig": "configuration_oellm_engram.OellmEngramConfig",
            "AutoModelForCausalLM": "modeling_oellm_engram.OellmEngramForCausalLM",
        },
        "torch_dtype": "bfloat16",
        "vocab_size": vocab_size,
        "hidden_size": m["hidden_size"],
        "num_hidden_layers": len(types),
        # not "layer_types": transformers 5 validates that name against its own attention kinds
        "block_types": types,
        "rms_norm_eps": float(m["layernorm_epsilon"]),
        # backbone norms: stored w, applied as (1 + w). The Engram norms are plain (applied as w).
        "layernorm_zero_centered_gamma": bool(m.get("layernorm_zero_centered_gamma")),
        "tie_word_embeddings": not m.get("untie_embeddings_and_output_weights", False),
        # gated attention ('*')
        "num_attention_heads": heads,
        # Megatron uses num_query_groups only with --group-query-attention
        "num_key_value_heads": m["num_query_groups"] if m.get("group_query_attention") else heads,
        "head_dim": head_dim,
        "qk_layernorm": bool(m.get("qk_layernorm")),
        "attention_output_gate": bool(m.get("attention_output_gate")),
        # RoPE on the first rotary_dim channels of every head. The attention rotates interleaved pairs
        # (2i, 2i + 1) under rotary_interleaved, but HybridModel builds its RotaryEmbedding without
        # rotary_interleaved (megatron/core/models/hybrid/hybrid_model.py), so the per-channel angles
        # keep the non-interleaved layout [f0 .. fn, f0 .. fn]
        "rope_theta": float(m["rotary_base"]),
        "partial_rotary_factor": rotary_percent,
        "rotary_dim": int(head_dim * rotary_percent),
        "rope_interleaved": bool(m.get("rotary_interleaved")),
        "rope_interleaved_freqs": False,
        "max_position_embeddings": int(m.get("max_position_embeddings") or m["seq_length"]),
        # Gated DeltaNet ('L')
        "linear_num_key_heads": m["linear_num_key_heads"],
        "linear_num_value_heads": m["linear_num_value_heads"],
        "linear_key_head_dim": m["linear_key_head_dim"],
        "linear_value_head_dim": m["linear_value_head_dim"],
        "linear_conv_kernel_dim": m["linear_conv_kernel_dim"],
        # MoE ('E'): sigmoid scores in fp32, top-k chosen on scores + expert bias, weighted by the
        # unbiased scores normalised over the top-k (k > 1), times the scaling factor
        "num_experts": m["num_experts"],
        "num_experts_per_tok": m["moe_router_topk"],
        "moe_intermediate_size": m["moe_ffn_hidden_size"],
        "shared_expert_intermediate_size": m.get("moe_shared_expert_intermediate_size") or 0,
        "shared_expert_gate": bool(m.get("moe_shared_expert_gate")),
        "router_score_function": "sigmoid",
        "router_norm_topk_prob": m["moe_router_topk"] > 1,
        "router_scaling_factor": float(m.get("moe_router_topk_scaling_factor") or 1.0),
        "hidden_act": "silu",
    }
    if engram_layers:
        orders = list(m["engram_ngram_orders"])
        cfg.update({
            "engram_layers": engram_layers,
            "engram_ngram_orders": orders,
            "engram_num_heads": int(m["engram_num_heads"]),
            "engram_embed_dim": int(m["engram_embed_dim"]),
            "engram_conv_kernel_size": int(m["engram_conv_kernel_size"]),
            "engram_conv_dilation": max(orders),
            # one prime per (order, head) lookup; the same for every Engram layer
            "engram_table_sizes": engram_table_sizes(m),
            # {HF layer index: [lookups][max order]}, seeded with Megatron's layer_number = index + 1
            "engram_hash_multipliers": {str(i): engram_hash_multipliers(m, i + 1) for i in engram_layers},
            "engram_gate_signed_sqrt": bool(m.get("engram_gate_signed_sqrt")),
            "engram_conv_prenorm": bool(m.get("engram_conv_prenorm")),
            # the canonical-id map is a non-persistent Megatron buffer: the export stores it as a weight
            "engram_token_map": m.get("engram_token_map") not in (None, "None"),
        })
    return cfg


def vocab_from_checkpoint(ckpt_dir: Path) -> int:
    """Rows of the saved word embedding (Megatron's padded vocabulary), from the DCP metadata."""
    import torch.distributed.checkpoint as dcp

    md = dcp.FileSystemReader(str(ckpt_dir)).read_metadata().state_dict_metadata
    return int(md["embedding.word_embeddings.weight"].size[0])


def tokenizer_dir(name: str) -> Path:
    """A tokenizer directory, or the local snapshot of a hub id (the tokenizer repo has no config.json,
    which AutoTokenizer asks the hub for when given the id offline)."""
    if Path(name).is_dir():
        return Path(name)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, local_files_only=True))


def build(config_yaml: Path, ckpt_dir: Path, out_dir: Path, tokenizer: str | None = None) -> dict:
    """Write ``config.json`` and ``generation_config.json`` of the run to ``out_dir``; returns the config."""
    m = megatron_args(config_yaml)
    cfg = hf_config(m, vocab_from_checkpoint(ckpt_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(out_dir, cfg, tokenizer_dir(tokenizer or m["tokenizer_model"]))
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="the run's resolved config-<job>.yaml")
    p.add_argument("--ckpt-dir", type=Path, required=True, help="an iter_NNNNNNN directory (for the vocabulary size)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer", help="tokenizer dir or hub id (default: the run's tokenizer_model)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = build(a.config, a.ckpt_dir, a.out_dir, a.tokenizer)
    LOGGER.info("wrote %s: %d layers, Engram layers %s", a.out_dir / "config.json", cfg["num_hidden_layers"],
                cfg.get("engram_layers", []))


if __name__ == "__main__":
    main()
