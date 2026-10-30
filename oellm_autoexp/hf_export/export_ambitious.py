"""Export the ambitious hybrid main run (Megatron HybridModel: CKDA + gated NoPE XSA attention,
mHC-4, MoE, Engram; ``fsdp_dtensor`` checkpoints) as a HF model (``oellm_ambitious/``,
trust_remote_code). One HF layer per Megatron pattern symbol; see the modeling file.

    python -m oellm_autoexp.hf_export.export_ambitious --ckpt-dir <run>/ckpt_eval/iter_0000500 \\
        --config <run>/config-<job>.yaml --out-dir <exports>/ambitious_500 --tokenizer <tok>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from .convert_megatron_to_hf import _fla_layer_spec
from .export_q35 import megatron_args, split_gated_qkv, write_config
from .fsdp_dtensor import ShardedSafetensorsWriter, iter_layers, stack_experts

LOGGER = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent
_PKG = _HERE / "oellm_ambitious"
_KINDS = {"*": "attention", "L": "ckda", "E": "moe", "N": "engram"}
_MHC = ("phi", "b_pre", "b_post", "b_res", "alpha_pre", "alpha_post", "alpha_res")

# megatron/core/ssm/engram.py
_HASH_SEED = 0x5EED_E46A


def _is_prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def engram_table_sizes(m: dict) -> list[int]:
    sizes, c = [], int(m["engram_table_size"])
    for _ in range(len(m["engram_ngram_orders"]) * int(m["engram_num_heads"])):
        while not _is_prime(c):
            c += 1
        sizes.append(c)
        c += 1
    return sizes


def engram_hash_multipliers(m: dict, layer_number: int) -> list[list[int]]:
    seed = _HASH_SEED + (10007 * int(layer_number) if m.get("engram_hash_per_layer") else 0)
    rng = np.random.default_rng(seed)
    n = len(m["engram_ngram_orders"]) * int(m["engram_num_heads"])
    mult = rng.integers(1 << 20, 1 << 31, size=(n, max(m["engram_ngram_orders"])))
    return (mult | 1).astype(np.int64).tolist()


def hf_config(m: dict, vocab: int) -> dict:
    syms = [c for c in m["hybrid_layer_pattern"] if c != "|"]
    bad = sorted({c for c in syms if c not in _KINDS})
    if bad:
        raise ValueError(f"unsupported pattern symbols {bad}")
    if not m.get("mhc") or m.get("mhc_routed_topk") or m.get("mhc_internal_attention", "none") != "none" or m.get("mhc_lerp"):
        raise ValueError("expects plain mHC (no routed / internal-attention / lerp variants)")
    if m.get("moe_router_score_function") != "sigmoid" or not m.get("moe_router_enable_expert_bias"):
        raise ValueError("expects the sigmoid router with expert bias")
    if m.get("moe_shared_expert_gate"):
        raise ValueError("gated shared expert not ported")
    if m.get("position_embedding_type") not in (None, "none"):
        raise ValueError("expects NoPE attention")
    if m.get("linear_xsa"):
        raise ValueError("linear XSA in CKDA not ported")
    heads = m["num_attention_heads"]
    types = [_KINDS[c] for c in syms]
    engram_layers = [i for i, t in enumerate(types) if t == "engram"]
    if engram_layers and m.get("engram_sign_hash"):  # Megatron engram_sign_hash: not in HF yet
        raise NotImplementedError("HF export of engram_sign_hash models is not implemented")
    return {
        "architectures": ["OellmAmbitiousForCausalLM"],
        "model_type": "oellm_ambitious",
        "auto_map": {
            "AutoConfig": "configuration_oellm_ambitious.OellmAmbitiousConfig",
            "AutoModelForCausalLM": "modeling_oellm_ambitious.OellmAmbitiousForCausalLM",
        },
        "vocab_size": vocab,
        "hidden_size": m["hidden_size"],
        "block_types": types,
        "rms_norm_eps": float(m["layernorm_epsilon"]),
        "layernorm_zero_centered_gamma": bool(m.get("layernorm_zero_centered_gamma", False)),
        "mhc_expansion_rate": int(m["mhc_expansion_rate"]),
        "mhc_sinkhorn_iters": int(m.get("mhc_sinkhorn_iters", 20)),
        "mhc_eps": float(m["layernorm_epsilon"]),
        "num_attention_heads": heads,
        # Megatron applies num_query_groups only with --group-query-attention
        "num_key_value_heads": m["num_query_groups"] if m.get("group_query_attention") else heads,
        "head_dim": m["kv_channels"],
        "qk_layernorm": bool(m.get("qk_layernorm")),
        "attention_output_gate": bool(m.get("attention_output_gate")),
        "exclusive_self_attention": bool(m.get("exclusive_self_attention")),
        "xsa_eps": 1e-6,
        "fla_layer": _fla_layer_spec(m),
        "num_experts": m["num_experts"],
        "num_experts_per_tok": m["moe_router_topk"],
        "moe_intermediate_size": m["moe_ffn_hidden_size"],
        "shared_expert_intermediate_size": m.get("moe_shared_expert_intermediate_size") or 0,
        "router_scaling_factor": float(m.get("moe_router_topk_scaling_factor") or 1.0),
        "activation_clamp_value": m.get("activation_func_clamp_value"),
        "engram_ngram_orders": list(m["engram_ngram_orders"]),
        "engram_num_heads": int(m["engram_num_heads"]),
        "engram_embed_dim": int(m["engram_embed_dim"]),
        "engram_conv_kernel_size": int(m.get("engram_conv_kernel_size", 4)),
        "engram_table_sizes": engram_table_sizes(m) if engram_layers else None,
        # HybridModel layer_number = pattern index + 1
        "engram_hash_multipliers": {str(i): engram_hash_multipliers(m, i + 1) for i in engram_layers},
        "engram_token_map": bool(m.get("engram_token_map")),
        "engram_gate_signed_sqrt": bool(m.get("engram_gate_signed_sqrt")),
        "engram_conv_prenorm": bool(m.get("engram_conv_prenorm")),
        "tie_word_embeddings": not m.get("untie_embeddings_and_output_weights", False),
        "torch_dtype": "bfloat16",
    }


def convert(ckpt_dir: Path, config_yaml: Path, out_dir: Path, tokenizer: Path | None, dtype: str = "bf16",
            engram_table_dtype: str = "bf16") -> Path:
    m = megatron_args(config_yaml)
    cast = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    tcast = {"bf16": torch.bfloat16, "fp32": torch.float32}[engram_table_dtype]
    out_dir = Path(out_dir)
    writer = ShardedSafetensorsWriter(out_dir)
    cfg = None
    for i, t in iter_layers(Path(ckpt_dir)):
        if i is None:
            vocab = t["embedding.word_embeddings.weight"].shape[0]
            cfg = hf_config(m, vocab)
            cfg["engram_table_dtype"] = engram_table_dtype
            writer.add("model.embed_tokens.weight", t.pop("embedding.word_embeddings.weight").to(cast))
            writer.add("model.norm.weight", t.pop("decoder.final_norm.weight").to(cast))
            if not cfg["tie_word_embeddings"]:
                writer.add("lm_head.weight", t.pop("output_layer.weight").to(cast))
            if t:
                raise KeyError(f"unmapped non-layer tensors {sorted(t)}")
            continue
        kind = cfg["block_types"][i]
        p, hp = f"decoder.layers.{i}.", f"model.layers.{i}."
        g = lambda name: t.pop(p + name)  # noqa: E731
        mixer = {"attention": "mhc_attn_mixer", "ckda": "mhc_attn_mixer", "moe": "mhc_mlp_mixer",
                 "engram": "residual_scheme.mhc_mixer"}[kind]
        for n in _MHC:  # mHC coefficients stay fp32 (Megatron computes them in fp32)
            writer.add(hp + f"mhc.{n}", g(f"{mixer}.{n}").float())
        if kind == "attention":
            q, z, kk, v = _split_attention(g("self_attention.linear_qkv.weight"), cfg)
            for n, w in (("q_proj", q), ("gate_proj", z), ("k_proj", kk), ("v_proj", v)):
                writer.add(hp + f"mixer.{n}.weight", w.to(cast))
            writer.add(hp + "mixer.o_proj.weight", g("self_attention.linear_proj.weight").to(cast))
            writer.add(hp + "mixer.q_norm.weight", g("self_attention.q_layernorm.weight").to(cast))
            writer.add(hp + "mixer.k_norm.weight", g("self_attention.k_layernorm.weight").to(cast))
            writer.add(hp + "input_layernorm.weight", g("self_attention.linear_qkv.layer_norm_weight").to(cast))
        elif kind == "ckda":
            writer.add(hp + "input_layernorm.weight", g("input_layernorm.weight").to(cast))
            pre = p + "self_attention.layer."
            for k in sorted(k for k in t if k.startswith(pre)):
                writer.add(hp + "mixer.layer." + k[len(pre):], t.pop(k).to(cast))
        elif kind == "moe":
            writer.add(hp + "input_layernorm.weight", g("pre_mlp_layernorm.weight").to(cast))
            writer.add(hp + "mixer.gate", g("mlp.router.weight").float())
            writer.add(hp + "mixer.e_score_correction_bias", g("mlp.router.expert_bias").float())
            gate = stack_experts(t, p + "mlp.experts.linear_fc1.weight", "_w")
            up = stack_experts(t, p + "mlp.experts.linear_fc1.weight", "_v")
            writer.add(hp + "mixer.gate_up_proj", torch.cat([gate, up], dim=1).to(cast))
            del gate, up
            writer.add(hp + "mixer.down_proj", stack_experts(t, p + "mlp.experts.linear_fc2.weight").to(cast))
            if cfg["shared_expert_intermediate_size"]:
                writer.add(hp + "mixer.shared_gate_proj.weight", g("mlp.shared_experts.linear_fc1.weight_w").to(cast))
                writer.add(hp + "mixer.shared_up_proj.weight", g("mlp.shared_experts.linear_fc1.weight_v").to(cast))
                writer.add(hp + "mixer.shared_down_proj.weight", g("mlp.shared_experts.linear_fc2.weight").to(cast))
        else:  # engram: drop the 4096-row alignment padding
            rows = sum(cfg["engram_table_sizes"])
            table = g("table")
            if table.shape[0] < rows:
                raise ValueError(f"layer {i}: table has {table.shape[0]} rows < {rows}")
            writer.add(hp + "mixer.table", table[:rows].to(tcast))
            del table
            for n in ("key_proj.weight", "value_proj.weight", "hidden_norm.weight", "key_norm.weight", "conv1d.weight"):
                writer.add(hp + f"mixer.{n}", g(n).to(cast))
            if cfg["engram_conv_prenorm"]:
                writer.add(hp + "mixer.conv_norm.weight", g("conv_norm.weight").to(cast))
            if cfg["engram_token_map"]:
                # the map covers the padded tokenizer vocabulary; ids >= vocab never reach the model
                tmap = np.load(m["engram_token_map"]).astype(np.int64)[: cfg["vocab_size"]]
                writer.add(hp + "mixer.token_map", torch.from_numpy(tmap))
        if t:
            raise KeyError(f"layer {i} ({kind}): unmapped tensors {sorted(t)}")
        LOGGER.info("layer %d (%s) done", i, kind)
    writer.close()
    write_config(out_dir, cfg, tokenizer)
    for f in ("configuration_oellm_ambitious.py", "modeling_oellm_ambitious.py"):
        shutil.copy2(_PKG / f, out_dir / f)
    # the reused modules (CKDA wrapper, Engram base, norm, cache) and their config
    for f in ("modeling_oellm_hybrid.py", "configuration_oellm_hybrid.py"):
        shutil.copy2(_HERE / "oellm_hybrid" / f, out_dir / f)
    if tokenizer is not None:
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
            if (Path(tokenizer) / name).exists():
                shutil.copy2(Path(tokenizer) / name, out_dir / name)
    (out_dir / "megatron_source.json").write_text(json.dumps({"checkpoint": str(ckpt_dir), "config": str(config_yaml)}, indent=2) + "\n")
    return out_dir


def _split_attention(qkv: torch.Tensor, cfg: dict):
    """Megatron's gated linear_qkv -> q, gate, k, v (HF keeps the gate as its own projection)."""
    heads, groups, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    if not cfg["attention_output_gate"]:
        raise ValueError("expects the gated attention")
    qz, k, v = split_gated_qkv(qkv, heads, groups, d)
    qz = qz.view(heads, 2 * d, -1)
    return qz[:, :d].reshape(heads * d, -1), qz[:, d:].reshape(heads * d, -1), k, v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="the run's resolved config-<job>.yaml")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--engram-table-dtype", default="bf16", choices=["bf16", "fp32"])
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    convert(a.ckpt_dir, a.config, a.out_dir, a.tokenizer, a.dtype, a.engram_table_dtype)


if __name__ == "__main__":
    main()
