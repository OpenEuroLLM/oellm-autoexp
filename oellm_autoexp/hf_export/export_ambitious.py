"""Export the ambitious hybrid main runs (Megatron HybridModel: CKDA + gated NoPE XSA attention,
mHC-4 or Linear Depth Memory, MoE with expert-bias or quantile balancing and the optional bounded
post-activation norm, Engram with the optional sign hash; ``fsdp_dtensor`` checkpoints) as a HF
model (``oellm_ambitious/``, trust_remote_code). One HF layer per Megatron pattern symbol; see the
modeling file.

    python -m oellm_autoexp.hf_export.export_ambitious --ckpt-dir <run>/ckpt_eval/iter_0000500 \\
        --config <run>/config-<job>.yaml --out-dir <exports>/ambitious_500 --tokenizer <tok>
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from .convert_megatron_to_hf import _fla_layer_spec
from .export_q35 import megatron_args, split_gated_qkv, write_config
from .fsdp_dtensor import ShardedSafetensorsWriter, iter_layers

LOGGER = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent
_PKG = _HERE / "oellm_ambitious"
_KINDS = {"*": "attention", "L": "ckda", "E": "moe", "N": "engram", "K": "kronlstm"}  # K: a ckda_branch KronLSTM
_MHC = ("phi", "b_pre", "b_post", "b_res", "alpha_pre", "alpha_post", "alpha_res")

# megatron/core/ssm/engram.py
_HASH_SEED = 0x5EED_E46A
_SIGN_HASH_SEED = 0x516E_0BAD
_SIGN_TABLE_SEED = 0x516E_7AB1
_LDM_NET = ("net.norm_weight", "net.w1.weight", "net.w1.bias", "net.address.weight", "net.address.bias",
            "net.gate.weight", "net.gate.bias")


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


def engram_hash_multipliers(m: dict, layer_number: int, seed: int = _HASH_SEED) -> list[list[int]]:
    seed = seed + (10007 * int(layer_number) if m.get("engram_hash_per_layer") else 0)
    rng = np.random.default_rng(seed)
    n = len(m["engram_ngram_orders"]) * int(m["engram_num_heads"])
    mult = rng.integers(1 << 20, 1 << 31, size=(n, max(m["engram_ngram_orders"])))
    return (mult | 1).astype(np.int64).tolist()


def engram_sign_multipliers(m: dict, layer_number: int) -> list[list[int]]:
    """Megatron ``engram_hash_multipliers(config, layer_number, seed=_SIGN_HASH_SEED)``."""
    return engram_hash_multipliers(m, layer_number, seed=_SIGN_HASH_SEED)


def engram_sign_table_np(m: dict) -> np.ndarray:
    """Megatron ``engram_sign_table``: the fixed +-1 table [sign_table_size, embed_dim] (fp32)."""
    rng = np.random.default_rng(_SIGN_TABLE_SEED)
    bits = rng.integers(0, 2, size=(int(m.get("engram_sign_table_size", 8192)), int(m["engram_embed_dim"])))
    return (2.0 * bits - 1.0).astype(np.float32)


FP8_BLOCK = 128
FP8_MAX = 448.0  # E4M3 (float8_e4m3fn) largest finite value


def fp8_tiles(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Megatron's fsdp_fp8_param_gather quantization of a bf16 weight (fp8_param_gather.quantize_pack):
    1 x 128 tiles along the last dim, scale = amax / 448 (1 for an all-zero tile), E4M3 of
    clamp(w / scale). The gathered shards' tiles are these rows' tiles: every expert tensor and every
    rank shard starts at a multiple of 128 elements, and the rows (2048 / 768) are multiples of 128.
    Returns (E4M3 bytes as uint8, fp32 scales [..., last / 128])."""
    if w.dtype != torch.bfloat16 or w.shape[-1] % FP8_BLOCK:
        raise ValueError(f"expects bf16 rows of a multiple of {FP8_BLOCK}, got {w.dtype} {tuple(w.shape)}")
    xb = w.float().reshape(-1, FP8_BLOCK)
    amax = xb.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (xb / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
    return q.view(w.shape), scale.view(*w.shape[:-1], w.shape[-1] // FP8_BLOCK)


def _balancing(m: dict) -> str:
    lb = m.get("moe_router_load_balancing_type")
    types = lb if isinstance(lb, (list, tuple)) else [lb]
    if "quantile_balancing" in types:  # the listed aux losses act in training only
        return "quantile"
    if not m.get("moe_router_enable_expert_bias"):
        raise ValueError("expects quantile balancing or the expert bias")
    return "expert_bias"


def branch_fold(m: dict) -> tuple[set[int], float]:
    """A ckda_branch run (megatron/core/ssm/ckda_branch.py) past its hand-over: the decoder layers whose
    complex_kda mixer is weighted 0 (alpha_end 0) and whose attention branch is the mixer, and the
    factor on gamma: the branch's output is scaled by factor * gamma (additive) or 1 + factor * gamma
    (ckda_branch_crossfade). Those layers export as standard attention layers.
    ``m["_ckpt_iteration"]``: the checkpoint's iteration."""
    if not m.get("ckda_branch"):
        return set(), 1.0
    if m["ckda_branch"] not in ("attention", "kronlstm"):
        raise ValueError(f"ckda_branch {m['ckda_branch']!r}: attention or kronlstm branches export")
    if float(m.get("ckda_branch_alpha_end", 0.0)) != 0.0:
        raise ValueError("ckda_branch: complex_kda still weighted after the ramp (alpha_end != 0): not exportable")
    it = m.get("_ckpt_iteration")
    start = int(m.get("ckda_branch_start_iteration", 0))
    hold, ramp = int(m.get("ckda_branch_alpha_hold", 25)), int(m.get("ckda_branch_alpha_ramp", 175))
    if it is None or it - start < hold + ramp:
        raise ValueError(f"ckda_branch: checkpoint {it} is inside the hand-over ({start} + {hold} + {ramp}): "
                         "complex_kda still contributes")
    groups = int(m.get("ckda_branch_num_query_groups", 2))
    stack = m["num_query_groups"] if m.get("group_query_attention") else m["num_attention_heads"]
    if m["ckda_branch"] == "attention" and groups != stack:
        raise ValueError(f"ckda_branch: {groups} KV groups in the branch, {stack} in the stack's attention: "
                         "per-layer KV groups are not in the HF model")
    t = it - start
    m0, mramp = float(m.get("ckda_branch_m_start", 4.0)), int(m.get("ckda_branch_m_ramp", 200))
    mult = 1.0 if mramp <= 0 or t >= mramp else m0 + (1.0 - m0) * t / mramp
    if m.get("ckda_branch_crossfade"):
        mult = 1.0  # the branch's weight 1 - alpha_end = 1, times (1 + scale * gamma)
    return set(int(i) for i in m.get("ckda_branch_layers") or []), mult * float(m.get("ckda_branch_gamma_scale", 1.0))


def _kronlstm_layer_spec(m: dict) -> dict:
    """The FLA KronLSTM layer a ckda_branch builds (Megatron fla_hybrid.KronLSTM._build, with
    ckda_branch_from_ckda: the convolution without SiLU when complex_kda has none)."""
    head_dim, heads = int(m["linear_key_head_dim"]), int(m["linear_num_key_heads"])
    expand_v = float(m.get("linear_fla_expand_v") or 1.0)
    kw = dict(
        hidden_size=int(m["hidden_size"]), k_head_dim=head_dim, v_head_dim=int(head_dim * expand_v),
        num_k_heads=heads, num_v_heads=heads, mode="chunk", use_short_conv=True,
        conv_size=int(m.get("linear_conv_kernel_dim") or 4), conv_bias=False,
        normalize_v=bool(m.get("linear_kronlstm_normalize_v", True)),
        allow_neg_eigval=bool(m.get("linear_fla_allow_neg_eigval")),
        use_target_gate=bool(m.get("linear_kronlstm_target_gate", True)),
        decouple_erase=bool(m.get("linear_kronlstm_decouple_erase", True)),
        neg_eigval_init_bias=bool(m.get("linear_kronlstm_neg_eigval_init_bias")),
        unit_write=bool(m.get("linear_kronlstm_unit_write")),
        forget_offset=float(m.get("linear_kronlstm_forget_offset") or 0.0),
        forget_angle=bool(m.get("linear_kronlstm_forget_angle")),
        forget_angle_init=str(m.get("linear_kronlstm_forget_angle_init") or "narrow"),
        erase_angle=bool(m.get("linear_kronlstm_erase_angle")),
        signed_write=bool(m.get("linear_kronlstm_signed_write")),
        drop_silu=bool(m.get("linear_drop_qkv_silu")),
        output_gate_rank={"linear": "full", "lowrank": "lowrank"}[m.get("linear_kronlstm_output_gate") or "linear"],
        decay_gate="vector", gate=str(m.get("linear_fla_gate") or "softplus"),
        lower_bound=float(m.get("linear_fla_lower_bound") if m.get("linear_fla_lower_bound") is not None else -5.0),
        norm_eps=float(m["layernorm_epsilon"]),
    )
    return {"class": "fla.layers.kronlstm.KronLSTM", "kwargs": kw}


def hf_config(m: dict, vocab: int) -> dict:
    syms = [c for c in m["hybrid_layer_pattern"] if c != "|"]
    bad = sorted({c for c in syms if c not in _KINDS})
    if bad:
        raise ValueError(f"unsupported pattern symbols {bad}")
    folded, _ = branch_fold(m)
    if any(syms[i] != "L" for i in folded):
        raise ValueError(f"ckda_branch layers {sorted(folded)} must be complex_kda (L) layers")
    new = "*" if m.get("ckda_branch") == "attention" else "K"
    syms = [new if i in folded else c for i, c in enumerate(syms)]
    ldm = bool(m.get("depth_memory"))
    if ldm:
        if m.get("mhc"):
            raise ValueError("depth_memory and mhc together")
        if m.get("depth_memory_lerp") or m.get("depth_memory_sites", "all") != "all":
            raise ValueError("expects LDM on every sublayer, additive residual (no lerp)")
    elif not m.get("mhc") or m.get("mhc_routed_topk") or m.get("mhc_internal_attention", "none") != "none" or m.get("mhc_lerp"):
        raise ValueError("expects plain mHC (no routed / internal-attention / lerp variants) or LDM")
    if m.get("moe_router_score_function") != "sigmoid":
        raise ValueError("expects the sigmoid router")
    balancing = _balancing(m)
    if m.get("moe_shared_expert_gate"):
        raise ValueError("gated shared expert not ported")
    if m.get("position_embedding_type") not in (None, "none"):
        raise ValueError("expects NoPE attention")
    if m.get("use_mup"):
        raise ValueError("muP embedding / output scaling not ported")
    if m.get("linear_xsa"):
        raise ValueError("linear XSA in CKDA not ported")
    heads = m["num_attention_heads"]
    types = [_KINDS[c] for c in syms]
    engram_layers = [i for i, t in enumerate(types) if t == "engram"]
    sign = bool(engram_layers and m.get("engram_sign_hash"))
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
        "residual": "ldm" if ldm else "mhc",
        "ldm_groups": int(m.get("depth_memory_groups", 16)),
        "ldm_key_dim": int(m.get("depth_memory_key_dim", 8)),
        "ldm_hidden_dim": int(m.get("depth_memory_hidden_dim", 32)),
        "ldm_alpha": float(m.get("depth_memory_alpha", 1.0)),
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
        "fla_layer_kronlstm": _kronlstm_layer_spec(m) if "kronlstm" in types else None,
        "num_experts": m["num_experts"],
        "num_experts_per_tok": m["moe_router_topk"],
        "moe_intermediate_size": m["moe_ffn_hidden_size"],
        "shared_expert_intermediate_size": m.get("moe_shared_expert_intermediate_size") or 0,
        "router_scaling_factor": float(m.get("moe_router_topk_scaling_factor") or 1.0),
        "activation_clamp_value": m.get("activation_func_clamp_value"),
        "router_balancing": balancing,
        "moe_expert_post_act_norm": bool(m.get("moe_expert_post_act_norm")),
        "mlp_post_act_norm": bool(m.get("mlp_post_act_norm")),
        "post_act_norm_floor": float(m.get("post_act_norm_floor") or 0.0),
        "post_act_norm_smooth": bool(m.get("post_act_norm_smooth")),
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
        "engram_sign_hash": sign,
        "engram_sign_table_size": int(m.get("engram_sign_table_size", 8192)),
        "engram_sign_multipliers": {str(i): engram_sign_multipliers(m, i + 1) for i in engram_layers} if sign else {},
        "embedding_multiplier": float(m.get("embedding_multiplier") or 1.0),
        "embedding_norm": bool(m.get("embedding_norm")),
        "tie_word_embeddings": not m.get("untie_embeddings_and_output_weights", False),
        "torch_dtype": "bfloat16",
        # the context window: vLLM falls back to 2048 when a config names none (every eval of
        # these exports ran with 2048 until 2026-11-06: few-shot contexts cut from the left,
        # generations capped). No position embeddings (NoPE + CKDA): no positional limit, 100M
        # (vLLM auto-fits max_model_len to its KV cache for these exports, submodules/vllm
        # config/model.py); a run with RoPE / learned positions keeps its own.
        "max_position_embeddings": max_positions(m),
    }


def max_positions(m: dict) -> int:
    """The exported context window: the run's own for position embeddings with a range, else
    OELLM_EXPORT_MAX_POSITIONS (default 100M: the model length-extrapolates; vLLM auto-fits)."""
    if str(m.get("position_embedding_type") or "none") == "none":
        return int(os.environ.get("OELLM_EXPORT_MAX_POSITIONS", "100000000"))
    return int(m.get("max_position_embeddings") or m.get("seq_length"))


def export_hf_config(m: dict, vocab: int, dtype: str = "bf16", engram_table_dtype: str = "bf16",
                     expert_dtype: str = "bf16") -> dict:
    cfg = hf_config(m, vocab)
    cfg["engram_table_dtype"] = engram_table_dtype
    cfg["expert_weight_dtype"] = expert_dtype
    if expert_dtype == "fp8" and dtype != "bf16":
        raise ValueError("fp8 experts are the bf16 weights' tiles: --dtype bf16")
    return cfg


def convert_nonlayer(t: dict, cfg: dict, add, cast: torch.dtype) -> None:
    """The non-layer tensors (embedding, final norm, LDM boundary, output layer) -> ``add(name, tensor)``."""
    add("model.embed_tokens.weight", t.pop("embedding.word_embeddings.weight").to(cast))
    if cfg["embedding_norm"]:
        add("model.embed_norm.weight", t.pop("embedding.embedding_norm_weight").to(cast))
    if cfg["residual"] == "ldm":  # the embedding write and the first query
        for site in ("embed_write", "first_query"):
            names = _LDM_NET + (("forget_logit",) if site == "embed_write" else ("read_gain",))
            for n in names:
                add(f"model.ldm_boundary.{site}.{n}", t.pop(f"decoder.ldm_stream_boundary.{site}.{n}").to(cast))
    add("model.norm.weight", t.pop("decoder.final_norm.weight").to(cast))
    if not cfg["tie_word_embeddings"]:
        add("lm_head.weight", t.pop("output_layer.weight").to(cast))
    if t:
        raise KeyError(f"unmapped non-layer tensors {sorted(t)}")


def _expert_ids(t: dict, prefix: str, suffix: str = "") -> list[str]:
    """The keys ``{prefix}{e}{suffix}``, e = 0..E-1 (a contiguous set)."""
    pat = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + "$")
    found = sorted((int(mm.group(1)), k) for k in t if (mm := pat.match(k)))
    if not found or [e for e, _ in found] != list(range(len(found))):
        raise KeyError(f"experts {prefix}N{suffix}: not a contiguous 0..E-1 set ({len(found)} found)")
    return [k for _, k in found]


def routed_experts(t: dict, p: str, cast: torch.dtype, expert_dtype: str) -> dict[str, torch.Tensor]:
    """The routed experts as [E, ...] tensors, one expert at a time (host memory: the stacked fp32 copies
    of 360 experts were ~15 GB per layer). Megatron's linear_fc1 rows: gate then up, as here. The fp8
    tiles are 1 x 128 along the last dim, so per expert = per stacked tensor."""
    fc1 = p + "mlp.experts.linear_fc1.weight"
    gates, ups = _expert_ids(t, fc1, "_w"), _expert_ids(t, fc1, "_v")
    downs = _expert_ids(t, p + "mlp.experts.linear_fc2.weight")
    if not len(gates) == len(ups) == len(downs):
        raise KeyError(f"{p}: {len(gates)} / {len(ups)} / {len(downs)} expert gate / up / down weights")
    parts = {
        "gate_up_proj": [lambda g=g, u=u: torch.cat([t.pop(g), t.pop(u)], dim=0) for g, u in zip(gates, ups)],
        "down_proj": [lambda d=d: t.pop(d) for d in downs],
    }
    out = {}
    for n, fetch in parts.items():
        for e, f in enumerate(fetch):
            w = f().to(cast)
            pieces = {n: w} if expert_dtype != "fp8" else dict(zip((n, n + "_scale"), fp8_tiles(w)))
            for k, v in pieces.items():
                if e == 0:
                    out[k] = torch.empty((len(fetch), *v.shape), dtype=v.dtype)
                out[k][e] = v
    return out


def convert_layer(i: int, t: dict, cfg: dict, m: dict, add, cast: torch.dtype, tcast: torch.dtype,
                  expert_dtype: str, table: bool = True) -> None:
    """Decoder layer ``i``'s tensors (training names) -> ``add(hf name, tensor)``. ``table=False``: the
    Engram table is converted elsewhere (export_parallel's row ranges), see :func:`engram_table_rows`."""
    kind = cfg["block_types"][i]
    p, hp = f"decoder.layers.{i}.", f"model.layers.{i}."
    g = lambda name: t.pop(p + name)  # noqa: E731
    mixer = {"attention": "mhc_attn_mixer", "ckda": "mhc_attn_mixer", "kronlstm": "mhc_attn_mixer", "moe": "mhc_mlp_mixer",
             "engram": "residual_scheme.mhc_mixer"}[kind]
    if cfg["residual"] == "ldm":  # this layer's write and the next read's query
        for n in _LDM_NET + ("forget_logit",):
            add(hp + f"ldm_mixer.write_site.{n}", g(f"{mixer}.write_site.{n}").to(cast))
        for n in _LDM_NET + ("read_gain",):
            add(hp + f"ldm_mixer.next_query.{n}", g(f"{mixer}.next_query.{n}").to(cast))
    else:
        for n in _MHC:  # mHC coefficients stay fp32 (Megatron computes them in fp32)
            add(hp + f"mhc.{n}", g(f"{mixer}.{n}").float())
    folded, scale = branch_fold(m)
    if kind == "attention" and i in folded:
        # a ckda_branch layer past its hand-over: the branch is the mixer (complex_kda weighted 0), its
        # input norm the layer's, the branch's output weight folded into the output projection
        b = "self_attention.branch."
        q, z, kk, v = _split_attention(g(b + "linear_qkv.weight"), cfg)
        for n, w in (("q_proj", q), ("gate_proj", z), ("k_proj", kk), ("v_proj", v)):
            add(hp + f"mixer.{n}.weight", w.to(cast))
        gamma = float(g("self_attention.branch_mix.gamma").float().reshape(-1)[0])
        weight = 1.0 + scale * gamma if m.get("ckda_branch_crossfade") else scale * gamma
        if p + "self_attention.branch_mix.calib" in t:  # ckda_branch_calibrate's measured scale
            weight *= float(g("self_attention.branch_mix.calib").float().reshape(-1)[0])
        add(hp + "mixer.o_proj.weight", (g(b + "linear_proj.weight").float() * weight).to(cast))
        add(hp + "mixer.q_norm.weight", g(b + "q_layernorm.weight").to(cast))
        add(hp + "mixer.k_norm.weight", g(b + "k_layernorm.weight").to(cast))
        add(hp + "input_layernorm.weight", g("input_layernorm.weight").to(cast))
        for k in [k for k in t if k.startswith(p + "self_attention.layer.")]:  # the weight-0 complex_kda
            t.pop(k)
    elif kind == "kronlstm" and i in folded:
        # a ckda_branch KronLSTM past its hand-over: the branch's layer is the mixer (complex_kda weighted 0),
        # the layer's input norm, the branch's output weight folded into its o_proj
        gamma = float(g("self_attention.branch_mix.gamma").float().reshape(-1)[0])
        weight = 1.0 + scale * gamma if m.get("ckda_branch_crossfade") else scale * gamma
        if p + "self_attention.branch_mix.calib" in t:
            weight *= float(g("self_attention.branch_mix.calib").float().reshape(-1)[0])
        add(hp + "input_layernorm.weight", g("input_layernorm.weight").to(cast))
        pre = p + "self_attention.branch.layer."
        for k in sorted(k for k in t if k.startswith(pre)):
            w = t.pop(k)
            if k == pre + "o_proj.weight":
                w = w.float() * weight
            add(hp + "mixer.layer." + k[len(pre):], w.to(cast))
        for k in [k for k in t if k.startswith(p + "self_attention.layer.")]:  # the weight-0 complex_kda
            t.pop(k)
    elif kind == "attention":
        q, z, kk, v = _split_attention(g("self_attention.linear_qkv.weight"), cfg)
        for n, w in (("q_proj", q), ("gate_proj", z), ("k_proj", kk), ("v_proj", v)):
            add(hp + f"mixer.{n}.weight", w.to(cast))
        add(hp + "mixer.o_proj.weight", g("self_attention.linear_proj.weight").to(cast))
        add(hp + "mixer.q_norm.weight", g("self_attention.q_layernorm.weight").to(cast))
        add(hp + "mixer.k_norm.weight", g("self_attention.k_layernorm.weight").to(cast))
        add(hp + "input_layernorm.weight", g("self_attention.linear_qkv.layer_norm_weight").to(cast))
        # trained sparse-attention parameters (page_sparse_selector nsa / csa2 / indexer): not part of the HF
        # model; campaign_midtraining/extract_variant_params.py takes them for the long-context probe
        for k in [k for k in t if k.startswith(p + "self_attention.core_attention.")]:
            t.pop(k)
    elif kind == "ckda":
        add(hp + "input_layernorm.weight", g("input_layernorm.weight").to(cast))
        pre = p + "self_attention.layer."
        for k in sorted(k for k in t if k.startswith(pre)):
            add(hp + "mixer.layer." + k[len(pre):], t.pop(k).to(cast))
    elif kind == "moe":
        add(hp + "input_layernorm.weight", g("pre_mlp_layernorm.weight").to(cast))
        add(hp + "mixer.gate", g("mlp.router.weight").float())
        if cfg["router_balancing"] == "quantile":
            add(hp + "mixer.qb_beta", g("mlp.router.qb_beta").float())
        else:
            add(hp + "mixer.e_score_correction_bias", g("mlp.router.expert_bias").float())
        experts = routed_experts(t, p, cast, expert_dtype)
        for n in list(experts):
            add(hp + f"mixer.{n}", experts.pop(n))
        if cfg["shared_expert_intermediate_size"]:
            add(hp + "mixer.shared_gate_proj.weight", g("mlp.shared_experts.linear_fc1.weight_w").to(cast))
            add(hp + "mixer.shared_up_proj.weight", g("mlp.shared_experts.linear_fc1.weight_v").to(cast))
            add(hp + "mixer.shared_down_proj.weight", g("mlp.shared_experts.linear_fc2.weight").to(cast))
    else:  # engram: drop the 4096-row alignment padding
        if table:
            rows = sum(cfg["engram_table_sizes"])
            tab = g("table")
            if tab.shape[0] < rows:
                raise ValueError(f"layer {i}: table has {tab.shape[0]} rows < {rows}")
            # host memory: the fp32 table (160 GB for v3 / v4) goes before the cast copy is written
            # (the writer's serialization copies it once more); peak ~385 -> ~240 GB
            cast_table = tab[:rows].to(tcast)
            del tab
            add(hp + "mixer.table", cast_table)
            del cast_table
        for n in ("key_proj.weight", "value_proj.weight", "hidden_norm.weight", "key_norm.weight", "conv1d.weight"):
            add(hp + f"mixer.{n}", g(n).to(cast))
        if cfg["engram_conv_prenorm"]:
            add(hp + "mixer.conv_norm.weight", g("conv_norm.weight").to(cast))
        if cfg["engram_sign_hash"]:  # not in the checkpoint: a function of Megatron's fixed seed
            add(hp + "mixer.sign_table", torch.from_numpy(engram_sign_table_np(m)))
        if cfg["engram_token_map"]:
            # the map covers the padded tokenizer vocabulary; ids >= vocab never reach the model
            tmap = np.load(m["engram_token_map"]).astype(np.int64)[: cfg["vocab_size"]]
            add(hp + "mixer.token_map", torch.from_numpy(tmap))
    if t:
        raise KeyError(f"layer {i} ({kind}): unmapped tensors {sorted(t)}")


def finish_export(out_dir: Path, cfg: dict, tokenizer: Path | None, ckpt_dir: Path, config_yaml: Path) -> None:
    """config.json, the model code, the tokenizer and megatron_source.json next to the weights."""
    write_config(out_dir, cfg, tokenizer)
    for f in ("configuration_oellm_ambitious.py", "modeling_oellm_ambitious.py", "mhc_kernels.py", "oellm_kernels.py"):
        shutil.copy2(_PKG / f, out_dir / f)
    # the reused modules (CKDA wrapper, Engram base, norm, cache) and their config
    for f in ("modeling_oellm_hybrid.py", "configuration_oellm_hybrid.py"):
        shutil.copy2(_HERE / "oellm_hybrid" / f, out_dir / f)
    if tokenizer is not None:
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
            if (Path(tokenizer) / name).exists():
                shutil.copy2(Path(tokenizer) / name, out_dir / name)
    (out_dir / "megatron_source.json").write_text(json.dumps({"checkpoint": str(ckpt_dir), "config": str(config_yaml)}, indent=2) + "\n")


def convert(ckpt_dir: Path, config_yaml: Path, out_dir: Path, tokenizer: Path | None, dtype: str = "bf16",
            engram_table_dtype: str = "bf16", expert_dtype: str = "bf16") -> Path:
    m = megatron_args(config_yaml)
    it = re.search(r"iter_(\d+)", Path(ckpt_dir).name)
    m["_ckpt_iteration"] = int(it.group(1)) if it else None  # ckda_branch folding (branch_fold)
    cast = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    tcast = {"bf16": torch.bfloat16, "fp32": torch.float32}[engram_table_dtype]
    out_dir = Path(out_dir)
    writer = ShardedSafetensorsWriter(out_dir)
    cfg = None
    for i, t in iter_layers(Path(ckpt_dir)):
        if i is None:
            cfg = export_hf_config(m, t["embedding.word_embeddings.weight"].shape[0], dtype, engram_table_dtype, expert_dtype)
            convert_nonlayer(t, cfg, writer.add, cast)
            continue
        convert_layer(i, t, cfg, m, writer.add, cast, tcast, expert_dtype)
        LOGGER.info("layer %d (%s) done", i, cfg["block_types"][i])
    writer.close()
    finish_export(out_dir, cfg, tokenizer, ckpt_dir, config_yaml)
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
    p.add_argument("--expert-dtype", default="bf16", choices=["bf16", "fp8"],
                   help="routed experts; fp8 = the run's fsdp_fp8_param_gather tiles (E4M3, 1 x 128 fp32 scales)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    convert(a.ckpt_dir, a.config, a.out_dir, a.tokenizer, a.dtype, a.engram_table_dtype, a.expert_dtype)


if __name__ == "__main__":
    main()
