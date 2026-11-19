"""Export the Engram scaling runs (Megatron HybridModel: Gated DeltaNet ``L``, gated RoPE attention ``*``,
sigmoid-routed MoE with a gated shared expert ``E``, Engram ``N``; plain residual; ``torch_dist``
checkpoints) as a HF model (``oellm_engram_scaling/``, trust_remote_code). One HF layer per Megatron
pattern symbol; see the modeling file.

What differs from ``export_ambitious``: ``torch_dist`` (DCP) instead of ``fsdp_dtensor`` checkpoints
(experts stored stacked, GDN in_proj / conv1d split into named sections), a plain residual instead of
mHC / LDM, Gated DeltaNet instead of CKDA, RoPE attention without XSA, and the gated shared expert.
Config flags this port does not implement raise instead of exporting a different model.

    python -m oellm_autoexp.hf_export.export_engram_scaling \\
        --ckpt-dir <run>/ckpt/iter_0004819 --config <run>/config-<job>.yaml \\
        --out-dir <exports>/<run>_4819 --tokenizer <tokenizer dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from .convert_megatron_to_hf import load_megatron_state
from .export_ambitious import _split_attention, engram_hash_multipliers, engram_table_sizes
from .export_q35 import megatron_args, write_config
from .fsdp_dtensor import ShardedSafetensorsWriter

LOGGER = logging.getLogger(__name__)
_HERE = Path(__file__).resolve().parent
_KINDS = {"L": "gdn", "*": "attention", "E": "moe", "N": "engram"}
_GDN_IN_PROJ = ("query", "key", "value", "z", "beta", "alpha")
_GDN_CONV = ("query", "key", "value")

# (flag, the values this port implements); None counts as unset
_REQUIRED = {
    "experimental_attention_variant": ("gated_delta_net",),
    "position_embedding_type": ("rope",),
    "normalization": ("RMSNorm",),
    "swiglu": (True,),
    "add_bias_linear": (False,),
    "attention_output_gate": (True,),
    "linear_beta_activation": ("sigmoid",),
    "linear_beta_max": (1.0,),
    "moe_router_score_function": ("sigmoid",),
    "moe_router_enable_expert_bias": (True,),
    "moe_router_topk_scaling_factor": (None, 1.0),
    "moe_router_num_groups": (None,),
    "moe_router_pre_softmax": (False,),
    "moe_mlp_glu_interleave_size": (None,),
    "glu_linear_offset": (0.0,),
    "activation_func_clamp_value": (None,),
    "moe_expert_post_act_norm": (False, None),
    "mlp_post_act_norm": (False, None),
    "moe_zero_expert_num": (0, None),
    "moe_latent_size": (None,),
    "use_rope_scaling": (False,),
    "window_size": (None,),
    "softmax_scale": (None,),
    "attention_length_scaling": ("none", None),
    "embedding_multiplier": (1.0, None),
    "embedding_norm": (False, None),
    "use_mup": (False,),
    "engram_sign_hash": (False, None),
    "engram_aux_ngram_orders": (None,),
}
# residual / depth schemes and mixer options the port does not implement (must be unset / falsy)
_UNSET = ("mhc", "depth_memory", "attn_res", "ngpt", "loop_block_layers", "exclusive_self_attention",
          "linear_xsa", "value_embedding", "canon_layers", "per_layer_embedding")


def check_supported(m: dict) -> None:
    bad = []
    for k, ok in _REQUIRED.items():
        v = m.get(k)
        v = None if v == "None" else v
        if v not in ok:
            bad.append(f"{k}={v!r} (ported: {ok})")
    bad += [f"{k}={m[k]!r} (not ported)" for k in _UNSET if m.get(k) not in (None, False, 0, "", "None", [])]
    pattern = m["hybrid_layer_pattern"]
    syms = set(pattern.replace("|", ""))
    if not syms <= set(_KINDS):
        bad.append(f"hybrid_layer_pattern {pattern!r}: symbols {sorted(syms - set(_KINDS))} not ported")
    if bad:
        raise NotImplementedError("HF export does not support: " + "; ".join(bad))


def hf_config(m: dict, vocab: int) -> dict:
    types = [_KINDS[c] for c in m["hybrid_layer_pattern"] if c != "|"]
    engram_layers = [i for i, t in enumerate(types) if t == "engram"]
    heads = m["num_attention_heads"]
    head_dim = m["kv_channels"]
    return {
        "architectures": ["OellmEngramScalingForCausalLM"],
        "model_type": "oellm_engram_scaling",
        "auto_map": {
            "AutoConfig": "configuration_oellm_engram_scaling.OellmEngramScalingConfig",
            "AutoModelForCausalLM": "modeling_oellm_engram_scaling.OellmEngramScalingForCausalLM",
        },
        "vocab_size": vocab,
        "hidden_size": m["hidden_size"],
        "block_types": types,
        "rms_norm_eps": float(m["layernorm_epsilon"]),
        "layernorm_zero_centered_gamma": bool(m.get("layernorm_zero_centered_gamma", False)),
        "num_attention_heads": heads,
        # Megatron applies num_query_groups only with --group-query-attention
        "num_key_value_heads": m["num_query_groups"] if m.get("group_query_attention") else heads,
        "head_dim": head_dim,
        "qk_layernorm": bool(m.get("qk_layernorm")),
        "attention_output_gate": True,
        "rope_theta": float(m["rotary_base"]),
        "rotary_dim": int(head_dim * float(m.get("rotary_percent") or 1.0)),  # RotaryEmbedding: int(kv_channels * percent)
        "rotary_interleaved": bool(m.get("rotary_interleaved")),
        "max_position_embeddings": int(m.get("max_position_embeddings") or m["seq_length"]),
        "linear_num_key_heads": m["linear_num_key_heads"],
        "linear_num_value_heads": m["linear_num_value_heads"],
        "linear_key_head_dim": m["linear_key_head_dim"],
        "linear_value_head_dim": m["linear_value_head_dim"],
        "linear_conv_kernel_dim": m["linear_conv_kernel_dim"],
        "num_experts": m["num_experts"],
        "num_experts_per_tok": m["moe_router_topk"],
        "moe_intermediate_size": m["moe_ffn_hidden_size"],
        "shared_expert_intermediate_size": m.get("moe_shared_expert_intermediate_size") or 0,
        "shared_expert_gate": bool(m.get("moe_shared_expert_gate")),
        "engram_ngram_orders": list(m["engram_ngram_orders"]),
        "engram_num_heads": int(m["engram_num_heads"]),
        "engram_embed_dim": int(m["engram_embed_dim"]),
        "engram_conv_kernel_size": int(m.get("engram_conv_kernel_size", 4)),
        "engram_table_sizes": engram_table_sizes(m) if engram_layers else None,
        # HybridModel layer_number = pattern index + 1
        "engram_hash_multipliers": {str(i): engram_hash_multipliers(m, i + 1) for i in engram_layers},
        "engram_token_map": m.get("engram_token_map") not in (None, "None"),
        "engram_gate_signed_sqrt": bool(m.get("engram_gate_signed_sqrt")),
        "engram_conv_prenorm": bool(m.get("engram_conv_prenorm")),
        "engram_table_dtype": "bf16",
        # the head is written out even when Megatron ties it to the embedding
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }


def _cat(t: dict, prefix: str, sections: tuple[str, ...]) -> torch.Tensor:
    """Megatron's fused tensor from its named DCP sections (``<prefix>.query`` ...), in forward order."""
    return torch.cat([t.pop(f"{prefix}.{s}") for s in sections], dim=0)


def convert_layer(i: int, t: dict, cfg: dict, m: dict, add, cast: torch.dtype) -> None:
    """Decoder layer ``i``'s tensors (Megatron names, popped from ``t``) -> ``add(hf name, tensor)``."""
    kind = cfg["block_types"][i]
    p, hp = f"decoder.layers.{i}.", f"model.layers.{i}."
    g = lambda name: t.pop(p + name)  # noqa: E731
    if kind == "gdn":
        add(hp + "input_layernorm.weight", g("self_attention.in_proj.layer_norm_weight").to(cast))
        add(hp + "mixer.in_proj.weight", _cat(t, p + "self_attention.in_proj.weight", _GDN_IN_PROJ).to(cast))
        add(hp + "mixer.conv1d.weight", _cat(t, p + "self_attention.conv1d.weight", _GDN_CONV).to(cast))
        for n in ("A_log", "dt_bias", "out_norm.weight", "out_proj.weight"):  # out_norm: zero-centred both sides
            add(hp + f"mixer.{n}", g(f"self_attention.{n}").to(cast))
    elif kind == "attention":
        add(hp + "input_layernorm.weight", g("self_attention.linear_qkv.layer_norm_weight").to(cast))
        q, z, kk, v = _split_attention(g("self_attention.linear_qkv.weight"), cfg)
        for n, w in (("q_proj", q), ("gate_proj", z), ("k_proj", kk), ("v_proj", v)):
            add(hp + f"mixer.{n}.weight", w.to(cast))
        add(hp + "mixer.o_proj.weight", g("self_attention.linear_proj.weight").to(cast))
        if cfg["qk_layernorm"]:
            add(hp + "mixer.q_norm.weight", g("self_attention.q_layernorm.weight").to(cast))
            add(hp + "mixer.k_norm.weight", g("self_attention.k_layernorm.weight").to(cast))
    elif kind == "moe":
        add(hp + "input_layernorm.weight", g("pre_mlp_layernorm.weight").to(cast))
        add(hp + "mixer.gate", g("mlp.router.weight").float())
        add(hp + "mixer.e_score_correction_bias", g("mlp.router.expert_bias").float())
        # torch_dist stacks the grouped experts: fc1 [E, 2 * ffn (gate rows, then up), h], fc2 [E, h, ffn]
        add(hp + "mixer.gate_up_proj", g("mlp.experts.experts.linear_fc1.weight").to(cast))
        add(hp + "mixer.down_proj", g("mlp.experts.experts.linear_fc2.weight").to(cast))
        if cfg["shared_expert_intermediate_size"]:
            gate, up = g("mlp.shared_experts.linear_fc1.weight").chunk(2, dim=0)
            add(hp + "mixer.shared_gate_proj.weight", gate.to(cast))
            add(hp + "mixer.shared_up_proj.weight", up.to(cast))
            add(hp + "mixer.shared_down_proj.weight", g("mlp.shared_experts.linear_fc2.weight").to(cast))
            if cfg["shared_expert_gate"]:
                add(hp + "mixer.shared_expert_gate.weight", g("mlp.shared_experts.gate_weight").to(cast))
    else:  # engram: drop the 4096-row alignment padding; rows are looked up in fp32, then cast (bf16 exact)
        rows = sum(cfg["engram_table_sizes"])
        tab = g("table")
        if tab.shape[0] < rows:
            raise ValueError(f"layer {i}: table has {tab.shape[0]} rows < {rows}")
        add(hp + "mixer.table", tab[:rows].to(cast))
        for n in ("key_proj.weight", "value_proj.weight", "hidden_norm.weight", "key_norm.weight", "conv1d.weight"):
            add(hp + f"mixer.{n}", g(n).to(cast))
        if cfg["engram_conv_prenorm"]:
            add(hp + "mixer.conv_norm.weight", g("conv_norm.weight").to(cast))
        if cfg["engram_token_map"]:
            # the map covers the padded tokenizer vocabulary; ids >= vocab never reach the model
            tmap = np.load(m["engram_token_map"]).astype(np.int64)[: cfg["vocab_size"]]
            add(hp + "mixer.token_map", torch.from_numpy(tmap))
    left = [k for k in t if k.startswith(p)]
    if left:
        raise KeyError(f"layer {i} ({kind}): unmapped tensors {sorted(left)}")


def convert(ckpt_dir: Path, config_yaml: Path, out_dir: Path, tokenizer: Path | None, dtype: str = "bf16") -> Path:
    m = megatron_args(config_yaml)
    check_supported(m)
    cast = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    t = load_megatron_state(Path(ckpt_dir))
    emb = t.pop("embedding.word_embeddings.weight")
    cfg = hf_config(m, emb.shape[0])
    layers = {int(mm.group(1)) for k in t if (mm := re.match(r"decoder\.layers\.(\d+)\.", k))}
    if layers != set(range(len(cfg["block_types"]))):
        raise ValueError(f"checkpoint layers {sorted(layers)} != pattern {m['hybrid_layer_pattern']!r}")
    out_dir = Path(out_dir)
    writer = ShardedSafetensorsWriter(out_dir)
    writer.add("model.embed_tokens.weight", emb.to(cast))
    head = t.pop("output_layer.weight", None)
    if m.get("untie_embeddings_and_output_weights"):
        if head is None:
            raise KeyError("untied run without output_layer.weight")
    else:
        head = emb
    writer.add("lm_head.weight", head.to(cast).clone())
    writer.add("model.norm.weight", t.pop("decoder.final_norm.weight").to(cast))
    for i in range(len(cfg["block_types"])):
        convert_layer(i, t, cfg, m, writer.add, cast)
        LOGGER.info("layer %d (%s) done", i, cfg["block_types"][i])
    if t:
        raise KeyError(f"unmapped tensors {sorted(t)}")
    writer.close()
    write_config(out_dir, cfg, tokenizer)
    # the model code: this package's, and the reused ambitious (MoE, Engram) and hybrid (GDN, norm) modules
    files = {_HERE / "oellm_engram_scaling": ("configuration_oellm_engram_scaling.py", "modeling_oellm_engram_scaling.py"),
             _HERE / "oellm_ambitious": ("configuration_oellm_ambitious.py", "modeling_oellm_ambitious.py",
                                         "mhc_kernels.py", "oellm_kernels.py"),
             _HERE / "oellm_hybrid": ("configuration_oellm_hybrid.py", "modeling_oellm_hybrid.py")}
    for src, names in files.items():
        for f in names:
            shutil.copy2(src / f, out_dir / f)
    if tokenizer is not None:
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
            if (Path(tokenizer) / name).exists():
                shutil.copy2(Path(tokenizer) / name, out_dir / name)
    (out_dir / "megatron_source.json").write_text(json.dumps({"checkpoint": str(ckpt_dir), "config": str(config_yaml)}, indent=2) + "\n")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="the run's resolved config-<job>.yaml")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    convert(a.ckpt_dir, a.config, a.out_dir, a.tokenizer, a.dtype)


if __name__ == "__main__":
    main()
