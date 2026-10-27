"""Export the conservative main run (Qwen3.5-like GDN 3:1 + gated attention MoE, Megatron
HybridModel, ``fsdp_dtensor`` checkpoints) as a HF model: transformers' Qwen3-Next with the
router replaced by Megatron's sigmoid + expert-bias routing (``oellm_q35/``, trust_remote_code).

Megatron numbers every pattern symbol as its own layer: layer 2k is the mixer ('*' gated
attention or 'L' Gated DeltaNet), 2k+1 its MoE ('E'). HF layer k holds both. The fused-layout
transforms follow Megatron-Bridge's Qwen3-Next bridge (``split_qkv_weights``,
``split_gdn_linear_weights`` with tp 1, ``RMSNorm2ZeroCenteredRMSNormMapping`` for the GDN output
norm); the other norms are zero-centred on both sides and copy over.

    python -m oellm_autoexp.hf_export.export_q35 --ckpt-dir <run>/ckpt_eval/iter_0002000 \\
        --config <run>/config-<job>.yaml --out-dir <exports>/q35_conservative_2k --tokenizer <tok>
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .fsdp_dtensor import ShardedSafetensorsWriter, iter_layers, stack_experts

LOGGER = logging.getLogger(__name__)
_PKG = Path(__file__).resolve().parent / "oellm_q35"


def megatron_args(config_yaml: Path) -> dict:
    doc = yaml.safe_load(Path(config_yaml).read_text())
    doc = doc.get("config", doc)  # the run's resolved config-<job>.yaml wraps it
    m = doc.get("backend", {}).get("megatron", doc)
    if "hidden_size" not in m:
        raise KeyError(f"{config_yaml}: no backend.megatron.hidden_size")
    return m


def layer_types(pattern: str) -> list[str]:
    syms = [c for c in pattern if c not in "|"]
    if len(syms) % 2 or any(e != "E" for e in syms[1::2]):
        raise ValueError(f"expected <mixer><E> pairs in {pattern!r}")
    kinds = {"*": "full_attention", "L": "linear_attention"}
    bad = sorted({c for c in syms[0::2] if c not in kinds})
    if bad:
        raise ValueError(f"unsupported mixers {bad} in {pattern!r}")
    return [kinds[c] for c in syms[0::2]]


def hf_config(m: dict, vocab: int) -> dict:
    if m.get("moe_router_score_function") != "sigmoid" or not m.get("moe_router_enable_expert_bias"):
        raise ValueError("expects the sigmoid router with expert bias")
    if m.get("moe_router_topk_scaling_factor") not in (None, 1, 1.0) or m.get("moe_router_num_groups"):
        raise ValueError("router scaling / group-limited routing not ported")
    heads = m["num_attention_heads"]
    # Megatron only applies num_query_groups with --group-query-attention; otherwise MHA
    kv_heads = m["num_query_groups"] if m.get("group_query_attention") else heads
    if m.get("activation_func_clamp_value") is not None:
        raise ValueError("clamped SwiGLU not ported to Qwen3-Next")
    return {
        "architectures": ["OellmQ35ForCausalLM"],
        "model_type": "oellm_q35",
        "auto_map": {
            "AutoConfig": "configuration_oellm_q35.OellmQ35Config",
            "AutoModelForCausalLM": "modeling_oellm_q35.OellmQ35ForCausalLM",
        },
        "vocab_size": vocab,
        "hidden_size": m["hidden_size"],
        "num_hidden_layers": len(layer_types(m["hybrid_layer_pattern"])),
        "layer_types": layer_types(m["hybrid_layer_pattern"]),
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": m["kv_channels"],
        "linear_num_key_heads": m["linear_num_key_heads"],
        "linear_num_value_heads": m["linear_num_value_heads"],
        "linear_key_head_dim": m["linear_key_head_dim"],
        "linear_value_head_dim": m["linear_value_head_dim"],
        "linear_conv_kernel_dim": m["linear_conv_kernel_dim"],
        "num_experts": m["num_experts"],
        "num_experts_per_tok": m["moe_router_topk"],
        "moe_intermediate_size": m["moe_ffn_hidden_size"],
        "shared_expert_intermediate_size": m["moe_shared_expert_intermediate_size"],
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "router_score_function": "sigmoid",
        "router_scaling_factor": 1.0,
        "hidden_act": "silu",
        "rms_norm_eps": m.get("layernorm_epsilon", m.get("norm_epsilon", 1e-5)),
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(m["rotary_base"]),
            "partial_rotary_factor": float(m.get("rotary_percent", 1.0)),
        },
        "max_position_embeddings": m.get("max_position_embeddings") or m["seq_length"],
        "attention_bias": False,
        "tie_word_embeddings": not m.get("untie_embeddings_and_output_weights", False),
        "torch_dtype": "bfloat16",
    }


# --- fused-layout splits (Megatron-Bridge param_mapping.py, tp 1) --------------------------------
def split_gated_qkv(qkv: torch.Tensor, heads: int, groups: int, head_dim: int):
    """Megatron's per-group [q heads | gate heads | k | v] -> HF q_proj (per head [q | gate]), k, v."""
    hpg = heads // groups
    per = 2 * hpg + 2
    w = qkv.view(groups * per, head_dim, -1)
    q = torch.cat([torch.arange(per * g, per * g + hpg) for g in range(groups)])
    z = torch.cat([torch.arange(per * g + hpg, per * g + 2 * hpg) for g in range(groups)])
    k = torch.arange(per - 2, groups * per, per)
    v = torch.arange(per - 1, groups * per, per)
    qz = torch.cat([w[q], w[z]], dim=1)  # [heads, 2*head_dim, H]
    h = qkv.shape[-1]
    return qz.reshape(-1, h), w[k].reshape(-1, h), w[v].reshape(-1, h)


def split_gdn_in_proj(w: torch.Tensor, nk: int, nv: int, dk: int, dv: int):
    """Megatron's [q | k | v | z | b | a] -> HF in_proj_qkvz / in_proj_ba (grouped per key head)."""
    h = w.shape[-1]
    q, k, v, z, b, a = torch.split(w, [nk * dk, nk * dk, nv * dv, nv * dv, nv, nv], dim=0)
    q, k, v, z, b, a = (t.reshape(nk, -1, h) for t in (q, k, v, z, b, a))
    return torch.cat([q, k, v, z], 1).reshape(-1, h), torch.cat([b, a], 1).reshape(-1, h)


def special_token_ids(tokenizer: Path | None) -> dict:
    """bos / eos / pad ids of the tokenizer. transformers 5 configs have no default for them;
    generate() reads them from generation_config.json."""
    if tokenizer is None:
        return {}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    return {k: getattr(tok, k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")}


def model_token_ids(ids: dict, vocab: int) -> dict:
    """The tokenizer's ids, with pad -> eos when pad lies outside the model's vocabulary (our
    tokenizer has 262272 entries, pad 262144, the embedding 262144 rows: a padded batch would
    index past the embedding)."""
    ids = dict(ids)
    if ids and (ids.get("pad_token_id") is None or ids["pad_token_id"] >= vocab):
        ids["pad_token_id"] = ids.get("eos_token_id")
    return ids


def write_config(out_dir: Path, cfg: dict, tokenizer: Path | None) -> None:
    ids = model_token_ids(special_token_ids(tokenizer), cfg["vocab_size"])
    (out_dir / "config.json").write_text(json.dumps({**cfg, **ids}, indent=2) + "\n")
    if ids:
        (out_dir / "generation_config.json").write_text(json.dumps(ids, indent=2) + "\n")


def convert(ckpt_dir: Path, config_yaml: Path, out_dir: Path, tokenizer: Path | None, dtype: str = "bf16") -> Path:
    m = megatron_args(config_yaml)
    cast = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    out_dir = Path(out_dir)
    writer = ShardedSafetensorsWriter(out_dir)
    cfg = None
    kv = None
    pending_mixer: dict[str, torch.Tensor] | None = None
    for i, t in iter_layers(Path(ckpt_dir)):
        if i is None:
            vocab = t["embedding.word_embeddings.weight"].shape[0]
            cfg = hf_config(m, vocab)
            kv = cfg["num_key_value_heads"]
            writer.add("model.embed_tokens.weight", t.pop("embedding.word_embeddings.weight").to(cast))
            writer.add("model.norm.weight", t.pop("decoder.final_norm.weight").to(cast))
            if not cfg["tie_word_embeddings"]:
                writer.add("lm_head.weight", t.pop("output_layer.weight").to(cast))
            if t:
                raise KeyError(f"unmapped non-layer tensors {sorted(t)}")
            continue
        k, half = divmod(i, 2)
        p, hp = f"decoder.layers.{i}.", f"model.layers.{k}."
        g = lambda name: t.pop(p + name)  # noqa: E731
        if half == 0:  # mixer
            kind = cfg["layer_types"][k]
            if kind == "full_attention":
                q, kk, v = split_gated_qkv(g("self_attention.linear_qkv.weight"), cfg["num_attention_heads"], kv, cfg["head_dim"])
                for n, w in (("q_proj", q), ("k_proj", kk), ("v_proj", v)):
                    writer.add(hp + f"self_attn.{n}.weight", w.to(cast))
                writer.add(hp + "self_attn.o_proj.weight", g("self_attention.linear_proj.weight").to(cast))
                writer.add(hp + "self_attn.q_norm.weight", g("self_attention.q_layernorm.weight").to(cast))
                writer.add(hp + "self_attn.k_norm.weight", g("self_attention.k_layernorm.weight").to(cast))
                writer.add(hp + "input_layernorm.weight", g("self_attention.linear_qkv.layer_norm_weight").to(cast))
            else:
                qkvz, ba = split_gdn_in_proj(g("self_attention.in_proj.weight"), cfg["linear_num_key_heads"], cfg["linear_num_value_heads"],
                                             cfg["linear_key_head_dim"], cfg["linear_value_head_dim"])
                writer.add(hp + "linear_attn.in_proj_qkvz.weight", qkvz.to(cast))
                writer.add(hp + "linear_attn.in_proj_ba.weight", ba.to(cast))
                writer.add(hp + "linear_attn.conv1d.weight", g("self_attention.conv1d.weight").to(cast))
                writer.add(hp + "linear_attn.A_log", g("self_attention.A_log").to(cast))
                writer.add(hp + "linear_attn.dt_bias", g("self_attention.dt_bias").to(cast))
                # zero-centred in Megatron, a plain (ones-initialised) RMSNorm in HF's gated norm
                writer.add(hp + "linear_attn.norm.weight", (g("self_attention.out_norm.weight") + 1.0).to(cast))
                writer.add(hp + "linear_attn.out_proj.weight", g("self_attention.out_proj.weight").to(cast))
                writer.add(hp + "input_layernorm.weight", g("self_attention.in_proj.layer_norm_weight").to(cast))
        else:  # MoE
            writer.add(hp + "post_attention_layernorm.weight", g("pre_mlp_layernorm.weight").to(cast))
            writer.add(hp + "mlp.gate.weight", g("mlp.router.weight").float())
            writer.add(hp + "mlp.gate.e_score_correction_bias", g("mlp.router.expert_bias").float())
            gate = stack_experts(t, p + "mlp.experts.linear_fc1.weight", "_w")
            up = stack_experts(t, p + "mlp.experts.linear_fc1.weight", "_v")
            writer.add(hp + "mlp.experts.gate_up_proj", torch.cat([gate, up], dim=1).to(cast))
            del gate, up
            writer.add(hp + "mlp.experts.down_proj", stack_experts(t, p + "mlp.experts.linear_fc2.weight").to(cast))
            writer.add(hp + "mlp.shared_expert.gate_proj.weight", g("mlp.shared_experts.linear_fc1.weight_w").to(cast))
            writer.add(hp + "mlp.shared_expert.up_proj.weight", g("mlp.shared_experts.linear_fc1.weight_v").to(cast))
            writer.add(hp + "mlp.shared_expert.down_proj.weight", g("mlp.shared_experts.linear_fc2.weight").to(cast))
            writer.add(hp + "mlp.shared_expert_gate.weight", g("mlp.shared_experts.gate_weight").to(cast))
        if t:
            raise KeyError(f"layer {i}: unmapped tensors {sorted(t)}")
        LOGGER.info("layer %d done", i)
    writer.close()
    write_config(out_dir, cfg, tokenizer)
    for f in ("configuration_oellm_q35.py", "modeling_oellm_q35.py"):
        shutil.copy2(_PKG / f, out_dir / f)
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
