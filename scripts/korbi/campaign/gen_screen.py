"""Generate the parameter-matched feature screen of the 3B campaign (docs/hybrid_moe_3b_campaign.md 3.4).

    python scripts/korbi/campaign/gen_screen.py

Two sizes (s1: hidden 768 x 12 blocks, the speed suite's geometry; s2: hidden 1024 x 20 blocks),
datamix4 (OELLM 256k vocabulary, tied embeddings), equal lr / bs per size, stable only (no cooldown)
to aux.tokens; arms are compared on the stable-phase loss.
Arms differ from the size's base in ONE feature; d_ffn is set per arm so that the non-embedding
parameters match the base within 1 % (config/experiments/korbi/campaign_3b/param_match.json, from
scripts/korbi/campaign/count_params.py; arms missing there use the base d_ffn and are marked).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "config/experiments/korbi/campaign_3b"
B = "backend.megatron."

SIZES = {
    # name: (hidden, blocks, heads, d_ffn, lr, gbs)
    "s1": (768, 12, 12, 2048, 1e-3, 64),
    "s2": (1024, 20, 16, 2816, 1e-3, 64),
}

# round-1 CKDA as first screened: spread alpha init and a linear output gate (NOT the reference
# CKDA; kept so the h_ckda* arms regenerate as they ran)
CKDA = {B + "experimental_attention_variant": "complex_kda", B + "linear_beta_max": 2.0,
        B + "linear_output_gate": "linear", B + "linear_drop_qkv_silu": True,
        B + "linear_gate_init_style": "spread"}
# the reference CKDA (config/experiments/korbi/complex_kda/sweep_spec.yaml): both ranges
# extended (signed_sigmoid2 alpha, beta_max 2), BOTH inits standard (alpha "shipped", bias-free
# beta), low-rank output gate, no q/k/v SiLU. The campaign's CKDA from here on.
CKDA_STD = {B + "experimental_attention_variant": "complex_kda", B + "linear_beta_max": 2.0,
            B + "linear_gate_activation": "signed_sigmoid2", B + "linear_gate_init_style": "shipped",
            B + "linear_output_gate": "lowrank", B + "linear_drop_qkv_silu": True}
# triadic CKDA (megatron/core/ssm/triadic.py, base ckda: signed gate, negative eigenvalues,
# standard inits), E = 4 slices per head
TRI_CKDA = {B + "experimental_attention_variant": "triadic", B + "linear_triadic_base": "ckda",
            B + "linear_triadic_e": 4}


def hybrid(blocks: int, mixer: str = "L", ratio: int = 3, engram_at: tuple[int, ...] = ()) -> str:
    """``ratio`` linear blocks per attention block (3:1 -> L-L-L-*-); ``engram_at``: block indices
    that get an Engram layer (N) in front."""
    period = ratio + 1
    assert blocks % period == 0, f"{blocks} blocks do not divide into {ratio}:1"
    out = []
    for i in range(blocks):
        mix = "*" if i % period == ratio else mixer
        out.append(("N" if i in engram_at else "") + f"{mix}-")
    return "".join(out)


def arms(blocks: int):
    lin = lambda v: {"aux.pattern": hybrid(blocks), B + "experimental_attention_variant": v}  # noqa: E731
    xsa = {B + "exclusive_self_attention": True}
    gdn128 = {B + "linear_key_head_dim": 128, B + "linear_value_head_dim": 128}
    gate = {B + "attention_output_gate": True}
    return [
        ("attn", "base: attention (qk-norm, RoPE) + SwiGLU", {}),
        ("attn_xsa", "exclusive self attention", xsa),
        ("attn_gate", "sigmoid attention output gate", gate),
        ("h_ckda", "3:1 CKDA : attention hybrid", {"aux.pattern": hybrid(blocks), **CKDA}),
        ("h_ckda_xsa", "3:1 CKDA hybrid, XSA in the attention layers",
         {"aux.pattern": hybrid(blocks), **CKDA, **xsa}),
        ("h_ckda_gate", "3:1 CKDA hybrid, gated attention",
         {"aux.pattern": hybrid(blocks), **CKDA, **gate}),
        ("h_ckda_std", "3:1 CKDA (reference: standard inits, low-rank gate) : attention hybrid",
         {"aux.pattern": hybrid(blocks), **CKDA_STD}),
        ("h_ckda_std_xsa", "3:1 CKDA (standard inits) hybrid, XSA in the attention layers",
         {"aux.pattern": hybrid(blocks), **CKDA_STD, **xsa}),
        ("h_tri_ckda", "3:1 triadic CKDA (E=4) : attention hybrid",
         {"aux.pattern": hybrid(blocks), **TRI_CKDA}),
        ("h_tri_ckda_xsa", "3:1 triadic CKDA (E=4) hybrid, XSA in the attention layers",
         {"aux.pattern": hybrid(blocks), **TRI_CKDA, **xsa}),
        ("h_kda", "3:1 KDA : attention hybrid", lin("kda")),
        ("h_kda_xsa", "3:1 KDA hybrid, XSA", {**lin("kda"), **xsa}),
        ("h_gdn", "3:1 Gated DeltaNet : attention hybrid", lin("gated_delta_net")),
        # the multilingual sweep's GDN arms (architecture_scaling_variants/multilingual/
        # gdn3_scaling_all.yaml, qwen35_scaling_all.yaml): GDN heads of 128 (k = query groups,
        # v = heads), NoPE attention; qwen35 adds the attention output gate, partial RoPE (25 %)
        # and zero-centred norm gamma (Qwen3.5 / Qwen3-Next dense block)
        # (screen2b ran both with fused_qk_norm_rope false; the fused path now covers NoPE and
        # zero-centred gamma, numerically the same)
        ("h_gdn_nope", "3:1 GDN (heads of 128) : NoPE attention hybrid (multilingual gdn3_nope)",
         {**lin("gated_delta_net"), **gdn128, B + "position_embedding_type": "none"}),
        ("h_gdn_qwen35", "3:1 GDN : gated attention, partial RoPE, zero-centred norm (Qwen3.5-like)",
         {**lin("gated_delta_net"), **gdn128, **gate, B + "position_embedding_type": "rope",
          B + "rotary_percent": 0.25, B + "layernorm_zero_centered_gamma": True}),
        ("h_mlstm", "3:1 mLSTM : attention hybrid", lin("mlstm")),
    ]


def arms3(blocks: int):
    """Level-1 round 3 (docs §10): the hybrid recipe -- NoPE attention with qk-norm, XSA and the
    output gate, XSA also in the linear (CKDA) layers -- and the residual / memory variants on it."""
    # layout (user 2026-10-17): head dim 128 everywhere, Qwen3.5-proportioned. At hidden 768:
    # attention 6 q heads x 128 with GQA (2 KV groups; Qwen3.5-4B: 4:1), linear mixers 6 heads x
    # 128 (width 768 = hidden); the Qwen3.5-like GDN 6 key / 12 value heads x 128 (Qwen3.5-4B:
    # key width 0.8x, value 1.6x hidden, v:k heads 2:1)
    geo = {B + "num_attention_heads": 6, B + "num_query_groups": 2, B + "kv_channels": 128,
           B + "linear_key_head_dim": 128, B + "linear_value_head_dim": 128,
           B + "linear_num_key_heads": 6, B + "linear_num_value_heads": 6}
    xsa = {B + "exclusive_self_attention": True}
    attn_h = {B + "position_embedding_type": "none", B + "attention_output_gate": True, **xsa}
    lxsa = {B + "linear_xsa": True}
    h3 = {"aux.pattern": hybrid(blocks), **geo, **CKDA_STD, **attn_h, **lxsa}
    nocg = {B + "cuda_graph_impl": "none"}
    mhc = {B + "mhc": True, B + "mhc_expansion_rate": 4}
    seed2 = {B + "seed": 1235}
    # (nGPT scales q / k by s_qk, which the fused q/k-norm + RoPE kernel does not do: off)
    ngpt = {B + "ngpt": True, B + "weight_normalizer": True, B + "fused_qk_norm_rope": False}
    qwen = {**[a for a in arms(blocks) if a[0] == "h_gdn_qwen35"][0][2], **geo,
            B + "linear_num_value_heads": 12}
    ref = {**geo, **xsa}
    return [
        ("attn_ref", "attention + qk-norm + XSA (RoPE): the reference transformer", ref),
        ("attn_ref_seed2", "attn_ref, seed 1235", {**ref, **seed2}),
        ("h3", "3:1 CKDA (std) + linear XSA; attention NoPE + qk-norm + XSA + gate", h3),
        ("h3_seed2", "h3, seed 1235", {**h3, **seed2}),
        # the short arms' paired references (added 2026-10-18): Megatron's data order depends on
        # train_iters, so the 4000-step arms see other batches than the 4BT validation arms
        # (per-step loss residual correlation -0.02 vs +1.00 paired); these two run the short
        # arms' exact schedule (4000 steps, seed 1234)
        ("attn_ref_4k", "attn_ref, 4000 steps: the short arms' paired reference", ref),
        ("h3_4k", "h3, 4000 steps: the short arms' paired h3", h3),
        ("h5", "5:1 CKDA hybrid, same recipe (2 attention layers: 6x less KV cache)",
         {**h3, "aux.pattern": hybrid(blocks, ratio=5)}),
        ("h3_tri", "3:1 triadic CKDA (E=4), same recipe", {**h3, **TRI_CKDA}),
        ("h3_nolxsa", "h3 without XSA in the linear layers (ablation)",
         {k: v for k, v in h3.items() if k != B + "linear_xsa"}),
        ("qwen35", "Qwen3.5-like GDN hybrid (gated attention, partial RoPE, zero-centred norm; "
         "GDN 6 key / 12 value heads x 128)", qwen),
        ("h3_mhc4", "h3 + mHC, 4 residual streams", {**h3, **mhc}),
        ("h3_routed", "h3 + routed residual, 2 of 4 streams", {**h3, **mhc, **nocg, B + "mhc_routed_topk": 2}),
        # routed-residual extensions (Megatron 931938bdf): the router also reads a 64-channel tag
        # of every stream (content-key routing; gain 0 at init = the positional router), and the
        # nGPT LERP write (needs ngpt: h3_ngpt is its baseline)
        ("h3_routed_key", "h3 + routed residual (2 of 4), content-key router over all streams",
         {**h3, **mhc, **nocg, B + "mhc_routed_topk": 2, B + "mhc_routed_key_dim": 64}),
        ("h3_ngpt", "h3 in nGPT form (normalized weights / hidden states, LERP residuals)",
         {**h3, **ngpt}),
        ("h3_routed_lerp", "h3_ngpt + routed residual (2 of 4) with the nGPT LERP write",
         {**h3, **ngpt, **mhc, **nocg, B + "mhc_routed_topk": 2, B + "mhc_lerp": True}),
        ("h3_routed_key_lerp", "h3_ngpt + routed residual, content-key router, LERP write",
         {**h3, **ngpt, **mhc, **nocg, B + "mhc_routed_topk": 2, B + "mhc_routed_key_dim": 64,
          B + "mhc_lerp": True}),
        ("h3_attnres", "h3 + block attention residuals (4 blocks)",
         {**h3, **nocg, B + "attn_res": True, B + "attn_res_num_blocks": 4}),
        ("h3_engram", "h3 + Engram n-gram memory before blocks 2 and 8 (tables not matched: sparse)",
         {**h3, "aux.pattern": hybrid(blocks, engram_at=(1, 7)), B + "engram_ngram_orders": [2, 3],
          B + "engram_table_parallel": "dp"}),
    ]


VALIDATION3 = ("attn_ref", "h3")
R3_KEY = "r3b:"  # param_match.json keys of round 3 (r3b: the head-dim-128 / GQA layout)
NOT_MATCHED = {"h3_engram": "h3"}  # Engram tables are sparse lookups (like embeddings): h3's d_ffn


HEADER = """# @package _global_
# GENERATED by scripts/korbi/campaign/gen_screen.py -- edit there.
# {title}, size {size}: hidden {hidden} x {blocks} blocks, {heads} heads
# of 64, d_ffn {ffn} (base), datamix4, lr {lr}, gbs {gbs} x 4096, 1000-step warmup, then constant
# (stable only, no cooldown) to aux.tokens. Arms: one feature each; d_ffn matched per arm ({matched}).
defaults:
  - base_train_mn5
  - _self_

aux:
  suite: {suite}
  family: {size}
  tokens: {tokens}
  gbs: {gbs}
  pattern: "{pattern}"

backend:
  megatron:
    hidden_size: {hidden}
    num_attention_heads: {heads}
    num_query_groups: {heads}
    kv_channels: 64
    ffn_hidden_size: {ffn}
    linear_num_key_heads: {heads}
    linear_num_value_heads: {heads}
    lr: {lr}
    min_lr: {lr}
    # stable only: no cooldown. Arms are compared on the stable-phase loss; at equal lr / gbs the
    # stable ranking is the cooled one (round 1: Spearman 0.98-1.00 over 4 screens of 9 arms).
    # Round-1 screens ran WSD (20 % linear cooldown to lr / 10).
    lr_decay_style: constant
    lr_wsd_decay_iters: null
    # 1000 warmup steps (the round-1 AdamW screen ran max(100, 1 %) = 100; angular MuOwn runs
    # override it to 0)
    lr_warmup_iters: 1000
    eval_interval: 500
    eval_iters: 20

job:
  array_group: ${{aux.suite}}
  est_step_time_s: {step_s}

slurm:
  sbatch:
    time: "{time}"

sweep:
  type: product
  store_sweep_json: true
  filter: "\\\\${{oc.eval:'len(\\\\${{aux.only}}) == 0 or \\"\\\\${{aux.arm}}\\" in \\\\${{aux.only}}'}}"
  groups:
    - type: list
      configs:
"""


# screen 2 (round 1b): the valid setup -- AdamW with a 1000-step warmup, stable only, gbs 128
# ("common value" at 4k sequence and ~120M), lr: s1 2e-3 (warmup study: 2e-3 / 4e-3 within 0.003
# at 1000 warmup steps), s2 scaled by the ComplexKDA law (N2/N1)^-0.417 -> 1.25e-3
SIZES2 = {
    "s1": (768, 12, 12, 2048, 2e-3, 128),
    "s2": (1024, 20, 16, 2816, 1.25e-3, 128),
}
# short-run protocol (docs §9.6): every arm to SHORT_STEPS, then compared through its fitted
# curve; the validation arms run to the full budget (tokens below) to check the extrapolation
SHORT_STEPS = 4000
VALIDATION_ARMS = ("attn", "h_ckda_std_xsa", "h_gdn_qwen35")
ARMS2 = ["attn", "attn_xsa", "attn_gate", "h_ckda_std", "h_ckda_std_xsa", "h_tri_ckda",
         "h_tri_ckda_xsa", "h_gdn", "h_gdn_nope", "h_gdn_qwen35"]


# round 3B (docs §10.3): MoE design on the round-3 attention base (attn_ref, head dim 128 / GQA),
# an MoE FFN in every block, ACTIVE FFN width matched to attn_ref's d_ffn 2048 by construction
# (top-k x expert width (+ shared) = 2048; the router adds 768 x E per layer). EP 4 on 1 node,
# the speed suite's MoE stack; per-layer router / expert-viability metrics to TB / wandb.
MOE = {B + "moe_grouped_gemm": True, B + "moe_token_dispatcher_type": "alltoall",
       B + "moe_router_dtype": "fp32", B + "moe_permute_fusion": True,
       B + "expert_model_parallel_size": 4, B + "moe_router_load_balancing_type": "aux_loss",
       B + "moe_aux_loss_coeff": 1e-3, B + "moe_per_layer_logging": True,
       B + "moe_expert_viability_metrics": True,
       # attention only in CUDA graphs: the per-layer router statistics are not graph-safe
       B + "cuda_graph_modules": ["attn"]}
# optimizers at their round-1 optima (gbs 128): AdamW 2e-3 with 1000 warmup steps, angular MuOwn
# 4e-3 without warmup
OPT3B = {"adamw": {"aux.lr": 2e-3, "aux.warmup_iters": 1000},
         "am": {"aux.lr": 4e-3, "aux.warmup_iters": 0, B + "optimizer": "angular_muown"}}


def moe(experts: int, width: int, topk: int, shared: int | None = None, **extra):
    return {B + "num_experts": experts, B + "moe_ffn_hidden_size": width, B + "moe_router_topk": topk,
            B + "moe_shared_expert_intermediate_size": shared, **MOE, **extra}


def arms3b(blocks: int):
    """(name, comment, overrides) per design; every one runs with both optimizers (OPT3B)."""
    ref = next(a for a in arms3(blocks) if a[0] == "attn_ref")[2]
    pat = "*E" * blocks
    base = {**ref, "aux.pattern": pat, **moe(64, 512, 4)}
    sig = {B + "moe_router_score_function": "sigmoid"}
    bias = {**sig, B + "moe_router_enable_expert_bias": True, B + "moe_router_bias_update_rate": 1e-3}
    return [
        ("dense", "attn_ref, dense FFN 2048 (the dense twin)", {**ref, B + "ffn_hidden_size": 2048}),
        ("e16k4", "16 x 512, top-4 (4x sparse)", {**base, **moe(16, 512, 4)}),
        ("e32k4", "32 x 512, top-4 (8x sparse)", {**base, **moe(32, 512, 4)}),
        ("e64k4", "64 x 512, top-4 (16x sparse): the base", base),
        ("e128k4", "128 x 512, top-4 (32x sparse)", {**base, **moe(128, 512, 4)}),
        ("e32k2w1024", "granularity: 32 x 1024, top-2 (16x sparse)", {**base, **moe(32, 1024, 2)}),
        ("e128k8w256", "granularity: 128 x 256, top-8 (16x sparse)", {**base, **moe(128, 256, 8)}),
        ("e64k3s512", "64 x 512, top-3 + shared expert 512", {**base, **moe(64, 512, 3, 512)}),
        ("aux1e-2", "base, aux loss 1e-2", {**base, B + "moe_aux_loss_coeff": 1e-2}),
        ("sigmoid", "base, sigmoid router (aux 1e-3): separates the score function from the bias",
         {**base, **sig}),
        ("bias", "base, aux-loss-free expert bias (sigmoid, rate 1e-3, no aux loss)",
         {**base, **bias, B + "moe_router_load_balancing_type": "none", B + "moe_aux_loss_coeff": 0.0}),
        ("bias_aux1e-4", "base, expert bias + aux loss 1e-4 (sigmoid)",
         {**base, **bias, B + "moe_aux_loss_coeff": 1e-4}),
        ("lr_half", "base at lr x 0.5", {**base, "lr_scale": 0.5}),
        ("lr_x2", "base at lr x 2", {**base, "lr_scale": 2.0}),
        ("gbs64", "base at gbs 64, equal tokens (8000 steps)", {**base, "aux.gbs": 64}),
        ("gbs256", "base at gbs 256, equal tokens (2000 steps)", {**base, "aux.gbs": 256}),
    ]


def arms3b_64x(blocks: int):
    """64x sparsity (user 2026-10-18: the 80B-A3B class, ~43x total / active at s1's attention
    share), angular MuOwn only, the ladder's router (softmax + aux 1e-3) so the sparsity step from
    e128k4 is clean; micro batch 4 (e128k4 used 94 % of the memory at 8; same gbs, same data
    order)."""
    base = dict(arms3b(blocks)[3][2])  # e64k4
    mbs4 = {B + "micro_batch_size": 4}
    return [
        ("e256k4", "256 x 512, top-4 (64x sparse)", {**base, **moe(256, 512, 4), **mbs4}),
        ("e512k8w256", "512 x 256, top-8 (64x sparse, fine)", {**base, **moe(512, 256, 8), **mbs4}),
    ]


HEADER3B = """# @package _global_
# GENERATED by scripts/korbi/campaign/gen_screen.py -- edit there.
# Round 3B (docs §10.3): MoE design on the round-3 attention base (attn_ref: hidden {hidden} x
# {blocks} blocks, 6 q heads x 128, GQA 2), an MoE FFN in every block with the active FFN width
# matched to the dense twin's 2048; sparsity, granularity, shared expert, router balancing, and
# lr / gbs, each with AdamW (2e-3, 1000 warmup) and angular MuOwn (4e-3, no warmup). Stable
# only, {steps} steps at gbs 128 (equal tokens {tokens:.3g} for every arm), every step logged.
defaults:
  - base_train_mn5
  - _self_

aux:
  suite: screen3b_s1
  family: s1
  tokens: {tokens}
  gbs: 128
  lr: 2.0e-3
  warmup_iters: 1000
  pattern: "{pattern}"

backend:
  megatron:
    hidden_size: {hidden}
    ffn_hidden_size: 2048
    lr: ${{aux.lr}}
    min_lr: ${{aux.lr}}
    lr_decay_style: constant
    lr_wsd_decay_iters: null
    lr_warmup_iters: ${{aux.warmup_iters}}
    log_interval: 1
    eval_interval: 500
    eval_iters: 20

job:
  array_group: ${{aux.suite}}
  est_step_time_s: 1.2

slurm:
  sbatch:
    time: "04:00:00"

sweep:
  type: product
  store_sweep_json: true
  filter: "\\\\${{oc.eval:'len(\\\\${{aux.only}}) == 0 or \\"\\\\${{aux.arm}}\\" in \\\\${{aux.only}}'}}"
  groups:
    - type: list
      configs:
"""


def write3b():
    hidden, blocks = SIZES2["s1"][:2]
    tokens = SHORT_STEPS * 128 * 4096
    lines = [HEADER3B.format(hidden=hidden, blocks=blocks, steps=SHORT_STEPS, tokens=tokens,
                             pattern="*-" * blocks)]
    n = 0
    for opt, okv in OPT3B.items():
        for name, comment, kv in arms3b(blocks) + (arms3b_64x(blocks) if opt == "am" else []):
            kv = {**okv, **kv}
            # literals (a sweep-entry interpolation resolves against the base config)
            kv["aux.lr"] = kv["aux.lr"] * kv.pop("lr_scale", 1.0)
            lines.append(f"        # {comment} [{opt}]\n        - aux.arm: {opt}_{name}\n")
            for k, v in kv.items():
                val = yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
                lines.append(f"          {k}: {val}\n")
            n += 1
    (OUT / "screen3b_s1_mn5.yaml").write_text("".join(lines))
    print(f"screen3b_s1_mn5.yaml: {n} arms")


# round 3C (docs §10.6, user 2026-10-18): the residual variants under angular MuOwn (warmup,
# mixer lr), routed 8 streams, mixed / pure-linear stacks with triadic `T` layers, nGPT's own lr
# sweep (4BT + cooldown), and the N trend at s2 (exit_interval protocol).
AM = {B + "optimizer": "angular_muown", B + "lr": 4e-3, B + "min_lr": 4e-3, B + "lr_warmup_iters": 0}
ADAMW = {B + "optimizer": "adam", B + "lr": 2e-3, B + "min_lr": 2e-3, B + "lr_warmup_iters": 1000}
R3C_KEY = "r3c:"  # param_match.json keys of the new round-3C stacks (s1); "r3c2:" at s2


def period6(*mixers: str) -> str:
    """12 blocks as 2 periods of the 6 given mixers (each followed by an MLP)."""
    assert len(mixers) == 6
    return "".join(f"{m}-" for m in mixers) * 2


def arms3c(blocks: int):
    """(name, comment, overrides, param_match key) for the s1 short suite (paired with round 3:
    4000 steps, seed 1234, gbs 128)."""
    r3 = {a[0]: a[2] for a in arms3(blocks)}
    h3, mhc4, routed, attnres = r3["h3"], r3["h3_mhc4"], r3["h3_routed"], r3["h3_attnres"]
    nolxsa = {k: v for k, v in h3.items() if k != B + "linear_xsa"}
    tri = {B + "linear_triadic_base": "ckda", B + "linear_triadic_e": 4}
    mlr = {B + "residual_mixer_lr_mult": 0.25}
    wu = {B + "lr_warmup_iters": 1000}
    out = []
    # angular MuOwn maturity: warmup, and the mixers' Adam lr (4e-3 x 0.25 = 1e-3)
    for name, base, key in (("h3", h3, "h3"), ("h3_mhc4", mhc4, "h3_mhc4"),
                            ("h3_routed", routed, "h3_routed"), ("h3_attnres", attnres, "h3_attnres")):
        out.append((f"{name}_wu1k", f"{name}, angular MuOwn with a 1000-step warmup",
                    {**base, **AM, **wu}, R3_KEY + key))
        if name != "h3":
            out.append((f"{name}_mlr025", f"{name}, angular MuOwn, residual-mixer lr x 0.25",
                        {**base, **AM, **mlr}, R3_KEY + key))
    # routed 8 streams, 2 active; mHC 8 as its dense-mixing counterpart
    r8 = {**routed, B + "mhc_expansion_rate": 8}
    out += [
        ("h3_routed8", "h3 + routed residual, 8 streams, 2 active (angular MuOwn)", {**r8, **AM},
         R3C_KEY + "h3_routed8"),
        ("adamw_h3_routed8", "h3 + routed residual, 8 streams, 2 active (AdamW)", {**r8, **ADAMW},
         R3C_KEY + "adamw_h3_routed8"),
        ("h3_mhc8", "h3 + mHC, 8 streams (angular MuOwn)", {**mhc4, B + "mhc_expansion_rate": 8, **AM},
         R3C_KEY + "h3_mhc8"),
    ]
    # mixed / pure-linear stacks (no linear XSA), angular MuOwn
    out += [
        ("h5_nolxsa", "attn 1 + CKDA 5 (reference)",
         {**nolxsa, "aux.pattern": hybrid(blocks, ratio=5), **AM}, R3C_KEY + "h5_nolxsa"),
        ("h5_tri1", "attn 1 + triadic CKDA 1 + CKDA 4",
         {**nolxsa, **tri, "aux.pattern": period6("L", "L", "T", "L", "L", "*"), **AM},
         R3C_KEY + "h5_tri1"),
        ("lin_ckda", "pure linear: CKDA 6",
         {**nolxsa, "aux.pattern": period6(*"LLLLLL"), **AM}, R3C_KEY + "lin_ckda"),
        ("lin_tri1_e4", "pure linear: triadic CKDA 1 (E = 4) + CKDA 5",
         {**nolxsa, **tri, "aux.pattern": period6("L", "L", "T", "L", "L", "L"), **AM},
         R3C_KEY + "lin_tri1_e4"),
        ("lin_tri1_e8", "pure linear: triadic CKDA 1 (E = 8) + CKDA 5",
         {**nolxsa, **tri, B + "linear_triadic_e": 8,
          "aux.pattern": period6("L", "L", "T", "L", "L", "L"), **AM}, R3C_KEY + "lin_tri1_e8"),
    ]
    return out


def arms3c_ngpt(blocks: int):
    """nGPT's own lr sweep (AdamW, no warmup) to 4BT with a 20 % linear WSD cooldown to lr / 10,
    next to h3 (no linear XSA) with the same schedule under both optimizers: final cooled losses
    (nGPT's curve crosses the others twice, so stable short runs cannot judge it)."""
    r3 = {a[0]: a[2] for a in arms3(blocks)}
    h3 = {k: v for k, v in r3["h3"].items() if k != B + "linear_xsa"}
    ngpt = {B + "ngpt": True, B + "weight_normalizer": True, B + "fused_qk_norm_rope": False}
    steps = int(4.0e9) // (128 * 4096)
    wsd = {B + "lr_decay_style": "WSD", B + "lr_wsd_decay_style": "linear",
           B + "lr_wsd_decay_iters": int(0.2 * steps)}
    out = []
    for lr in (4e-3, 8e-3, 1.6e-2):
        out.append((f"ngpt_lr{lr:g}", f"h3 (no linear XSA) in nGPT form, AdamW lr {lr:g}, no warmup, WSD",
                    {**h3, **ngpt, **ADAMW, B + "lr": lr, B + "min_lr": lr / 10, B + "lr_warmup_iters": 0,
                     **wsd}, R3C_KEY + f"ngpt_lr{lr:g}"))
    out += [
        ("adamw_h3", "h3 (no linear XSA), AdamW 2e-3, 1000 warmup, WSD",
         {**h3, **ADAMW, B + "min_lr": 2e-4, **wsd}, R3C_KEY + "adamw_h3"),
        ("am_h3", "h3 (no linear XSA), angular MuOwn 4e-3, no warmup, WSD",
         {**h3, **AM, B + "min_lr": 4e-4, **wsd}, R3C_KEY + "am_h3"),
    ]
    return out


def arms3c_s2(blocks: int):
    """The N trend at s2 (hidden 1024 x 20 blocks, head dim 128: 8 q heads / 2 KV groups, CKDA
    8 heads), angular MuOwn, no linear XSA. exit_interval protocol: train_iters = 4BT for every arm,
    exit at 4000, so short and full arms share the data order and winners extend by resuming."""
    geo = {B + "num_attention_heads": 8, B + "num_query_groups": 2, B + "kv_channels": 128,
           B + "linear_key_head_dim": 128, B + "linear_value_head_dim": 128,
           B + "linear_num_key_heads": 8, B + "linear_num_value_heads": 8}
    # the round-3 recipes (built at s1's 12 blocks: arms3 also builds 5:1, which 20 blocks do
    # not divide into), re-patterned for ``blocks``
    r3 = {a[0]: a[2] for a in arms3(SIZES2["s1"][1])}
    pat = {"attn_ref": "*-" * blocks}

    def strip(name):
        d = {k: v for k, v in r3[name].items() if k != B + "linear_xsa"}
        d["aux.pattern"] = pat.get(name, hybrid(blocks))
        return d

    ex = {B + "exit_interval": SHORT_STEPS, "job.est_steps": SHORT_STEPS}
    return [
        ("attn_ref", "attention + qk-norm + XSA (RoPE)", {**strip("attn_ref"), **geo, **AM, **ex},
         "r3c2:attn_ref"),
        ("h3", "CKDA 3:1, no linear XSA", {**strip("h3"), **geo, **AM, **ex}, "r3c2:h3"),
        # 4 residual streams OOM at mbs 8 (s2 smoke, 2026-10-18): mbs 4, same gbs and data order
        ("h3_mhc4", "h3 + mHC-4", {**strip("h3_mhc4"), **geo, **AM, **ex, B + "micro_batch_size": 4},
         "r3c2:h3_mhc4"),
        ("h3_routed", "h3 + routed residual (2 of 4)",
         {**strip("h3_routed"), **geo, **AM, **ex, B + "micro_batch_size": 4}, "r3c2:h3_routed"),
    ]


def write3c(pm):
    hidden1, blocks1 = SIZES2["s1"][:2]
    hidden2, blocks2, _, ffn2, _, _ = SIZES2["s2"]
    for fname, suite, size, (hidden, blocks, ffn), tokens, pool, title in (
        ("screen3c_s1_mn5.yaml", "screen3c_s1", "s1", (hidden1, blocks1, 2048),
         SHORT_STEPS * 128 * 4096, arms3c(blocks1),
         "Round 3C: residual variants under angular MuOwn, routed-8, mixed / pure-linear stacks"),
        ("screen3c_ngpt_s1_mn5.yaml", "screen3c_ngpt_s1", "s1", (hidden1, blocks1, 2048), "4.0e9",
         arms3c_ngpt(blocks1), "Round 3C: nGPT lr sweep to 4BT with a cooldown, h3 references"),
        ("screen3c_s2_mn5.yaml", "screen3c_s2", "s2", (hidden2, blocks2, ffn2), "4.0e9",
         arms3c_s2(blocks2), "Round 3C: N trend at s2 (exit_interval 4000 of 4BT)"),
    ):
        lines = [HEADER.format(title=title, suite=suite, size=size, hidden=hidden, tokens=tokens,
                               blocks=blocks, heads=12 if size == "s1" else 16, ffn=ffn, lr=2e-3,
                               gbs=128, pattern="*-" * blocks, matched="per arm (param_match.json)",
                               time="10:00:00", step_s=0.6 if size == "s1" else 1.6)]
        unmatched = []
        for name, comment, kv, key in pool:
            kv = dict(kv)
            entry = pm.get(size, {}).get(key)
            if entry:
                kv[B + "ffn_hidden_size"] = entry["ffn"]
                comment += f" (d_ffn {entry['ffn']}, non-emb {entry['non_emb'] / 1e6:.1f}M)"
            else:
                unmatched.append(name)
                comment += " (d_ffn not matched yet)"
            lines.append(f"        # {comment}\n        - aux.arm: {name}\n")
            for k, v in kv.items():
                val = yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
                lines.append(f"          {k}: {val}\n")
        (OUT / fname).write_text("".join(lines))
        print(f"{fname}: {len(pool)} arms" + (f", NOT matched: {' '.join(unmatched)}" if unmatched else ""))


# round 4 (docs §11, user 2026-10-18): the MoE hybrid with a residual stream, plus LatentMoE.
# Base: round-3B MoE recipe at 48x (144 routed experts, top-3 + 1 shared expert of the expert
# width: the 80-90B-A3B class; 64x OOMed at s2 on one node), sigmoid
# router + expert bias + aux 1e-4), CKDA 5:1 (reference CKDA, no linear XSA), attention NoPE +
# qk-norm + XSA + gate; angular MuOwn without warmup, matrix scale 0.2, residual mixers at lr x
# 0.25 (§10.7). exit_interval protocol: train_iters = 4BT for every arm, exit at 4000 (equal data
# order, finalists extend by resuming).
R4_SIZES = {  # hidden, blocks, q heads (x 128, 2 KV groups), active FFN (= dense twin), lr
    "s1": (768, 12, 6, 2048, 4e-3),
    "s2": (1024, 18, 8, 2816, 2.5e-3),  # 18 blocks: 5:1 needs blocks % 6 == 0
    # round 5 only: ~0.5B active / ~12B total non-embedding; lr by the (N3/N2)^-0.417 law
    "s3": (1280, 24, 10, 3584, 1.8e-3),
}


def arms4(size: str):
    hidden, blocks, heads, ffn, lr = R4_SIZES[size]
    w = ffn // 4  # expert width: top-3 routed + 1 shared = the dense twin's active FFN
    geo = {B + "num_attention_heads": heads, B + "num_query_groups": 2, B + "kv_channels": 128,
           B + "linear_key_head_dim": 128, B + "linear_value_head_dim": 128,
           B + "linear_num_key_heads": heads, B + "linear_num_value_heads": heads}
    opt = {B + "optimizer": "angular_muown", B + "lr": lr, B + "min_lr": lr,
           B + "lr_warmup_iters": 0}
    ex = {B + "exit_interval": SHORT_STEPS, "job.est_steps": SHORT_STEPS}
    router = {B + "moe_router_score_function": "sigmoid", B + "moe_router_enable_expert_bias": True,
              B + "moe_router_bias_update_rate": 1e-3, B + "moe_aux_loss_coeff": 1e-4,
              # mbs 2 (gbs and data order unchanged): mbs 4 OOMed (no activation recompute)
              B + "micro_batch_size": 2}
    base_moe = {**moe(144, w, 3, w), **router}
    attn_pat = "*E" * blocks
    hyb_pat = hybrid(blocks, ratio=5).replace("-", "E")
    attn = {**geo, B + "exclusive_self_attention": True, "aux.pattern": attn_pat}
    h5 = {**geo, **CKDA_STD, B + "position_embedding_type": "none",
          B + "attention_output_gate": True, B + "exclusive_self_attention": True,
          "aux.pattern": hyb_pat}
    mlr = {B + "residual_mixer_lr_mult": 0.25}
    nocg = {B + "cuda_graph_impl": "none"}
    routed = lambda n, **kw: {B + "mhc": True, B + "mhc_expansion_rate": n,  # noqa: E731
                              B + "mhc_routed_topk": 2, **nocg, **mlr, **kw}
    mhc4 = {B + "mhc": True, B + "mhc_expansion_rate": 4, **nocg, **mlr}
    common = {**opt, **ex}
    out = [
        ("attn_moe", "all-attention MoE (RoPE + qk-norm + XSA): the reference", {**attn, **base_moe, **common}),
        ("h5_moe", "CKDA 5:1 MoE", {**h5, **base_moe, **common}),
        ("h5_moe_routed4", "+ routed residual, 2 of 4 streams", {**h5, **base_moe, **common, **routed(4)}),
        ("h5_moe_routed8", "+ routed residual, 2 of 8 streams, reversible",
         {**h5, **base_moe, **common, **routed(8, **{B + "mhc_routed_reversible": True})}),
        ("h5_moe_mhc4", "+ mHC, 4 streams", {**h5, **base_moe, **common, **mhc4}),
        # the routing control (user 2026-10-18): dense mHC over 2 streams touches as many
        # streams per sublayer as routed 2-of-n (n = 4 / 8 held); if it matches them, the
        # routing adds nothing beyond 2 active streams
        ("h5_moe_mhc2", "+ mHC, 2 streams (the routing control)",
         {**h5, **base_moe, **common, **mhc4, B + "mhc_expansion_rate": 2}),
    ]
    if size == "s1":
        lat = hidden // 2
        out += [
            ("h5_moe_routed8_mlr0125", "routed 2 of 8, mixer lr x 0.125",
             {**h5, **base_moe, **common, **routed(8, **{B + "mhc_routed_reversible": True}),
              B + "residual_mixer_lr_mult": 0.125}),
            ("h5_moe_routed8_mlr05", "routed 2 of 8, mixer lr x 0.5",
             {**h5, **base_moe, **common, **routed(8, **{B + "mhc_routed_reversible": True}),
              B + "residual_mixer_lr_mult": 0.5}),
            ("h5_moe_routed8_k3", "routed 3 of 8 (stream 0 + 2 others: each stream about as "
             "active as in 2 of 4), reversible",
             {**h5, **base_moe, **common, **routed(8, **{B + "mhc_routed_reversible": True}),
              B + "mhc_routed_topk": 3}),
            ("h5_moe_routed8_key", "routed 2 of 8, content-key router (64 channels)",
             {**h5, **base_moe, **common, **routed(8, **{B + "mhc_routed_key_dim": 64})}),
            # LatentMoE (moe_latent_size): routed experts in a latent of hidden / 2 (the shared
            # expert stays at full width). lat2: 2x the experts and the top-k -> equal active
            # expert compute, total expert parameters and dispatch volume; lat2_half: same
            # experts / top-k -> half of each
            ("h5_moe_lat2", f"LatentMoE, latent {lat}: 288 x {w}, top-6 + shared (equal cost)",
             {**h5, **moe(288, w, 6, w), **router, **common, B + "moe_latent_size": lat}),
            ("h5_moe_lat2_half", f"LatentMoE, latent {lat}: 144 x {w}, top-3 + shared (half cost)",
             {**h5, **base_moe, **common, B + "moe_latent_size": lat}),
        ]
    return out


def arms4g(size: str):
    """Round 4G, MoE granularity (user 2026-10-18): h5_moe (144 x w, top-3 + shared) with the
    experts split finer at fixed sparsity (E / k = 48), fixed active and total expert parameters
    (E x w and k x w constant) and the same shared expert -- only the expert width and the top-k
    change. Granularity 4 d / w_expert: s1 h5_moe 6, g2 12, g4 24 (Qwen3-30B-A3B ~11,
    Qwen3-Next-80B-A3B 16, DeepSeek-V3 14). Pairs with screen4's h5_moe (same data order)."""
    hidden, blocks, heads, ffn, lr = R4_SIZES[size]
    w = ffn // 4
    h5_moe = dict((n, kv) for n, _, kv in arms4(size))["h5_moe"]
    out = []
    for g in (2, 4):
        out.append((f"h5_moe_g{g}", f"granularity x{g}: {144 * g} x {w // g}, top-{3 * g} + shared {w}",
                    {**h5_moe, B + "num_experts": 144 * g, B + "moe_ffn_hidden_size": w // g,
                     B + "moe_router_topk": 3 * g}))
    return out


# The flag mix at 8k with document masking (data_flag_mn5.yaml's keys; the base config composes
# datamix4, so its per-split file is switched off here)
FLAG8K = {B + "per_split_data_args_path": None,
          B + "data_args_path": "/gpfs/projects/ehpc533/users/korbip/hybrid_v019/data/flag_datamix_mn5.txt",
          B + "split": "998,2,0", B + "seq_length": 8192, B + "max_position_embeddings": 8192,
          B + "dataloader_inter_document_masking": True,
          # 8 workers, prefetch 2 (round 4's): 16 x prefetch 8 (the production GPFS fix) put ~400 GB
          # of host memory into 64 workers per node; the checkpoint save then OOM-killed s2 nodes
          # (4 per rank since 2026-10-19 noon: each worker's own heap on top of the ~36 GB index
          # each rank shares copy-on-write with its workers; s2 nodes ran at ~405 of 503 GB)
          B + "num_workers": 4, B + "dataloader_prefetch_factor": 2,
          # manual GC every 10 steps + no mmap of the .bin files (exp_irene's settings): the
          # doc-masked hybrid is host-bound and Python-GC stalls cost 16 % (EP agent k1, 8 nodes)
          B + "manual_gc": True, B + "manual_gc_interval": 10, B + "mmap_bin_files": False}


def flag_smoke(size: str):
    """Smoke of the flag mix at 8k with document masking (user 2026-10-18): the round-4 best bet
    (CKDA 5:1 + routed 2 of 8 + MoE) for 200 steps at mbs 1 (= mbs 2's tokens at 4k)."""
    arm = dict((n, kv) for n, _, kv in arms4(size))["h5_moe_routed8"]
    return [("flag8k_h5_moe_routed8", "flag mix, 8k, document masking: routed 2 of 8, reversible",
             {**arm, **FLAG8K, B + "micro_batch_size": 1, B + "exit_interval": 200,
              "job.est_steps": 200})]


# Rounds 4R / 4R-x / 4R-y restarted with CUDA graphs (user 2026-10-18): per-layer graphs on the
# attention / mixer layers and the MoE router + preprocess graphs; the routing statistics stay on
# (graph-safe: arch_metrics graph mode, the MoE trackers record inside the router graph). The arms
# resumed from their step-1000 checkpoints; graphed and eager runs agree within the run-to-run
# noise of these MoE runs (§11.7).
CG4 = {B + "cuda_graph_impl": "local", B + "cuda_graph_modules": ["attn", "moe_router", "moe_preprocess"]}


def _graphed(arms):
    return [(n, c, {**kv, **CG4}) for n, c, kv in arms]


def arms4r(size: str):
    """Round 4R (user 2026-10-18): the residual arms again with the residual-stream diagnostics
    on (stream sizes per layer, read / pick shares, write strengths; every 25 steps), to find
    why dense mHC diverged (mHC-4 NaN at step 24 / 32, mHC-2 at 1935 under MoE) and why plain
    routed 2-of-8 gained nothing while its content-keyed and 3-of-8 variants did; plus keyed
    3-of-8. Same protocol and data order as round 4 (pairs with its runs)."""
    pool = dict((n, kv) for n, _, kv in arms4(size))
    diag = {B + "arch_metrics_interval": 25}
    key = {B + "mhc_routed_key_dim": 64}
    out = [
        ("h5_moe_mhc4_diag", "mHC-4 (diverged at step 24 in round 4)", {**pool["h5_moe_mhc4"], **diag}),
        ("h5_moe_mhc2_diag", "mHC-2 (diverged at step 1935 in round 4)", {**pool["h5_moe_mhc2"], **diag}),
        ("h5_moe_routed8_diag", "routed 2 of 8, reversible (gained ~0 in round 4)",
         {**pool["h5_moe_routed8"], **diag}),
        ("h5_moe_routed8_key_diag", "routed 2 of 8, content-key router", {**pool["h5_moe_routed8_key"], **diag}),
        ("h5_moe_routed8_k3_key", "routed 3 of 8, content-key router (new)",
         {**pool["h5_moe_routed8_key"], B + "mhc_routed_topk": 3, **diag}),
    ]
    return _graphed(out)


def arms4rx(size: str):
    """Round 4R extension (user 2026-10-18). s2: does the residual gain grow with size
    (mHC-2, keyed 2-of-8 and 3-of-8, paired with screen4_s2's h5_moe). s1: what could make the
    routed residual pay -- mixing the picked streams (mhc_routed_mix), balancing the picks
    (mhc_routed_balance_rate), the seed noise of a gap (seed 1235 changes init AND data order: the
    two seed-2 arms pair with each other), and its speed with per-layer CUDA graphs on the
    attention layers as the h5_moe reference has (round 4's residual arms ran without graphs)."""
    pool = dict((n, kv) for n, _, kv in arms4(size))
    diag = {B + "arch_metrics_interval": 25}
    key8 = {**pool["h5_moe_routed8"], B + "mhc_routed_reversible": False, B + "mhc_routed_key_dim": 64}
    k3 = {**key8, B + "mhc_routed_topk": 3}
    if size == "s2":
        return _graphed([
            ("h5_moe_mhc2_diag", "mHC-2 (the s2 leader in round 4)", {**pool["h5_moe_mhc2"], **diag}),
            ("h5_moe_routed8_key_diag", "routed 2 of 8, content-key router", {**key8, **diag}),
            ("h5_moe_routed8_k3_key", "routed 3 of 8, content-key router", {**k3, **diag}),
        ])
    seed2 = {B + "seed": 1235}
    return _graphed([
        ("h5_moe_routed8_k3_key_mix", "routed 3 of 8, keyed, + mixing of the picked streams",
         {**k3, B + "mhc_routed_mix": True, **diag}),
        ("h5_moe_routed8_k3_key_bal", "routed 3 of 8, keyed, + pick balancing (bias rate 1e-3)",
         {**k3, B + "mhc_routed_balance_rate": 1e-3, **diag}),
        ("h5_moe_seed2", "h5_moe, seed 1235 (pairs with h5_moe_routed8_k3_key_seed2 only)",
         {**pool["h5_moe"], **seed2}),
        ("h5_moe_routed8_k3_key_seed2", "routed 3 of 8, keyed, seed 1235", {**k3, **seed2}),
        ("h5_moe_routed8_k3_key_cg", "routed 3 of 8, keyed, CUDA graphs on attn (speed only, 300 steps)",
         {**k3, B + "cuda_graph_impl": "local", B + "cuda_graph_modules": ["attn"],
          B + "exit_interval": 300, "job.est_steps": 300}),
    ])


def arms4ry(size: str):
    """Round 4R-y (user 2026-10-18): the routed residual as the CHEAP scaling axis -- 2 of 8 and
    1 of 8 (each sublayer moves 2k + 1 rows of stream state), keyed, with pick balancing; mixing
    of the picked streams; free routing (mhc_routed_free: all k picked by score, no forced main
    stream; at k = 1 one of the 8 streams per sublayer, learning through the straight-through
    write gate). Same protocol as round 4 (pairs with its h5_moe and routed arms)."""
    pool = dict((n, kv) for n, _, kv in arms4(size))
    diag = {B + "arch_metrics_interval": 25}
    key2 = {**pool["h5_moe_routed8"], B + "mhc_routed_reversible": False,
            B + "mhc_routed_key_dim": 64, B + "mhc_routed_balance_rate": 1e-3, **diag}
    free = {B + "mhc_routed_free": True}
    return _graphed([
        ("h5_moe_r8k2_key_bal", "routed 2 of 8, keyed, pick balancing", key2),
        ("h5_moe_r8k2_key_mix_bal", "routed 2 of 8, keyed, balancing, mixing of the picked streams",
         {**key2, B + "mhc_routed_mix": True}),
        ("h5_moe_r8k2_free_key_bal", "routed 2 of 8, free (no forced main stream), keyed, balancing",
         {**key2, **free}),
        ("h5_moe_r8k1_free_key_bal", "routed 1 of 8, free, keyed, balancing (the cheapest)",
         {**key2, **free, B + "mhc_routed_topk": 1}),
    ])


def hybrid_every(blocks: int, period: int, mixer: str) -> str:
    """Attention every ``period``-th block, ``mixer`` elsewhere, MoE after each (a remainder of
    linear blocks at the end when ``period`` does not divide ``blocks``)."""
    return "".join(("*" if i % period == period - 1 else mixer) + "E" for i in range(blocks))


R5_CACHE = "/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/data_cache_flag8k_r5"
# seed-1235 arms: no data_cache_path in the entry -> base_train's per-run cache (another order;
# two of them would race on a shared one)
own_cache = lambda kv: {k: v for k, v in kv.items() if k != B + "data_cache_path"}  # noqa: E731
# document masking hands TE a packed batch: RoPE per document is TE's unfused-path job, the
# fused q/k-norm + RoPE kernel rejects packed batches with RoPE
NOFUSED_ROPE = {B + "fused_qk_norm_rope": False}


def arms5(size: str):
    """Round 5 (docs §12): the architecture decision before the two main runs (a Qwen3.5-35B-A3B
    replica and our 2-3B-active hybrid), on the production data setting: the flag mix at 8k with
    document masking, CUDA graphs (attention + MoE router) with the routing statistics, angular
    MuOwn without warmup, stable only, exit at 4000 steps of a 4BT schedule (gbs 64 x 8k = round
    4's token batch). Every arm at equal active and total MoE parameters (144 x w, top-3 + shared
    w), mixers / residuals / MoE shape varied one at a time. CUDA graphs on the MoE router
    only (document masking: no static capture of the attention / mixer layers)."""
    hidden, blocks, heads, ffn, lr = R4_SIZES[size]
    pool = dict((n, kv) for n, _, kv in arms4(size))
    # literals (sweep-entry values do not reach the base config's train_iters interpolation):
    # 4BT schedule at 64 x 8192 = 7629 steps, exit at 4000 (round 4's token batch and protocol)
    # CUDA graphs on the MoE router / preprocess only: document masking hands the attention /
    # mixer layers a cu_seqlens whose length changes every micro-batch (no static capture; fla's
    # varlen kernels also read it on the host); the router sees a fixed [t, 1, h]
    cg5 = {B + "cuda_graph_impl": "local", B + "cuda_graph_modules": ["moe_router", "moe_preprocess"]}
    # one dataset index for every seed-1234 arm of every size (same schedule, seed, seq, split):
    # built once (screen5 prebuild), then only read -- concurrent BUILDS of one cache race, and
    # each run building its own index of the 452-prefix flag mix took 12-45 min at startup
    base = {**FLAG8K, "aux.seq": 8192, B + "micro_batch_size": 1, B + "global_batch_size": 64,
            "aux.gbs": 64, B + "train_iters": 7629, **cg5, B + "data_cache_path": R5_CACHE}
    h5 = {**pool["h5_moe"], **base}
    mlr = {B + "residual_mixer_lr_mult": 0.25}
    mhc = lambda n: {B + "mhc": True, B + "mhc_expansion_rate": n, **mlr}  # noqa: E731
    rr = {B + "mhc": True, B + "mhc_expansion_rate": 8, B + "mhc_routed_topk": 2,
          B + "mhc_routed_key_dim": 64, B + "mhc_routed_balance_rate": 1e-3, **mlr}
    w = ffn // 4
    # Qwen3.5-like: Gated DeltaNet 3:1, gated attention with partial RoPE, zero-centred norm,
    # GDN 128-wide heads (k = heads, v = 2 x heads) -- same MoE as ours
    qwen = {**{k: v for k, v in h5.items() if not k.startswith(B + "linear_")},
            "aux.pattern": hybrid_every(blocks, 4, "L"),
            B + "experimental_attention_variant": "gated_delta_net",
            B + "linear_key_head_dim": 128, B + "linear_value_head_dim": 128,
            B + "linear_num_key_heads": heads, B + "linear_num_value_heads": 2 * heads,
            B + "attention_output_gate": True, B + "exclusive_self_attention": False,
            B + "position_embedding_type": "rope", B + "rotary_percent": 0.25,
            B + "layernorm_zero_centered_gamma": True}
    out = [
        ("attn_moe", "all-attention MoE (RoPE + qk-norm + XSA): is the hybrid worth it at 8k",
         {**pool["attn_moe"], **base, **NOFUSED_ROPE}),
        ("qwen35_moe", "Qwen3.5-like GDN 3:1 + gated attention, same MoE: main run A's proxy",
         {**qwen, **NOFUSED_ROPE}),
        ("h5_moe", "ours: CKDA 5:1 + NoPE / qk-norm / XSA / gated attention, MoE", h5),
        ("h5_moe_mhc4", "+ mHC, 4 streams (DeepSeek's)", {**h5, **mhc(4)}),
        ("h5_moe_mhc2", "+ mHC, 2 streams", {**h5, **mhc(2)}),
        ("h5_moe_rr28", "+ routed residual 1 + 1 of 8, keyed, pick balancing", {**h5, **rr}),
        ("h5_moe_attnres", "+ block attention residuals (4 blocks; no CUDA graphs: not verified)",
         {**h5, B + "attn_res": True, B + "attn_res_num_blocks": 4,
          B + "cuda_graph_impl": "none", B + "cuda_graph_modules": []}),
        ("h5_moe_g2", f"MoE granularity x2: {144 * 2} x {w // 2}, top-6 + shared {w}",
         {**h5, B + "num_experts": 288, B + "moe_ffn_hidden_size": w // 2, B + "moe_router_topk": 6}),
        # candidates from round 4 (user 2026-10-19): the routed residual with mixing of the picked
        # streams; LatentMoE at equal cost (round 4 s1: best 10BT extrapolation, one seed); both
        ("h5_moe_rr28_mix", "+ routed residual 1 + 1 of 8, keyed, balancing, mixing of the picked",
         {**h5, **rr, B + "mhc_routed_mix": True}),
        ("h5_moe_lat2", f"LatentMoE, latent {hidden // 2}: 288 x {w}, top-6 + shared (equal cost)",
         {**h5, B + "num_experts": 288, B + "moe_router_topk": 6, B + "moe_latent_size": hidden // 2}),
        ("h5_moe_rr28_mix_lat2", "routed residual (keyed, balancing, mixing) + LatentMoE",
         {**h5, **rr, B + "mhc_routed_mix": True, B + "num_experts": 288, B + "moe_router_topk": 6,
          B + "moe_latent_size": hidden // 2}),
    ]
    if size == "s1":
        # after round 4 (user 2026-10-19): mHC is the residual; does 8 streams add over 4, and do
        # the two winners (mHC-4, LatentMoE) combine? (the routed-residual arms are called off)
        out.append(("h5_moe_mhc8", "+ mHC, 8 streams", {**h5, **mhc(8)}))
        out.append(("h5_moe_mhc4_lat2", f"mHC-4 + LatentMoE, latent {hidden // 2}: 288 x {w}, top-6",
                    {**h5, **mhc(4), B + "num_experts": 288, B + "moe_router_topk": 6,
                     B + "moe_latent_size": hidden // 2}))
        # Linear Depth Memory (user 2026-10-19; arXiv:2610.18809, ELLIS / MPI Tübingen): next to
        # the plain residual, a per-token matrix memory of the sublayer outputs over depth, fit by
        # recursive least squares; the paper's setting G 16, K 8, address width 32 (400M), Adam at
        # the residual-mixer lr (0.25 x 4e-3 = the paper's 1e-3). Reference formulation (autograd
        # keeps every write's K x h fp32 state): no CUDA graphs, s1 only
        out.append(("h5_moe_ldm", "+ Linear Depth Memory (G 16, K 8, H 32)",
                    {**h5, B + "depth_memory": True, B + "depth_memory_groups": 16,
                     B + "depth_memory_key_dim": 8, B + "depth_memory_hidden_dim": 32, **mlr,
                     B + "cuda_graph_impl": "none", B + "cuda_graph_modules": []}))
        out.append(("h5_moe_seed2", "h5_moe, seed 1235 (noise: pairs with h5_moe_rr28_seed2)",
                    own_cache({**h5, B + "seed": 1235})))
        out.append(("h5_moe_rr28_seed2", "h5_moe_rr28, seed 1235",
                    own_cache({**h5, **rr, B + "seed": 1235})))
    if size == "s2":
        # compact angular-MuOwn row states in the checkpoints (Megatron: ANGULAR_MUOWN_COMPACT_CKPT):
        # the weight-shaped form cost ~76 GB of host staging per s2 save and OOM-killed the nodes
        # three times; s2 restarted from step 0 with it (a run keeps one format throughout)
        out = [(n, c, {**kv, "backend.env.ANGULAR_MUOWN_COMPACT_CKPT": "1"}) for n, c, kv in out]
    if size == "s3":
        # 4 nodes, 144 experts over EP16 (9 per GPU; all-to-all across nodes, HybridEP is
        # intra-node only). 2 nodes / EP8 OOMed at step 1 (54 GB: 1.5B expert parameters per GPU
        # with unsharded optimizer state + the 8k activations)
        nodes = {"slurm.sbatch.nodes": 4, B + "expert_model_parallel_size": 16}
        out = [(n, c, {**kv, **nodes}) for n, c, kv in out
               if n in ("qwen35_moe", "h5_moe", "h5_moe_mhc4", "h5_moe_rr28")]
    return out


# Engram allocation sweep (user 2026-10-19; docs §12.2): Cheng et al. 2026 find a U-shaped loss
# over the split of the INACTIVE parameters between routed experts and Engram tables, best at
# ~75-80 % experts (20-25 % Engram), at fixed total and active parameters. Here the round-5 MoE
# hybrid (h5_moe: 144 x w, top-3 + shared) gives up routed experts for tables of the same size,
# 0 (= h5_moe of round 5, same data order) / 11 / 20 / 34 % of the inactive expert parameters,
# with the reference code's recipe: Engram before the 2nd and the middle block, 2- and 3-grams,
# 8 hash heads x 64 per order, per-layer hashes, the signed-sqrt gate, the pre-normed conv,
# DeepSeek's tokenizer compression (scripts/korbi/campaign/engram_token_map.py), Adam at 5 x lr.
ENGRAM_TOKEN_MAP = "/gpfs/projects/ehpc533/users/korbip/hybrid_v019/data/engram_token_map_256k.npy"
ENGRAM_ORDERS, ENGRAM_HEADS, ENGRAM_DIM, ENGRAM_LAYERS = (2, 3), 8, 64, 2
# the lean table optimizer: one second moment per row + bf16 momentum (10 instead of 16 B per
# table parameter on the GPU, with the fp32 table and gradient buffer)
ENGRAM_LEAN = {B + "engram_table_rowwise_v": True, B + "engram_table_momentum_dtype": "bf16"}


def engram_arm(size: str, experts: int, **extra):
    """h5_moe with ``experts`` routed experts (of 144) and the freed expert parameters in two
    Engram layers: (overrides, Engram table parameters, share of the inactive parameters)."""
    hidden, blocks, heads, ffn, lr = R4_SIZES[size]
    w = ffn // 4
    per_expert = 3 * hidden * w  # SwiGLU: fc1 (2 w x h) + fc2 (w x h)
    table_params = (144 - experts) * per_expert * blocks  # one MoE per block
    lookups = len(ENGRAM_ORDERS) * ENGRAM_HEADS
    rows = table_params // (ENGRAM_LAYERS * lookups * ENGRAM_DIM)  # per hash table
    kv = {"aux.pattern": hybrid(blocks, ratio=5, engram_at=(1, blocks // 2)).replace("-", "E"),
          B + "num_experts": experts,
          B + "engram_ngram_orders": list(ENGRAM_ORDERS), B + "engram_num_heads": ENGRAM_HEADS,
          B + "engram_embed_dim": ENGRAM_DIM, B + "engram_table_size": rows,
          B + "engram_table_parallel": "dp", B + "engram_token_map": ENGRAM_TOKEN_MAP,
          B + "engram_gate_signed_sqrt": True, B + "engram_conv_prenorm": True,
          B + "engram_hash_per_layer": True, B + "engram_lr_mult": 5.0, **extra}
    return kv, table_params, (144 - experts) / (144 - 3)


def arms5e(size: str):
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    grid = {"s1": [(128, "eng11", {}), (116, "eng20", {}), (96, "eng34", {}),
                   (116, "eng20_lean", ENGRAM_LEAN),
                   (116, "eng20_lrx2p5", {B + "engram_lr_mult": 2.5})],
            "s2": [(116, "eng20", {})]}[size]
    out = []
    for experts, tag, extra in grid:
        kv, table, share = engram_arm(size, experts, **extra)
        note = {"eng20_lean": ", lean table optimizer (row-wise v, bf16 momentum)",
                "eng20_lrx2p5": ", tables at 2.5 x lr (the paper's 5 x is of an Adam lr)"}
        out.append((f"h5_moe_{tag}", f"{experts} experts + Engram {table / 1e6:.0f}M "
                    f"({share:.0%} of the inactive expert parameters){note.get(tag, '')}",
                    {**h5, **kv}))
    return out


def speed5_ab(size: str = "s1"):
    """Concurrent A/B of the wall clock (user 2026-10-20): plain / mHC-4 / LDM (CuTe) side by side,
    no profiler -- the tok/s of runs started at different times differ by 10-30 % on MN5 (GPFS /
    node load), so only arms that run together compare. 130 steps: the median over steps 60-130
    (user: ~50 steps past the warm-up suffice)."""
    keep = ("h5_moe", "h5_moe_mhc4", "h5_moe_ldm_cute", "h5_moe_mhc4_pl", "h5_moe_ldm_cute_pl")
    # 16 workers x prefetch 8: with FLAG8K's 4 x 2 every arm ran at the same ~23.8k (data-bound;
    # 2026-10-20 A/B); 130 steps without checkpoint saves stay clear of the save-time host OOM
    ab = {B + "exit_interval": 130, "job.est_steps": 130, B + "num_workers": 16,
          B + "dataloader_prefetch_factor": 8}
    return [(n, c, {**kv, **ab}) for n, c, kv in speed5_residuals(size) if n in keep]


def arms5ldm(size: str = "s1"):
    """LDM learning-rate sweep (user 2026-10-19): the round-5 s1 LDM arm (reference, no graphs)
    trails mHC-4 by ~0.008 at matched steps, against the paper's -0.010..-0.013 LEAD; the one knob
    not tuned is LDM's own lr (its parameters on Adam at residual_mixer_lr_mult x 4e-3; the paper
    tuned per method, LDM's optimum 1e-3 of AdamW). The CuTe stream form with the MoE router graphs,
    multipliers 0.125 / 0.25 / 0.5 / 1.0 (Adam 5e-4 / 1e-3 / 2e-3 / 4e-3); x0.25 also checks the
    CuTe numerics (bf16 memory) against the reference run. Same data order as round 5."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    ldm = {B + "depth_memory": True, B + "depth_memory_impl": "cute", B + "depth_memory_groups": 16,
           B + "depth_memory_key_dim": 8, B + "depth_memory_hidden_dim": 32}
    out = []
    for mult, tag in ((0.125, "0125"), (0.25, "025"), (0.5, "05"), (1.0, "1")):
        # 8 data workers x prefetch 4 (FLAG8K's 4 x 2 capped s1 at ~23.8k tok/s/GPU for every arm;
        # 16 x 8 OOM-killed s2 nodes at saves): restarted with it from the step-~500 checkpoints
        out.append((f"h5_moe_ldmc_lr{tag}", f"LDM (CuTe, router graphs), its lr x{mult} of 4e-3",
                    {**h5, **ldm, B + "residual_mixer_lr_mult": mult, B + "num_workers": 8,
                     B + "dataloader_prefetch_factor": 4}))
    return out


def arms5ldm2(size: str = "s1"):
    """The LDM lr sweep again, fast (user 2026-10-20): 2 nodes per arm (data parallel 8, EP 4 in a
    node; global batch 64 x 8k and the data order unchanged, so the curves still pair with round
    5's), 8 data workers x prefetch 4, from step 0 -- plus LDM x2.0 (the best lr was the grid's
    top) and mHC-4 with its mixer lr x1.0 (never swept; fixed at x0.25 since round 4)."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    fast = {"slurm.sbatch.nodes": 2, B + "num_workers": 8, B + "dataloader_prefetch_factor": 4}
    ldm = {B + "depth_memory": True, B + "depth_memory_impl": "cute", B + "depth_memory_groups": 16,
           B + "depth_memory_key_dim": 8, B + "depth_memory_hidden_dim": 32}
    out = []
    for mult, tag in ((0.125, "0125"), (0.25, "025"), (0.5, "05"), (1.0, "1"), (2.0, "2")):
        out.append((f"h5_moe_ldmc_lr{tag}", f"LDM (CuTe, router graphs), its lr x{mult} of 4e-3",
                    {**h5, **ldm, **fast, B + "residual_mixer_lr_mult": mult}))
    mhc4 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe_mhc4"]
    out.append(("h5_moe_mhc4_lr1", "mHC-4, its mixer lr x1.0 of 4e-3 (round 5: x0.25)",
                {**mhc4, **fast, B + "residual_mixer_lr_mult": 1.0}))
    # the 2-node runs do not pair with round 5's 1-node runs (+-0.04 per window at step 500 for the
    # same arm), so the references run in this setup too: mHC-4 at round 5's x0.25, and plain
    out.append(("h5_moe_mhc4", "mHC-4, mixer lr x0.25 (round 5's), 2 nodes", {**mhc4, **fast}))
    out.append(("h5_moe", "plain residual, 2 nodes", {**h5, **fast}))
    # per transformer layer (user 2026-10-20; mhc_per_layer: one read + one write per mixer + MoE
    # block, plain residual inside -- half the residual-scheme work), at each one's best lr
    pl = {B + "mhc_per_layer": True}
    out.append(("h5_moe_ldmc_lr2_pl", "LDM x2.0 per transformer layer",
                {**h5, **ldm, **fast, B + "residual_mixer_lr_mult": 2.0, **pl}))
    out.append(("h5_moe_mhc4_pl", "mHC-4 x0.25 per transformer layer", {**mhc4, **fast, **pl}))
    # round 5d (user 2026-10-20): LDM lr x4.0 (x2.0 was the grid's edge), K = 4 (half the state,
    # ~ mHC-4's 4h: the production pipeline payload), and where the gain sits -- LDM at the
    # mixer sublayers only / at the MoE sublayers only (12 sites each, as per layer)
    ldm2 = {**h5, **ldm, **fast, B + "residual_mixer_lr_mult": 2.0}
    out.append(("h5_moe_ldmc_lr4", "LDM x4.0", {**h5, **ldm, **fast,
                                                B + "residual_mixer_lr_mult": 4.0}))
    out.append(("h5_moe_ldmc_lr2_k4", "LDM x2.0, K = 4", {**ldm2, B + "depth_memory_key_dim": 4}))
    out.append(("h5_moe_ldmc_lr2_mixer", "LDM x2.0 at the mixer sublayers only",
                {**ldm2, B + "depth_memory_sites": "mixer"}))
    out.append(("h5_moe_ldmc_lr2_mlp", "LDM x2.0 at the MoE sublayers only",
                {**ldm2, B + "depth_memory_sites": "mlp"}))
    return out


def arms5ldm2_s2():
    """s2 confirmation (user 2026-10-20): LDM x2.0 vs mHC-4 x0.25 vs plain at s2, 2 nodes each
    (they pair with each other only), 6 data workers x prefetch 3 (s2 nodes OOM-killed at saves
    with 16 x 8; 4 x 2 is data-bound)."""
    pool = dict((n, kv) for n, _, kv in arms5("s2"))
    fast = {"slurm.sbatch.nodes": 2, B + "num_workers": 6, B + "dataloader_prefetch_factor": 3}
    ldm = {B + "depth_memory": True, B + "depth_memory_impl": "cute", B + "depth_memory_groups": 16,
           B + "depth_memory_key_dim": 8, B + "depth_memory_hidden_dim": 32,
           B + "residual_mixer_lr_mult": 2.0}
    return [("h5_moe", "plain residual", {**pool["h5_moe"], **fast}),
            ("h5_moe_mhc4", "mHC-4, mixer lr x0.25", {**pool["h5_moe_mhc4"], **fast}),
            ("h5_moe_ldmc_lr2", "LDM (CuTe) x2.0", {**pool["h5_moe"], **ldm, **fast})]


def arms5kron(size: str = "s1"):
    """KronLSTM (user 2026-10-20; github.com/NX-AI/KronLSTM, the Kronecker delta rule) in place of
    CKDA as the 5:1 mixer, plain residual, round 5d's 2-node setting -- pairs with screen5ldm2's
    2-node h5_moe (CKDA) run, same data order. The paper's layer (softplus forget gate in (0, 1],
    beta in [0, 1]) and the extended ranges as CKDA over KDA: the signed forget gate in [-1, 1]
    (Complex KDA's signed_sigmoid2 with its standard init, via the sign gauge) and beta in [0, 2].
    Head grid as CKDA (128-wide, hidden / 128 heads). The layer's dense output gate is
    hidden x hidden per mixer (CKDA's is low rank): more parameters, reported with the result."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    fast = {"slurm.sbatch.nodes": 2, B + "num_workers": 8, B + "dataloader_prefetch_factor": 4}
    kron = {**{k: v for k, v in h5.items() if k not in CKDA_STD},
            B + "experimental_attention_variant": "kronlstm", **fast}
    return [
        ("h5_moe_kron", "KronLSTM 5:1 (softplus forget gate, beta in [0, 1])",
         {**kron, B + "linear_fla_gate": "softplus", B + "linear_fla_allow_neg_eigval": False}),
        ("h5_moe_kron_ext", "KronLSTM 5:1, forget gate in [-1, 1] and beta in [0, 2]",
         {**kron, B + "linear_fla_gate": "signed_sigmoid2", B + "linear_fla_allow_neg_eigval": True}),
    ]


def arms5kron_scale(size: str):
    """KronLSTM's scaling (user 2026-10-20): the paper's layer (softplus forget gate, beta in
    [0, 1]; it trailed CKDA by +0.02 at step 1000 in the 2-node screen, the extended ranges by
    +0.15) as the 5:1 mixer in round 5's own setting at s1 / s2 / s3 (1 node; s3 4 nodes, EP16),
    so each arm pairs with round 5's CKDA h5_moe of its size (same data order). The fused qkv_proj
    is split for angular MuOwn as complex_kda's three projections are (Megatron b8c7069b7)."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    kron = {**{k: v for k, v in h5.items() if k not in CKDA_STD},
            B + "experimental_attention_variant": "kronlstm", B + "linear_fla_gate": "softplus",
            B + "linear_fla_allow_neg_eigval": False}
    # data workers per layout (FLAG8K's 4 x 2 capped s1 at ~23.8k tok/s/GPU; s2 OOM-killed
    # nodes at saves with 16 x 8): loss unaffected, the tok/s comparable to round 5d
    workers = {"s1": (8, 4), "s2": (6, 3)}.get(size)
    if workers:
        kron.update({B + "num_workers": workers[0], B + "dataloader_prefetch_factor": workers[1]})
    out = [("h5_moe_kron", "KronLSTM 5:1 (paper's gates), round 5's setting", kron)]
    if size == "s1":
        # the extended ranges again, with the signed gate's init fixed (FLA f82079c: A_log = 0 as in
        # Complex KDA; the screen's arm ran with 96 % of the forget gates saturated at |f| = 1)
        out.append(("h5_moe_kron_ext", "KronLSTM 5:1, f in [-1, 1] + beta in [0, 2] (init fixed)",
                    {**kron, B + "linear_fla_gate": "signed_sigmoid2",
                     B + "linear_fla_allow_neg_eigval": True}))
    return out


def arms5udecay(size: str):
    """Angular MuOwn's hidden U-decay (user via the EP agent, 2026-10-20): Megatron's defaults
    (angular_muown_u_decay_schedule poly, scale 1e-3, p 1) multiply the direction step by
    m(t) = 1 / (1 + 0.001 t) ON TOP of the lr schedule (0.2 at step 4000, ~0.001 at 1M steps);
    every MuOwn arm so far carried it. Here it is off (p = 0: m = 1) on round 5's plain h5_moe,
    paired with round 5's h5_moe of the size (decay on, same data order), with a sweep of the
    direction step (the lr was tuned with the decay): angular_muown_extra_scale_factor -- it scales
    the U step alone, so the Adam-routed parameters (row magnitudes, embeddings, norms) keep their lr
    -- x1 / x0.5 / x0.25 of 0.2 at s1, x1 / x0.5 at s2. The fair comparison -- the decay in the
    visible schedule vs the hidden one -- follows as cooldown branches from step 4000."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    workers = {"s1": (8, 4), "s2": (6, 3)}[size]
    off = {**h5, B + "angular_muown_u_decay_p": 0.0, B + "num_workers": workers[0],
           B + "dataloader_prefetch_factor": workers[1]}
    mults = (1.0, 0.5, 0.25) if size == "s1" else (1.0, 0.5)
    tag = lambda m: "" if m == 1.0 else f"_u{m:g}".replace(".", "")  # noqa: E731
    return [(f"h5_moe_udoff{tag(m)}", f"angular MuOwn U-decay off, direction step x{m:g} ({0.2 * m:g})",
             {**off, B + "angular_muown_extra_scale_factor": 0.2 * m}) for m in mults]


def arms5kron_ext(size: str):
    """KronLSTM's extended ranges at every size (user 2026-10-21), started as the paper's layer: the
    signed forget gate with Complex KDA's shipped init (A_log = 0, f = +exp(-dt) in (0, 1) at step
    0: positive bias), and beta in [0, 2] with the erase gate biased to start at mean 0.5, mostly in
    [0, 1] (linear_kronlstm_neg_eigval_init_bias). The write gate stays c in [0, 1] by construction
    (linear_kronlstm_unit_write: the op's s = 2 would double it). Own suite (the kronscale suites
    have running arms); pairs with round 5's h5_moe."""
    kron = arms5kron_scale(size)[0][2]
    return [("h5_moe_kron_extb", "KronLSTM 5:1, f in [-1, 1] + erase in [0, 2] (started in [0, 1]), write in [0, 1]",
             {**kron, B + "linear_fla_gate": "signed_sigmoid2", B + "linear_fla_allow_neg_eigval": True,
              B + "linear_kronlstm_neg_eigval_init_bias": True, B + "linear_kronlstm_unit_write": True})]


def arms5udecay_sqrt(size: str):
    """Angular MuOwn's U-decay at p = 1/2 (user 2026-10-21): kappa = (1 + 0.001 t)^-1/2 -- the
    t^-1/2 that plain Muown's growing row norms give implicitly (arXiv 2606.23637, Table 1),
    sustainable to ~1M steps (kappa ~0.03 there; p = 1: ~0.001). Default direction step (0.2).
    Pairs with round 5's h5_moe (p = 1) and screen5udecay's p = 0 arms; own suite (those run)."""
    off = arms5udecay(size)[0][2]
    return [("h5_moe_udp05", "angular MuOwn U-decay p = 1/2 (c = 0.001), direction step 0.2",
             {**off, B + "angular_muown_u_decay_p": 0.5})]


def arms5kron_lowrank(size: str):
    """KronLSTM with complex_kda's low-rank output gate (user 2026-10-21: low-rank from now on, so
    the comparison with CKDA is the recurrence alone, parameter-matched), in round 5's setting,
    paired with its h5_moe: the paper's gates at s1 / s2 / s3, and at s1 the polar gates (FLA
    fab34df) -- f = m cos(theta) with m the paper's softplus gate (its short- and long-memory spread)
    and the angle starting near 0, so each arm starts as the paper's layer: the forget angle alone,
    + the erase angle (1 - beta = r cos(phi): beta in [0, 2], the write kept in [0, 1]), + the
    signed write c in [-1, 1]."""
    paper = {**arms5kron_scale(size)[0][2], B + "linear_kronlstm_output_gate": "lowrank"}
    out = [("h5_moe_kron_lr", "KronLSTM 5:1, paper's gates, low-rank output gate", paper)]
    if size == "s1":
        f = {**paper, B + "linear_kronlstm_forget_angle": True}
        fb = {**f, B + "linear_kronlstm_erase_angle": True}
        out += [("h5_moe_kron_lr_pf", "+ polar forget gate (f = m cos theta)", f),
                ("h5_moe_kron_lr_pfb", "+ polar forget and erase gates", fb),
                ("h5_moe_kron_lr_pfbc", "+ polar forget and erase gates, signed write c in [-1, 1]",
                 {**fb, B + "linear_kronlstm_signed_write": True})]
    return out


# the step-4000 checkpoints each cooldown branch starts from (hard-linked into the branch's
# ckpt_rolling before the launch: scripts/korbi/campaign/link_branch_ckpt.sh)
UDCOOL_FROM = {"h5_moe_p1": "screen5_{z}/h5_moe", "h5_moe_p05": "screen5udecaysqrt_{z}/h5_moe_udp05",
               "h5_moe_p0_u025": "screen5udecay_{z}/h5_moe_udoff_u025"}


def arms5udcool(size: str):
    """Angular MuOwn's U-decay, the fair comparison (user 2026-10-20/21): each schedule cooled down
    the visible way -- WSD, linear to lr / 10 over steps 4000-4800 -- from its own step-4000
    checkpoint (same data order: train_iters unchanged, the branch resumes at 4000), so the hidden
    decay's annealing advantage in the stable phase is matched by the visible one. p = 1 (round 5's
    h5_moe), p = 1/2 (screen5udecaysqrt) and, at s1, p = 0 at direction step 0.05 (the best p = 0
    arm; s2's best p = 0 arm died at step 3360 from a node failure, past the point of interest)."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    lr = h5[B + "lr"]
    cool = {B + "lr_decay_style": "WSD", B + "lr_wsd_decay_style": "linear", B + "lr_decay_iters": 4800,
            B + "lr_wsd_decay_iters": 800, B + "min_lr": lr / 10, B + "override_opt_param_scheduler": True,
            # min_lr is part of the optimizer's param-group key: load each group's state from its
            # structurally equivalent saved group, keeping the new lr bounds
            B + "allow_new_param_groups_on_load": True,
            B + "exit_interval": 4800, "job.est_steps": 800}
    workers = {"s1": (8, 4), "s2": (6, 3)}[size]
    w = {B + "num_workers": workers[0], B + "dataloader_prefetch_factor": workers[1]}
    # round 5's s2 checkpoint has the pre-1bbeb2b55 compact layout for factory weights
    legacy = {"backend.env.ANGULAR_MUOWN_LEGACY_FACTORY_ROWS": "1"} if size == "s2" else {}
    out = [("h5_moe_p1", "p = 1 (default), cooled down 4000 -> 4800", {**h5, **w, **cool, **legacy}),
           ("h5_moe_p05", "p = 1/2, cooled down 4000 -> 4800",
            {**arms5udecay_sqrt(size)[0][2], **cool})]
    if size == "s1":
        out.append(("h5_moe_p0_u025", "p = 0, direction step 0.05, cooled down 4000 -> 4800",
                    {**dict((n, kv) for n, _, kv in arms5udecay(size))["h5_moe_udoff_u025"], **cool}))
    return out


def arms5kron_pfb(size: str):
    """CKLSTM's best s1 variant at s2 / s3 (user 2026-10-21, budget raised to 7.5k GPU-h): the polar
    forget and erase gates with the low-rank output gate (s1: +0.003 at step 4000, trend -0.005 to
    -0.008 at 10B tokens vs CKDA). Own suite (screen5kronlr's suites hold finished runs); pairs with
    round 5's h5_moe and with screen5kronlr's paper-gate arm of the size."""
    paper = arms5kron_lowrank(size)[0][2]
    return [("h5_moe_kron_lr_pfb", "KronLSTM 5:1, polar forget and erase gates, low-rank output gate",
             {**paper, B + "linear_kronlstm_forget_angle": True, B + "linear_kronlstm_erase_angle": True})]


# MN5 nodes that kill jobs at launch (torchrun NUMA binding: "Invalid device id"; EP agent 2026-10-21)
BAD_NODES = {"slurm.sbatch.exclude": "as07r3b27,as06r2b26,as05r3b28"}
# the polar forget gate's init cap, |f| <= cos(pi * 0.05), as a fixed decay: -log cos(0.157)
POLAR_OFFSET = round(-math.log(math.cos(math.pi * 0.05)), 5)


def arms5offset(size: str = "s1"):
    """Shorter memory horizons, without the polar angle (user 2026-10-21): the s1 checkpoints show
    the polar forget gate never turns negative and acts as an init cap |f| <= 0.988 (p90 horizon
    ~280 -> ~80 tokens). A fixed extra decay per step (forget_offset = -log 0.988) on
    KronLSTM's paper gates (low-rank output gate; compare screen5kronlr's h5_moe_kron_lr /
    _pfb) and on CKDA (round 5's h5_moe with the offset: does the reference mixer gain the same?)."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    kron = arms5kron_lowrank(size)[0][2]
    w = {B + "num_workers": 8, B + "dataloader_prefetch_factor": 4}
    return [("h5_moe_kron_lr_off", f"KronLSTM paper gates + forget offset {POLAR_OFFSET}",
             {**kron, B + "linear_kronlstm_forget_offset": POLAR_OFFSET, **BAD_NODES}),
            ("h5_moe_off", f"CKDA + forget offset {POLAR_OFFSET}",
             {**h5, **w, B + "linear_gate_forget_offset": POLAR_OFFSET, **BAD_NODES})]


def arms5udadapt(size: str = "s1"):
    """AngularMuown's adaptive direction schedules (user 2026-10-21; Megatron 43dc8243c), each over
    the whole schedule in one run -- stable to step 4000, then WSD linear to lr / 10 at 4800, the
    data order of round 5 (train_iters unchanged) -- so they pair with screen5udcool's p = 1 / 1/2 / 0
    branches and with round 5's h5_moe (p = 1) in the stable phase:
      radius: one shared radius, kappa = 1 / rho, rho^2 <- (1 - 2 lr lambda) rho^2 + c (lr/max_lr)^2;
        c = 0.0074, lambda = 0.083: kappa falls as (1 + c t)^-1/2 at first and settles at
        kappa* = 1 / sqrt(c tau) = 0.3 within ~1500-3000 steps (tau = 1 / (2 lr lambda) = 1500):
        an equilibrium step instead of a decay that keeps going (lambda alone would only show after
        ~10k steps);
      hyper: log kappa learnt from the gradient / previous-momentum correlation (defaults);
      p = 1/2 at half the direction step (0.1): the s2 cooldown left p = 1/2 only 0.008 behind p = 1,
        with its step untuned."""
    h5 = dict((n, kv) for n, _, kv in arms5(size))["h5_moe"]
    lr = h5[B + "lr"]
    sched = {B + "lr_decay_style": "WSD", B + "lr_wsd_decay_style": "linear", B + "lr_decay_iters": 4800,
             B + "lr_wsd_decay_iters": 800, B + "min_lr": lr / 10, B + "exit_interval": 4800,
             "job.est_steps": 4800, B + "num_workers": 8, B + "dataloader_prefetch_factor": 4, **BAD_NODES}
    base = {**h5, **sched}
    return [("h5_moe_radius", "AngularMuown radius schedule (c 0.0074, lambda 0.083: kappa* 0.3)",
             {**base, B + "angular_muown_u_decay_schedule": "radius", B + "angular_muown_u_decay_scale": 0.0074,
              B + "angular_muown_u_radius_lambda": 0.083}),
            ("h5_moe_hyper", "AngularMuown hypergradient schedule (hyper_lr 0.02, floor 0.02)",
             {**base, B + "angular_muown_u_decay_schedule": "hyper"}),
            ("h5_moe_p05_u05", "AngularMuown p = 1/2, direction step 0.1",
             {**base, B + "angular_muown_u_decay_p": 0.5, B + "angular_muown_extra_scale_factor": 0.1})]


RUNS5 = "/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/campaign_3b"


def arms5quant(size: str):
    """4-bit robustness (user 2026-10-21): round 5's step-4000 checkpoints evaluated twice -- full
    precision and weight-only 4-bit (QUANT_EVAL_BITS=4, groups of 32 along the input dim, close to
    llama.cpp's Q4_K; embeddings / output / routers / norms / residual mixers kept; Megatron
    a33f2abd1) -- on the same validation batches, in each run's own layout. Evaluation only
    (skip_train, model weights only): our hybrid vs the Qwen3.5-like and all-attention arms."""
    pool = dict((n, kv) for n, _, kv in arms5(size))
    arms = {"s2": ("h5_moe", "h5_moe_mhc4", "qwen35_moe", "attn_moe"),
            "s3": ("h5_moe", "h5_moe_mhc4", "qwen35_moe")}[size]
    out = []
    for a in arms:
        src = f"{RUNS5}/screen5_{size}/{a}"
        ev = {B + "skip_train": True, B + "eval_iters": 50, B + "no_load_optim": True,
              B + "no_load_rng": True, B + "load": f"{src}/ckpt",
              B + "non_persistent_global_ckpt_dir": f"{src}/ckpt_rolling", "job.est_steps": 1,
              "slurm.sbatch.time": "01:00:00", **BAD_NODES}
        for tag, bits in (("fp", None), ("q4", "4")):
            env = {"backend.env.QUANT_EVAL_BITS": bits} if bits else {}
            out.append((f"{a}_{tag}", f"{a}, step 4000, {'4-bit weights' if bits else 'full precision'}",
                        {**pool[a], **ev, **env}))
    return out


def speed5_profiled(size: str = "s1"):
    """speed5_residuals for 300 steps with a torch profile of steps 200-203 (rank 0; summarise
    with scripts/korbi/hybrid_speed/kernel_summary.py <arm dir>): where LDM's time goes. 100
    steps were too short -- plain, mHC-4 and AttnRes all ran at ~32k tok/s/GPU there."""
    keep = ("h5_moe", "h5_moe_mhc4", "h5_moe_attnres_streams", "h5_moe_ldm_cute", "h5_moe_ldm_torch")
    # no evaluation in the window (its forward-only micro-batches entered one profile window;
    # eval_iters 0 leaves an empty validation set: "no sample to consume"), window 250-253
    prof = {B + "exit_interval": 300, "job.est_steps": 300, B + "profile": True,
            B + "use_pytorch_profiler": True, B + "profile_step_start": 250,
            B + "profile_step_end": 253, B + "profile_ranks": [0],
            B + "eval_interval": 100000, B + "eval_iters": 1}
    return [(n, c, {**kv, **prof}) for n, c, kv in speed5_residuals(size) if n in keep]


def speed5_residuals(size: str = "s1"):
    """Throughput of the residual variants in round 5's setting (flag mix, 8k, document masking,
    MoE router / preprocess CUDA graphs), 100 steps each (user 2026-10-19: proper kernels and CUDA
    graphs for AttnRes and LDM): the stream forms (graph-capturable) against the state forms."""
    pool = dict((n, kv) for n, _, kv in arms5(size))
    h5 = pool["h5_moe"]
    short = {B + "exit_interval": 100, "job.est_steps": 100}
    nocg = {B + "cuda_graph_impl": "none", B + "cuda_graph_modules": []}
    ldm = {B + "depth_memory": True, B + "depth_memory_groups": 16, B + "depth_memory_key_dim": 8,
           B + "depth_memory_hidden_dim": 32, B + "residual_mixer_lr_mult": 0.25}
    attnres = {B + "attn_res": True, B + "attn_res_num_blocks": 4}
    arms = [
        ("h5_moe", "plain residual (router graphs)", h5),
        ("h5_moe_mhc4", "mHC-4 (router graphs)", pool["h5_moe_mhc4"]),
        ("h5_moe_attnres", "AttnRes, state form (no graphs)", pool["h5_moe_attnres"]),
        ("h5_moe_attnres_streams", "AttnRes, stream form (router graphs)",
         {**h5, **attnres, B + "attn_res_impl": "streams"}),
        ("h5_moe_ldm", "LDM reference (no graphs)", {**h5, **ldm, **nocg}),
        ("h5_moe_ldm_torch", "LDM stream form, torch unwinding (router graphs)",
         {**h5, **ldm, B + "depth_memory_impl": "torch"}),
        ("h5_moe_ldm_cute", "LDM stream form, the paper's CuTe kernels (router graphs)",
         {**h5, **ldm, B + "depth_memory_impl": "cute"}),
        ("h5_moe_mhc4_pl", "mHC-4 per transformer layer (router graphs)",
         {**pool["h5_moe_mhc4"], B + "mhc_per_layer": True}),
        ("h5_moe_ldm_cute_pl", "LDM (CuTe) per transformer layer (router graphs)",
         {**h5, **ldm, B + "depth_memory_impl": "cute", B + "mhc_per_layer": True}),
    ]
    return [(n, c, {**kv, **short}) for n, c, kv in arms]


def write5():
    steps_s = {"s1": 3.0, "s2": 6.0, "s3": 7.0}
    times = {"s1": "08:00:00", "s2": "12:00:00", "s3": "12:00:00"}
    # smoke: every s1 arm (and the s3 layout) for 40 steps, before the unattended start
    smoke = [(n, c, {**kv, B + "exit_interval": 40, "job.est_steps": 40})
             for n, c, kv in arms5("s1")]
    # (the 2-node s3 layout cannot share the 1-node array: checked separately)
    smoke3 = [(n, c, {**kv, B + "exit_interval": 40, "job.est_steps": 40})
              for n, c, kv in arms5("s3")]
    # Engram sweep and its smoke (40 steps of every arm, s1 and s2)
    smoke_e = lambda z: [(n, c, {**kv, B + "exit_interval": 40, "job.est_steps": 40})  # noqa: E731
                         for n, c, kv in arms5e(z)]
    for size, prefix, pool in [(z, "screen5", arms5(z)) for z in ("s1", "s2", "s3")] + [
            ("s1", "speed5res", speed5_residuals("s1")),
            ("s1", "speed5prof", speed5_profiled("s1")),
            ("s1", "screen5ldm", arms5ldm("s1")),
            ("s1", "speed5ab", speed5_ab("s1")),
            ("s1", "screen5ldm2", arms5ldm2("s1")),
            ("s2", "screen5ldm2", arms5ldm2_s2()),
            ("s1", "screen5kron", arms5kron("s1")),
            *[(z, "screen5kronscale", arms5kron_scale(z)) for z in ("s1", "s2", "s3")],
            *[(z, "screen5kronext", arms5kron_ext(z)) for z in ("s1", "s2", "s3")],
            *[(z, "screen5kronlr", arms5kron_lowrank(z)) for z in ("s1", "s2", "s3")],
            *[(z, "screen5kronpfb", arms5kron_pfb(z)) for z in ("s2", "s3")],
            ("s1", "screen5off", arms5offset("s1")),
            ("s1", "screen5udadapt", arms5udadapt("s1")),
            *[(z, "quanteval5", arms5quant(z)) for z in ("s2", "s3")],
            *[(z, "screen5udecay", arms5udecay(z)) for z in ("s1", "s2")],
            *[(z, "screen5udecaysqrt", arms5udecay_sqrt(z)) for z in ("s1", "s2")],
            *[(z, "screen5udcool", arms5udcool(z)) for z in ("s1", "s2")],
            ("s1", "screen5kronsmoke", [(n, c, {**kv, B + "exit_interval": 40, "job.est_steps": 40})
                                        for n, c, kv in arms5kron("s1")])] + [
            ("s1", "screen5smoke", smoke), ("s3", "screen5smoke", smoke3)] + [
            (z, p, arms) for z in ("s1", "s2")
            for p, arms in (("screen5e", arms5e(z)), ("screen5esmoke", smoke_e(z)))]:
        hidden, blocks, heads, ffn, lr = R4_SIZES[size]
        lines = [HEADER.format(
            title="Round 5: the architecture decision on the production data setting (flag, 8k)",
            suite=f"{prefix}_{size}", size=size, hidden=hidden, tokens="4.0e9", blocks=blocks,
            heads=heads, ffn=ffn, lr=lr, gbs=64, pattern="*-" * blocks,
            matched="MoE: equal active and total by construction", time=times[size],
            step_s=steps_s[size])]
        for name, comment, kv in pool:
            lines.append(f"        # {comment}\n        - aux.arm: {name}\n")
            for k, v in kv.items():
                val = yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
                lines.append(f"          {k}: {val}\n")
        (OUT / f"{prefix}_{size}_mn5.yaml").write_text("".join(lines))
        print(f"{prefix}_{size}_mn5.yaml: {len(pool)} arms")


def write4():
    _write4("s1", "screen4ry", "Round 4R-y: the routed residual as a cheap axis (2 / 1 of 8)",
            arms4ry("s1"))
    for size in ("s1", "s2"):
        _write4(size, "screen4rx", "Round 4R extension: residual scaling and routed-residual levers",
                arms4rx(size))
    _write4("s1", "screen4r", "Round 4R: residual arms with the stream diagnostics", arms4r("s1"))
    _write4("s1", "flagsmoke", "Flag mix smoke at 8k with document masking", flag_smoke("s1"))
    for size in ("s1", "s2"):
        _write4(size, "screen4", "Round 4: the MoE hybrid with a residual stream + LatentMoE",
                arms4(size))
    for size in ("s1",):
        _write4(size, "screen4g", "Round 4G: MoE granularity at fixed sparsity and parameters",
                arms4g(size))


def _write4(size, prefix, title, pool):
    hidden, blocks, heads, ffn, lr = R4_SIZES[size]
    lines = [HEADER.format(title=title,
                           suite=f"{prefix}_{size}", size=size, hidden=hidden, tokens="4.0e9",
                           blocks=blocks, heads=heads, ffn=ffn, lr=lr, gbs=128,
                           pattern="*-" * blocks, matched="MoE: active FFN equal by construction",
                           time="12:00:00" if size == "s2" else "08:00:00",
                           step_s=1.0 if size == "s1" else 2.5)]
    for name, comment, kv in pool:
        lines.append(f"        # {comment}\n        - aux.arm: {name}\n")
        for k, v in kv.items():
            val = yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
            lines.append(f"          {k}: {val}\n")
    (OUT / f"{prefix}_{size}_mn5.yaml").write_text("".join(lines))
    print(f"{prefix}_{size}_mn5.yaml: {len(pool)} arms")


def main():
    pm_path = OUT / "param_match.json"
    pm = json.loads(pm_path.read_text()) if pm_path.exists() else {}
    for prefix, sizes, keep, title in (
            ("screen", SIZES, None, "Parameter-matched feature screen (round 1)"),
            ("screen2", SIZES2, ARMS2, "Parameter-matched feature screen 2 (AdamW, 1000-step warmup, stable)"),
            ("screen3", {"s1": SIZES2["s1"]}, None, "Level-1 round 3: hybrid recipe + residual / memory variants")):
        write(prefix, sizes, keep, title, pm)
    write3b()
    write3c(pm)
    write4()
    write5()


def write(prefix, sizes, keep, title, pm):
    for size, (hidden, blocks, heads, ffn, lr, gbs) in sizes.items():
        table = pm.get(size, {})
        # screen 2 runs 4BT: at gbs 128 that is ~6.6k stable steps, where the single-run curve
        # fits extrapolate (round 1: backtest within 1 % for 47/68 runs > 6.5k stable steps, vs
        # 8/44 at 2.5-6.5k)
        tokens = "4.0e9" if prefix in ("screen2", "screen3") else "2.0e9"
        lines = [HEADER.format(title=title, suite=f"{prefix}_{size}", size=size, hidden=hidden, tokens=tokens,
                               blocks=blocks, heads=heads, ffn=ffn, lr=lr,
                               gbs=gbs, pattern="*-" * blocks,
                               matched=f"{len(table)} arms counted" if table else "NOT YET COUNTED",
                               time="02:00:00" if size == "s1" else "05:00:00",
                               step_s=0.6 if size == "s1" else 1.6)]
        pool = arms3(blocks) if prefix == "screen3" else arms(blocks)
        chosen = [a for a in pool if keep is None or a[0] in keep]
        for name, comment, kv in chosen:
            kv = dict(kv)
            # memory at s2 (gbs unchanged): triadic CKDA OOMs at mbs 8 and 4, the GDN arms with
            # heads of 128 at mbs 8
            if prefix in ("screen2", "screen3"):
                # short runs (user decision): end at SHORT_STEPS through the config; the validation
                # arms run the suite's full budget to check the curve-fit extrapolation
                if name not in (VALIDATION_ARMS if prefix == "screen2" else VALIDATION3):
                    kv["aux.tokens"] = SHORT_STEPS * gbs * 4096
            if size == "s2" and name.startswith("h_tri"):
                kv[B + "micro_batch_size"] = 2
            if size == "s2" and name in ("h_gdn_nope", "h_gdn_qwen35"):
                kv[B + "micro_batch_size"] = 4
            kp = R3_KEY if prefix == "screen3" else ""
            entry = table.get(kp + name) or (table.get(kp + NOT_MATCHED[name]) if prefix == "screen3" and name in NOT_MATCHED else None)
            if prefix == "screen3" and name.endswith(("_seed2", "_4k")):
                entry = table.get(kp + name.removesuffix("_seed2").removesuffix("_4k"))
            if entry:
                kv[B + "ffn_hidden_size"] = entry["ffn"]
                comment += f" (d_ffn {entry['ffn']}, non-emb {entry['non_emb'] / 1e6:.1f}M)"
            else:
                comment += " (d_ffn not matched yet)"
            lines.append(f"        # {comment}\n        - aux.arm: {name}\n")
            for k, v in kv.items():
                val = yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()
                lines.append(f"          {k}: {val}\n")
        (OUT / f"{prefix}_{size}_mn5.yaml").write_text("".join(lines))
        print(f"{prefix}_{size}_mn5.yaml: {len(chosen)} arms, "
              f"{sum(((R3_KEY if prefix == 'screen3' else '') + a[0]) in table for a in chosen)} matched")


if __name__ == "__main__":
    main()
