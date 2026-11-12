#!/usr/bin/env python3
"""Add a ckda_branch (Megatron megatron/core/ssm/ckda_branch.py) to complex_kda layers of an fsdp_dtensor
checkpoint: a softmax attention or a KronLSTM mixer in parallel, started from the layer's own projections.

The model is unchanged at insertion: the branch's output is scaled by ``branch_mix.gamma = 0``. Every
new tensor gets optimizer state of the optimizer that will own it (MuOwn classical for 2-D matrices:
per-row g, m_g, v_g, v_norm as [rows, 1] and m_u; Adam exp_avg / exp_avg_sq otherwise).

**attention** (``self_attention.branch.*``: ``linear_qkv`` per query group [q | gate | k | v] x d, d = the
stack's attention head dim, 16 query heads, ``G`` KV groups; ``q_layernorm`` / ``k_layernorm``;
``linear_proj``), from the complex_kda projections (16 heads x dc = 128):
  q        query head h = complex_kda head h in one half of d (left for even h, right for odd h), the other
           half small-noise rows: q'_h . k'_g starts as q_h . k_h, the pad grows into the partner's half
  k, v     complex_kda heads merged in adjacent pairs (8 heads x d: [k_2j | k_2j+1]); G = 8 keeps them,
           G = 2 averages 4 merged heads per group (the GQA uptraining init; linear, so exact on the
           projection weights)
  gate     small-noise rows (sigmoid ~ 0.5)
  linear_proj  head h's own half of input columns = complex_kda o_proj's columns of head h, the other
           half small noise (as widen_attention_head_dim.py)
  q/k norm gains: the median gain of the stack's attention layers (their temperature)
  MuOwn states: copied rows keep their source rows' states; pooled rows the mean; new rows
           g = row norm, v_norm = median of the source's, m_u = m_g = v_g = 0. linear_proj: v_norm x
           sqrt(d_in_new / d_in_old), m_u / sqrt(...) (the widening rule).

**kvshare** (``self_attention.branch.*``: ``linear_qkv`` per query group [q | gate] x d, ``q_layernorm``,
``linear_proj``; the keys / values are an earlier attention layer's at run time, ``--kv-sources``): random
q / gate / o rows at the source attention layer's row norms (gate x eps), its q norm gains (kvshare_branch).

**attention, --init attn** (copy of a stack attention layer, ``--kv-sources``: one source per layer): the branch
is the source attention layer itself -- ``linear_qkv`` (same KV groups), ``linear_proj``, q / k norm and all
their optimizer states copied -- so a second attention step of the copy layer's kind runs on the branch
layer's input. The source's input-norm gain (``linear_qkv.layer_norm_weight``; the branch reads the
complex_kda layer's normalized input instead) is folded into the ``linear_qkv`` columns,
``W' = W diag(g_attn / g_layer)``: branch(x) = the source attention of RMSNorm(x) g_attn (``--no-fold-norm``
keeps ``W``). With the fold the MuOwn states follow the chain rule for the same function: ``g`` = the new row
norms, ``m_g`` / ``v_g`` x (g / g') / (g / g')^2, ``m_u`` columns / (g_attn / g_layer); ``v_norm`` copied.
gamma 0 (exact at insertion; an o_proj at 0 would leave MuOwn without row directions and calibrate without
a branch RMS). Recall surgery post4: layers 19 / 35 / 52 / 68 from 17 / 33 / 50 / 66.

**kronlstm** (``self_attention.branch.layer.*``): qkv_proj / qkv_conv1d = complex_kda's q / k / v
projections and convolution kernels stacked (exact: depthwise), f_proj / A_log / dt_bias / g_proj /
o_norm / o_proj copied, b_proj copied (erase s sigmoid(b), s = 2 as complex_kda's linear_beta_max 2; no bias:
neg_eigval_init_bias off), c_proj (write gate) = b_proj: with decouple_erase and without unit_write the write is
s c v, the update s sigmoid(b) (v - S k) -- complex_kda's delta rule, on the normalized v (the "hard break").

    python add_ckda_branch.py selftest
    PYTHONPATH=<fla fork> python add_ckda_branch.py gpucheck
    python add_ckda_branch.py apply --src <ckpt>/iter_0023300 --dst <new>/iter_0023300 \\
        --branch attention --groups 2 --layers 9,25,41 [--eps 1e-3] [--seed 1234]
    python add_ckda_branch.py apply --src <ckpt>/iter_0023400 --dst <new>/iter_0023400 \\
        --branch attention --groups 2 --layers 19,35,52,68 --init attn --kv-sources 17,33,50,66 --calibrate
    python add_ckda_branch.py verify --dst <new>/iter_0023300   (re-derive, compare the manifest's hashes)

apply writes <dst>/ckda_branch_surgery.json: arguments + seed, the source and its .metadata SHA-256, the code
(SYNC_STAMP / git), torch, and the SHA-256 of every new tensor. Production runs go through the 1-node pool job
config/experiments/korbi/production/ckda_branch_surgery_mn5.yaml (scripts/korbi/production/ckda_branch_surgery.sh).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

MODEL = "model.module.decoder.layers.{l}.self_attention.{n}"
OPT = "optimizer.0.state.module.module.module.decoder.layers.{l}.self_attention.{n}.{s}"
MUOWN = ("g", "m_g", "v_g", "v_norm", "m_u")
ADAM = ("exp_avg", "exp_avg_sq")
ATTN_LAYERS = (0, 17, 33, 50, 66, 82)  # the stack's attention layers (q/k norm gain reference)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _noise_rows(n, cols, scale, gen):
    """n random unit directions x scale (rows)."""
    x = torch.randn(n, cols, generator=gen)
    return x / x.norm(dim=1, keepdim=True) * scale


def _new_rows_states(w, v_norm_ref):
    """MuOwn states of fresh rows w: g = row norm, v_norm = the reference, momenta 0."""
    n, c = w.shape
    return {"g": w.norm(dim=1, keepdim=True), "m_g": torch.zeros(n, 1), "v_g": torch.zeros(n, 1),
            "v_norm": torch.full((n, 1), float(v_norm_ref)), "m_u": torch.zeros(n, c)}


def _rows(states, r0, r1):
    return {k: v[r0:r1].clone() for k, v in states.items()}


def _cat_states(parts):
    return {k: torch.cat([p[k] for p in parts], 0) for k in MUOWN}


def _mean_states(parts, w):
    out = {k: torch.stack([p[k] for p in parts]).mean(0) for k in ("m_g", "v_g", "v_norm", "m_u")}
    out["g"] = w.norm(dim=1, keepdim=True)
    return out


# ---------------------------------------------------------------------------------------------
# the transforms (pure torch, CPU)
# ---------------------------------------------------------------------------------------------


def attention_branch(src: dict, ref_gains: dict, *, heads: int, d: int, groups: int, eps: float,
                     gen: torch.Generator) -> dict:
    """New tensors of an attention branch: {name: tensor}, names relative to ``self_attention.``.

    ``src``: the complex_kda layer's ``layer.q_proj.weight`` etc. and their MuOwn states
    (``<name>.<state>``). ``ref_gains``: q/k norm gains and Adam exp_avg_sq of the stack's attention."""
    q, k, v, o = (src[f"layer.{n}_proj.weight"] for n in "qkvo")
    hidden = q.shape[1]
    dc = q.shape[0] // heads
    if q.shape[0] != heads * dc or d != 2 * dc or heads % 2 or heads % groups:
        raise ValueError(f"attention_branch: {heads} heads x {dc} -> head dim {d}, {groups} groups")
    st = {n: {s: src[f"layer.{n}_proj.weight.{s}"] for s in MUOWN} for n in "qkvo"}
    hpg = heads // groups
    q_scale = eps * q.norm(dim=1).median()
    q_vref = st["q"]["v_norm"].median()

    def head(w, s, h):  # rows of complex_kda head h
        return w[h * dc:(h + 1) * dc], _rows(s, h * dc, (h + 1) * dc)

    def merged(w, s, j):  # merged KV head j = [head 2j | head 2j+1]
        a, sa = head(w, s, 2 * j)
        b, sb = head(w, s, 2 * j + 1)
        return torch.cat([a, b], 0), _cat_states([sa, sb])

    rows, states = [], []
    per_merged = (heads // 2) // groups  # merged KV heads per group (1 for G = 8, 4 for G = 2)
    for g in range(groups):
        for i in range(hpg):  # q heads of the group
            h = g * hpg + i
            own, s_own = head(q, st["q"], h)
            pad = _noise_rows(dc, hidden, q_scale, gen)
            s_pad = _new_rows_states(pad, q_vref)
            if h % 2 == 0:
                rows.append(torch.cat([own, pad], 0)); states.append(_cat_states([s_own, s_pad]))
            else:
                rows.append(torch.cat([pad, own], 0)); states.append(_cat_states([s_pad, s_own]))
        for _ in range(hpg):  # gate heads
            gate = _noise_rows(d, hidden, q_scale, gen)
            rows.append(gate); states.append(_new_rows_states(gate, q_vref))
        for w, s in ((k, st["k"]), (v, st["v"])):
            parts = [merged(w, s, g * per_merged + j) for j in range(per_merged)]
            if per_merged == 1:
                rows.append(parts[0][0]); states.append(parts[0][1])
            else:
                mean = torch.stack([p[0] for p in parts]).mean(0)
                rows.append(mean); states.append(_mean_states([p[1] for p in parts], mean))
    qkv = torch.cat(rows, 0)
    qkv_st = _cat_states(states)

    # linear_proj [hidden, heads * d]: head h's own half <- o_proj's columns of head h
    ratio = math.sqrt(heads * d / o.shape[1])
    proj = torch.zeros(o.shape[0], heads * d)
    m_u = torch.zeros(o.shape[0], heads * d)
    rms = o.pow(2).mean(dim=1, keepdim=True).sqrt()
    for h in range(heads):
        own = slice(h * d + (0 if h % 2 == 0 else dc), h * d + (dc if h % 2 == 0 else d))
        other = slice(h * d + (dc if h % 2 == 0 else 0), h * d + (d if h % 2 == 0 else dc))
        proj[:, own] = o[:, h * dc:(h + 1) * dc]
        proj[:, other] = torch.randn(o.shape[0], dc, generator=gen) * (eps * rms)
        m_u[:, own] = st["o"]["m_u"][:, h * dc:(h + 1) * dc] / ratio
    proj_st = {"g": proj.norm(dim=1, keepdim=True), "m_g": st["o"]["m_g"].clone(), "v_g": st["o"]["v_g"].clone(),
               "v_norm": st["o"]["v_norm"] * ratio, "m_u": m_u}

    out = {"branch.linear_qkv.weight": qkv, "branch.linear_proj.weight": proj}
    out.update({f"branch.linear_qkv.weight.{s}": t for s, t in qkv_st.items()})
    out.update({f"branch.linear_proj.weight.{s}": t for s, t in proj_st.items()})
    for n in ("q_layernorm", "k_layernorm"):
        out[f"branch.{n}.weight"] = torch.full((d,), float(ref_gains[n]))
        out[f"branch.{n}.weight.exp_avg"] = torch.zeros(d)
        out[f"branch.{n}.weight.exp_avg_sq"] = torch.full((d,), float(ref_gains[f"{n}.exp_avg_sq"]))
    out.update(_gamma())
    return out


def kvshare_branch(att: dict, *, heads: int, d: int, groups: int, eps: float, gen: torch.Generator) -> dict:
    """New tensors of a kvshare branch (names relative to ``self_attention.``): the keys / values are the
    source attention layer's at run time, so only ``linear_qkv`` = per query group [q heads | gate heads] x d,
    ``q_layernorm`` and ``linear_proj`` -- new, random (user 2026-11-10: q / o learnt from scratch):
      q rows       random directions x the median norm of the source's q rows
      gate rows    random directions x eps x that norm (sigmoid ~ 0.5), as the attention branch
      linear_proj  random directions x the median row norm of the source's linear_proj
      q_layernorm  the source's gains (the temperature its keys were trained with), its exp_avg_sq
      MuOwn states fresh rows: g = row norm, v_norm = the median of the source's rows, momenta 0

    ``att``: the source attention layer's ``linear_qkv.weight`` / ``linear_proj.weight`` (+ MuOwn states
    ``<name>.<state>``) and ``q_layernorm.weight`` (+ ``.exp_avg_sq``)."""
    qkv, proj = att["linear_qkv.weight"], att["linear_proj.weight"]
    hidden, hpg = qkv.shape[1], heads // groups
    per = (2 * hpg + 2) * d  # the source's rows per KV group: [q | gate | k | v]
    if heads % groups or qkv.shape[0] != groups * per or proj.shape[1] != heads * d:
        raise ValueError(f"kvshare_branch: source linear_qkv {tuple(qkv.shape)} / linear_proj {tuple(proj.shape)} "
                         f"is not {heads} gated heads x {d} in {groups} groups")
    q_rows = torch.cat([qkv[g * per:g * per + hpg * d] for g in range(groups)])
    q_vrows = torch.cat([att["linear_qkv.weight.v_norm"][g * per:g * per + hpg * d] for g in range(groups)])
    q_norm, q_vref = q_rows.norm(dim=1).median(), q_vrows.median()
    rows, states = [], []
    for _ in range(groups):
        q = _noise_rows(hpg * d, hidden, q_norm, gen)
        gate = _noise_rows(hpg * d, hidden, eps * q_norm, gen)
        rows += [q, gate]
        states += [_new_rows_states(q, q_vref), _new_rows_states(gate, q_vref)]
    w = torch.cat(rows, 0)
    o = _noise_rows(proj.shape[0], heads * d, proj.norm(dim=1).median(), gen)
    out = {"branch.linear_qkv.weight": w, "branch.linear_proj.weight": o}
    out.update({f"branch.linear_qkv.weight.{s}": t for s, t in _cat_states(states).items()})
    out.update({f"branch.linear_proj.weight.{s}": t
                for s, t in _new_rows_states(o, att["linear_proj.weight.v_norm"].median()).items()})
    out["branch.q_layernorm.weight"] = att["q_layernorm.weight"].clone()
    out["branch.q_layernorm.weight.exp_avg"] = torch.zeros(d)
    out["branch.q_layernorm.weight.exp_avg_sq"] = att["q_layernorm.weight.exp_avg_sq"].clone()
    out.update(_gamma())
    return out


def attention_branch_random(att: dict, *, heads: int, d: int, groups: int, eps: float,
                            gen: torch.Generator) -> dict:
    """New tensors of an attention branch initialized at RANDOM (user 2026-11-10: "separate qkvo parameters
    initialized randomly", after the arms started from complex_kda's projections did not help): the layout of
    attention_branch (per KV group [q heads | gate heads | k | v] x d, q / k norm, linear_proj), every row a
    random direction at the reference attention layer's median row norm of its kind (q, k, v rows; gate rows x
    eps: sigmoid ~ 0.5; linear_proj rows), the reference's q / k norm gains; fresh MuOwn states (g = row norm,
    v_norm = the reference's median of that kind, momenta 0). ``att``: the reference attention layer's tensors
    (KVSHARE_SRC + k_layernorm)."""
    qkv, proj = att["linear_qkv.weight"], att["linear_proj.weight"]
    hidden, hpg = qkv.shape[1], heads // groups
    # the reference's KV groups: its rows are (2 heads + 2 groups) x d
    g_ref = (qkv.shape[0] // d - 2 * heads) // 2
    hpg_ref = heads // g_ref if g_ref > 0 else 0
    per = (2 * hpg_ref + 2) * d
    if g_ref <= 0 or heads % groups or heads % g_ref or qkv.shape[0] != g_ref * per or proj.shape[1] != heads * d:
        raise ValueError(f"attention_branch_random: reference linear_qkv {tuple(qkv.shape)} / linear_proj "
                         f"{tuple(proj.shape)} is not {heads} gated heads x {d}")
    vn = att["linear_qkv.weight.v_norm"]

    def kind(lo, hi):  # rows [lo, hi) of every reference group
        return (torch.cat([qkv[g * per + lo:g * per + hi] for g in range(g_ref)]).norm(dim=1).median(),
                torch.cat([vn[g * per + lo:g * per + hi] for g in range(g_ref)]).median())

    q_n, q_v = kind(0, hpg_ref * d)
    k_n, k_v = kind(2 * hpg_ref * d, (2 * hpg_ref + 1) * d)
    v_n, v_v = kind((2 * hpg_ref + 1) * d, (2 * hpg_ref + 2) * d)
    rows, states = [], []
    for _ in range(groups):
        for n_rows, norm, vref in ((hpg * d, q_n, q_v), (hpg * d, eps * q_n, q_v), (d, k_n, k_v), (d, v_n, v_v)):
            w = _noise_rows(n_rows, hidden, norm, gen)
            rows.append(w); states.append(_new_rows_states(w, vref))
    w = torch.cat(rows, 0)
    o = _noise_rows(proj.shape[0], heads * d, proj.norm(dim=1).median(), gen)
    out = {"branch.linear_qkv.weight": w, "branch.linear_proj.weight": o}
    out.update({f"branch.linear_qkv.weight.{s}": t for s, t in _cat_states(states).items()})
    out.update({f"branch.linear_proj.weight.{s}": t
                for s, t in _new_rows_states(o, att["linear_proj.weight.v_norm"].median()).items()})
    for n in ("q_layernorm", "k_layernorm"):
        out[f"branch.{n}.weight"] = att[f"{n}.weight"].clone()
        out[f"branch.{n}.weight.exp_avg"] = torch.zeros(d)
        out[f"branch.{n}.weight.exp_avg_sq"] = att[f"{n}.weight.exp_avg_sq"].clone()
    out.update(_gamma())
    return out


def attention_branch_attn_copy(att: dict, layer_norm: torch.Tensor | None) -> tuple[dict, dict]:
    """New tensors of an attention branch that is a COPY of a stack attention layer (``--init attn``), names
    relative to ``self_attention.``, and fold statistics. ``att``: the source layer's ATTN_COPY_SRC tensors;
    ``layer_norm``: the branch layer's input-norm gain (None: no fold)."""
    qkv, st = att["linear_qkv.weight"].clone(), {s: att[f"linear_qkv.weight.{s}"].clone() for s in MUOWN}
    stats = {}
    if layer_norm is not None:
        g_attn = att["linear_qkv.layer_norm_weight"].float()
        g_layer = layer_norm.float()
        tiny = g_layer.abs() < 1e-6 * g_layer.abs().max()
        if tiny.any():
            raise SystemExit(f"--init attn: {int(tiny.sum())} input-norm gains of the branch layer are ~0: cannot "
                             "fold the source's norm (use --no-fold-norm)")
        r = g_attn / g_layer
        old = qkv.norm(dim=1, keepdim=True)
        qkv = qkv * r[None, :]
        new = qkv.norm(dim=1, keepdim=True)
        ratio = old / new.clamp_min(1e-30)  # g / g'
        st["g"] = st["g"] * (new / old.clamp_min(1e-30))
        st["m_g"] = st["m_g"] * ratio
        st["v_g"] = st["v_g"] * ratio.pow(2)
        st["m_u"] = st["m_u"] / r[None, :]
        stats = {"norm_ratio_min": r.min().item(), "norm_ratio_median": r.median().item(),
                 "norm_ratio_max": r.max().item(), "row_norm_scale_median": (1 / ratio).median().item()}
    out = {"branch.linear_qkv.weight": qkv, "branch.linear_proj.weight": att["linear_proj.weight"].clone()}
    out.update({f"branch.linear_qkv.weight.{s}": t for s, t in st.items()})
    out.update({f"branch.linear_proj.weight.{s}": att[f"linear_proj.weight.{s}"].clone() for s in MUOWN})
    for n in ("q_layernorm", "k_layernorm"):
        out[f"branch.{n}.weight"] = att[f"{n}.weight"].clone()
        for s in ADAM:
            out[f"branch.{n}.weight.{s}"] = att[f"{n}.weight.{s}"].clone()
    out.update(_gamma())
    return out, stats


KVSHARE_SRC = (["linear_qkv.weight"] + [f"linear_qkv.weight.{s}" for s in MUOWN] + ["linear_proj.weight"]
               + [f"linear_proj.weight.{s}" for s in MUOWN] + ["q_layernorm.weight", "q_layernorm.weight.exp_avg_sq"])


RANDOM_ATTN_SRC = KVSHARE_SRC + ["k_layernorm.weight", "k_layernorm.weight.exp_avg_sq"]


# --init attn: everything of the source attention layer the branch copies, and its input-norm gain
ATTN_COPY_SRC = (["linear_qkv.weight"] + [f"linear_qkv.weight.{s}" for s in MUOWN] + ["linear_proj.weight"]
                 + [f"linear_proj.weight.{s}" for s in MUOWN]
                 + [f"{n}.weight{x}" for n in ("q_layernorm", "k_layernorm") for x in ("", ".exp_avg", ".exp_avg_sq")]
                 + ["linear_qkv.layer_norm_weight"])
LAYER_NORM = "model.module.decoder.layers.{l}.input_layernorm.weight"  # a complex_kda layer's input norm


KRON_COPY_MUOWN = ("f_proj.0.weight", "f_proj.1.weight", "g_proj.0.weight", "g_proj.1.weight", "o_proj.weight")
KRON_COPY_ADAM = ("A_log", "dt_bias", "g_proj.1.bias", "o_norm.weight", "b_proj.weight")


def kronlstm_branch(src: dict, *, b_bias: bool = False) -> dict:
    """New tensors of a KronLSTM branch (names relative to ``self_attention.``)."""
    out = {}
    out["branch.layer.qkv_proj.weight"] = torch.cat([src[f"layer.{n}_proj.weight"] for n in "qkv"], 0)
    for s in MUOWN:
        out[f"branch.layer.qkv_proj.weight.{s}"] = torch.cat([src[f"layer.{n}_proj.weight.{s}"] for n in "qkv"], 0)
    out["branch.layer.qkv_conv1d.weight"] = torch.cat([src[f"layer.{n}_conv1d.weight"] for n in "qkv"], 0)
    for s in ADAM:
        out[f"branch.layer.qkv_conv1d.weight.{s}"] = torch.cat([src[f"layer.{n}_conv1d.weight.{s}"] for n in "qkv"], 0)
    for n in KRON_COPY_MUOWN:
        out[f"branch.layer.{n}"] = src[f"layer.{n}"].clone()
        for s in MUOWN:
            out[f"branch.layer.{n}.{s}"] = src[f"layer.{n}.{s}"].clone()
    for n in KRON_COPY_ADAM:
        out[f"branch.layer.{n}"] = src[f"layer.{n}"].clone()
        for s in ADAM:
            out[f"branch.layer.{n}.{s}"] = src[f"layer.{n}.{s}"].clone()
    nh = src["layer.b_proj.weight"].shape[0]
    if b_bias:  # erase gate bias (neg_eigval_init_bias on): 0 = complex_kda's beta
        out["branch.layer.b_proj.bias"] = torch.zeros(nh)
        out["branch.layer.b_proj.bias.exp_avg"] = torch.zeros(nh)
        out["branch.layer.b_proj.bias.exp_avg_sq"] = torch.zeros(nh)
    # write (target) gate = the erase gate: the delta rule; the two separate in training (decouple_erase)
    out["branch.layer.c_proj.weight"] = src["layer.b_proj.weight"].clone()
    for s in ADAM:
        out[f"branch.layer.c_proj.weight.{s}"] = src[f"layer.b_proj.weight.{s}"].clone()
    out.update(_gamma())
    return out


def _gamma(value: float = 0.0):
    return {"branch_mix.gamma": torch.full((1,), float(value)), "branch_mix.gamma.exp_avg": torch.zeros(1),
            "branch_mix.gamma.exp_avg_sq": torch.zeros(1)}


def gamma_init(spec: str | None, layer: int) -> float:
    """``--gamma-init``: one value for every layer, or ``layer:value,...`` (layers not listed: 0). Under
    ckda_branch_crossfade the branch's weight is (1 - alpha)(1 + gamma): ratio - 1 starts the branch at the
    complex_kda mixer's output RMS (the measured rms_ckda / rms_branch), -1 at weight 0 (the drop control)."""
    if not spec:
        return 0.0
    if ":" not in spec:
        return float(spec)
    table = {int(k): float(v) for k, v in (x.split(":") for x in spec.split(","))}
    return table.get(layer, 0.0)


def _key(layer: int, rel: str) -> str:
    """Checkpoint key of ``rel`` (a parameter, or ``<parameter>.<state>``) of layer ``layer``."""
    for s in MUOWN + ADAM:
        if rel.endswith("." + s):
            return OPT.format(l=layer, n=rel[: -len(s) - 1], s=s)
    return MODEL.format(l=layer, n=rel)


def _like_dtype(smd, layer: int, rel: str, t: torch.Tensor) -> torch.dtype:
    """The stored dtype of the source layer's tensor of the same kind: q_proj (2-D weights and their
    MuOwn states) or A_log (1-D parameters and Adam states)."""
    state = next((s for s in MUOWN + ADAM if rel.endswith("." + s)), None)
    ref = "layer.q_proj.weight" if (state in MUOWN or (state is None and t.dim() == 2)) else "layer.A_log"
    if state in ADAM and t.dim() >= 2:
        ref = "layer.b_proj.weight" if t.dim() == 2 else "layer.q_conv1d.weight"
    return smd[_key(layer, ref + (f".{state}" if state else ""))].properties.dtype


def _src_names(branch: str) -> list[str]:
    names = []
    for n in "qkv":
        names += [f"layer.{n}_proj.weight"] + [f"layer.{n}_proj.weight.{s}" for s in MUOWN]
    names += ["layer.o_proj.weight"] + [f"layer.o_proj.weight.{s}" for s in MUOWN]
    if branch == "kronlstm":
        for n in "qkv":
            names += [f"layer.{n}_conv1d.weight"] + [f"layer.{n}_conv1d.weight.{s}" for s in ADAM]
        for n in KRON_COPY_MUOWN[:-1]:
            names += [f"layer.{n}"] + [f"layer.{n}.{s}" for s in MUOWN]
        for n in KRON_COPY_ADAM:
            names += [f"layer.{n}"] + [f"layer.{n}.{s}" for s in ADAM]
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------------------------------------
# selftest (CPU)
# ---------------------------------------------------------------------------------------------


def _fake_src(heads, dc, hidden, gen):
    src = {}
    for n in "qkvo":
        w = torch.randn(heads * dc if n != "o" else hidden, hidden if n != "o" else heads * dc, generator=gen)
        src[f"layer.{n}_proj.weight"] = w
        r, c = w.shape
        src[f"layer.{n}_proj.weight.g"] = w.norm(dim=1, keepdim=True)
        for s in ("m_g", "v_g"):
            src[f"layer.{n}_proj.weight.{s}"] = torch.randn(r, 1, generator=gen)
        src[f"layer.{n}_proj.weight.v_norm"] = torch.rand(r, 1, generator=gen) + 0.5
        src[f"layer.{n}_proj.weight.m_u"] = torch.randn(r, c, generator=gen)
    for n in "qkv":
        src[f"layer.{n}_conv1d.weight"] = torch.randn(heads * dc, 1, 4, generator=gen)
        for s in ADAM:
            src[f"layer.{n}_conv1d.weight.{s}"] = torch.rand(heads * dc, 1, 4, generator=gen)
    shapes = {"f_proj.0.weight": (dc, hidden), "f_proj.1.weight": (heads * dc, dc), "g_proj.0.weight": (dc, hidden),
              "g_proj.1.weight": (heads * dc, dc), "A_log": (heads,), "dt_bias": (heads * dc,),
              "g_proj.1.bias": (heads * dc,), "o_norm.weight": (dc,), "b_proj.weight": (heads, hidden)}
    for n, shp in shapes.items():
        src[f"layer.{n}"] = torch.randn(*shp, generator=gen)
        for s in (MUOWN if n in KRON_COPY_MUOWN else ADAM):
            src[f"layer.{n}.{s}"] = torch.randn(*(shp if s in ("m_u",) + ADAM else (shp[0], 1)), generator=gen)
    return src


def _fake_attention(heads, d, hidden, groups, gen):
    """A source attention layer's kvshare inputs (KVSHARE_SRC names)."""
    att = {}
    for n, shp in (("linear_qkv", (groups * (2 * heads // groups + 2) * d, hidden)), ("linear_proj", (hidden, heads * d))):
        w = torch.randn(*shp, generator=gen) * (0.5 if n == "linear_proj" else 2.0)
        att[f"{n}.weight"] = w
        att[f"{n}.weight.g"] = w.norm(dim=1, keepdim=True)
        for s in ("m_g", "v_g"):
            att[f"{n}.weight.{s}"] = torch.randn(shp[0], 1, generator=gen)
        att[f"{n}.weight.v_norm"] = torch.rand(shp[0], 1, generator=gen) + 0.5
        att[f"{n}.weight.m_u"] = torch.randn(*shp, generator=gen)
    att["q_layernorm.weight"] = torch.rand(d, generator=gen) + 0.5
    att["q_layernorm.weight.exp_avg_sq"] = torch.rand(d, generator=gen) * 1e-6
    return att


def selftest() -> int:
    gen = torch.Generator().manual_seed(0)
    heads, dc, hidden = 8, 8, 32
    d = 2 * dc
    src = _fake_src(heads, dc, hidden, gen)
    gains = {"q_layernorm": 0.7, "k_layernorm": 0.9, "q_layernorm.exp_avg_sq": 1e-6, "k_layernorm.exp_avg_sq": 2e-6}
    x = torch.randn(5, hidden, generator=gen)
    q, k = src["layer.q_proj.weight"], src["layer.k_proj.weight"]
    for groups in (heads // 2, 2):
        out = attention_branch(src, gains, heads=heads, d=d, groups=groups, eps=0.0, gen=gen)
        hpg = heads // groups
        qkv = out["branch.linear_qkv.weight"]
        assert qkv.shape == (groups * (2 * hpg + 2) * d, hidden), qkv.shape
        for s in MUOWN:
            assert out[f"branch.linear_qkv.weight.{s}"].shape[0] == qkv.shape[0]
        per = (2 * hpg + 2) * d
        y = qkv @ x.T  # [rows, tokens]
        for g in range(groups):
            blk = y[g * per:(g + 1) * per]
            kk = blk[2 * hpg * d:(2 * hpg + 1) * d]
            for i in range(hpg):
                h = g * hpg + i
                qh = blk[i * d:(i + 1) * d]
                # with eps = 0 the pad is zero: q'_h . k'_g = q_h . (h's half of the group key)
                half = slice(0, dc) if h % 2 == 0 else slice(dc, d)
                assert torch.allclose(qh[half], q[h * dc:(h + 1) * dc] @ x.T, atol=1e-5)
                assert torch.count_nonzero(qh[slice(dc, d) if h % 2 == 0 else slice(0, dc)]) == 0
                if groups == heads // 2:  # merged pair, no pooling: exactly the original head's scores
                    want = (q[h * dc:(h + 1) * dc] @ x.T).T @ (k[h * dc:(h + 1) * dc] @ x.T)
                    got = qh.T @ kk
                    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()
            if groups == 2:  # pooled keys: the mean of the group's merged heads
                per_m = heads // 2 // groups
                mean_rows = torch.stack([torch.cat([k[(2 * j) * dc:(2 * j + 1) * dc], k[(2 * j + 1) * dc:(2 * j + 2) * dc]])
                                         for j in range(g * per_m, (g + 1) * per_m)]).mean(0)
                assert torch.allclose(qkv[g * per + 2 * hpg * d:g * per + (2 * hpg + 1) * d], mean_rows, atol=1e-6)
        proj = out["branch.linear_proj.weight"]
        o = src["layer.o_proj.weight"]
        for h in range(heads):
            own = slice(h * d, h * d + dc) if h % 2 == 0 else slice(h * d + dc, (h + 1) * d)
            assert torch.equal(proj[:, own], o[:, h * dc:(h + 1) * dc])
        assert torch.allclose(out["branch.linear_proj.weight.v_norm"], src["layer.o_proj.weight.v_norm"] * math.sqrt(2))
        assert out["branch_mix.gamma"].item() == 0.0
        assert torch.all(out["branch.q_layernorm.weight"] == 0.7)
    # eps > 0: the pads are small but non-zero (MuOwn needs a direction)
    out = attention_branch(src, gains, heads=heads, d=d, groups=2, eps=1e-3, gen=gen)
    pad = out["branch.linear_qkv.weight"][dc:d]
    assert 0 < pad.norm(dim=1).max() < 1e-2 * q.norm(dim=1).median()
    assert torch.all(out["branch.linear_qkv.weight.v_norm"][dc:d] > 0)
    # kronlstm: stacked projections and kernels, copies, the new write gate at 0
    kr = kronlstm_branch(src)
    assert torch.equal(kr["branch.layer.qkv_proj.weight"], torch.cat([q, k, src["layer.v_proj.weight"]]))
    assert torch.equal(kr["branch.layer.qkv_conv1d.weight"][heads * dc:2 * heads * dc], src["layer.k_conv1d.weight"])
    assert torch.equal(kr["branch.layer.o_proj.weight.m_u"], src["layer.o_proj.weight.m_u"])
    assert torch.equal(kr["branch.layer.c_proj.weight"], src["layer.b_proj.weight"])
    assert "branch.layer.b_proj.bias" not in kr and kronlstm_branch(src, b_bias=True)["branch.layer.b_proj.bias"].abs().sum() == 0
    # kvshare: random q / gate / o at the source attention's row norms, its q norm gains; no k / v rows
    att = _fake_attention(heads, d, hidden, 2, gen)
    kv = kvshare_branch(att, heads=heads, d=d, groups=2, eps=1e-3, gen=gen)
    w, hpg = kv["branch.linear_qkv.weight"], heads // 2
    assert w.shape == (2 * heads * d, hidden) and kv["branch.linear_proj.weight"].shape == (hidden, heads * d)
    assert set(k for k in kv if not k.startswith("branch_mix")) == (
        {"branch.linear_qkv.weight", "branch.linear_proj.weight", "branch.q_layernorm.weight",
         "branch.q_layernorm.weight.exp_avg", "branch.q_layernorm.weight.exp_avg_sq"}
        | {f"branch.{n}.weight.{s}" for n in ("linear_qkv", "linear_proj") for s in MUOWN})
    per = (2 * hpg + 2) * d
    src_q = torch.cat([att["linear_qkv.weight"][g * per:g * per + hpg * d] for g in range(2)]).norm(dim=1).median()
    for g in range(2):
        qn = w[g * 2 * hpg * d:(g * 2 + 1) * hpg * d].norm(dim=1)
        gn = w[(g * 2 + 1) * hpg * d:(g + 1) * 2 * hpg * d].norm(dim=1)
        assert torch.allclose(qn, src_q.expand_as(qn)) and torch.allclose(gn, 1e-3 * src_q.expand_as(gn))
    assert torch.allclose(kv["branch.linear_proj.weight"].norm(dim=1),
                          att["linear_proj.weight"].norm(dim=1).median().expand(hidden))
    assert torch.equal(kv["branch.q_layernorm.weight"], att["q_layernorm.weight"])
    assert torch.equal(kv["branch.linear_qkv.weight.g"], w.norm(dim=1, keepdim=True))
    assert not kv["branch.linear_qkv.weight.m_u"].any() and kv["branch_mix.gamma"].item() == 0.0
    # random-init attention: the attention_branch layout, every row random at the reference's row norm of its kind
    att["k_layernorm.weight"] = torch.rand(d, generator=gen) + 0.5
    att["k_layernorm.weight.exp_avg_sq"] = torch.rand(d, generator=gen) * 1e-6
    for groups in (2, 4):
        ra = attention_branch_random(att, heads=heads, d=d, groups=groups, eps=1e-3, gen=gen)
        hpg_b = heads // groups
        w = ra["branch.linear_qkv.weight"]
        assert w.shape == (groups * (2 * hpg_b + 2) * d, hidden)
        per_b, per_r = (2 * hpg_b + 2) * d, (2 * (heads // 2) + 2) * d
        ref_k = torch.cat([att["linear_qkv.weight"][g * per_r + heads * d:g * per_r + heads * d + d] for g in range(2)])
        blk_k = w[2 * hpg_b * d:(2 * hpg_b + 1) * d]  # group 0's k rows
        assert torch.allclose(blk_k.norm(dim=1), ref_k.norm(dim=1).median().expand(d))
        assert torch.equal(ra["branch.k_layernorm.weight"], att["k_layernorm.weight"])
        assert ra["branch.linear_proj.weight"].shape == (hidden, heads * d)
        assert not ra["branch.linear_qkv.weight.m_u"].any()
    # --init attn: a copy of the attention layer, its input-norm gain folded into linear_qkv's columns
    att = _fake_attention(heads, d, hidden, 2, gen)
    att["k_layernorm.weight"] = torch.rand(d, generator=gen) + 0.5
    for n in ("q_layernorm", "k_layernorm"):
        att[f"{n}.weight.exp_avg"] = torch.randn(d, generator=gen) * 1e-4
        att[f"{n}.weight.exp_avg_sq"] = torch.rand(d, generator=gen) * 1e-6
    att["linear_qkv.layer_norm_weight"] = torch.rand(hidden, generator=gen) + 0.5
    g_layer = torch.rand(hidden, generator=gen) + 0.2
    cp, stats = attention_branch_attn_copy(att, g_layer)
    xn = x / x.pow(2).mean(dim=1, keepdim=True).add(1e-6).sqrt()  # RMSNorm without gain
    want_qkv = (xn * att["linear_qkv.layer_norm_weight"]) @ att["linear_qkv.weight"].T
    got_qkv = (xn * g_layer) @ cp["branch.linear_qkv.weight"].T
    assert torch.allclose(got_qkv, want_qkv, rtol=1e-5, atol=1e-5), (got_qkv - want_qkv).abs().max()
    assert torch.allclose(cp["branch.linear_qkv.weight.g"], cp["branch.linear_qkv.weight"].norm(dim=1, keepdim=True),
                          rtol=1e-5)
    assert torch.equal(cp["branch.linear_qkv.weight.v_norm"], att["linear_qkv.weight.v_norm"])
    for n in ("linear_proj.weight", "q_layernorm.weight", "k_layernorm.weight", "k_layernorm.weight.exp_avg_sq",
              "linear_proj.weight.m_u", "linear_proj.weight.v_norm", "q_layernorm.weight.exp_avg"):
        assert torch.equal(cp[f"branch.{n}"], att[n]), n
    assert cp["branch_mix.gamma"].item() == 0.0 and stats["norm_ratio_min"] > 0
    # the moments follow the chain rule of the same function: dL/dW' = dL/dW / r (columns), dL/dg' = dL/dg g / g'
    r = att["linear_qkv.layer_norm_weight"] / g_layer
    assert torch.allclose(cp["branch.linear_qkv.weight.m_u"] * r, att["linear_qkv.weight.m_u"], rtol=1e-5, atol=1e-6)
    nofold, st0 = attention_branch_attn_copy(att, None)
    assert torch.equal(nofold["branch.linear_qkv.weight"], att["linear_qkv.weight"]) and not st0
    assert torch.equal(nofold["branch.linear_qkv.weight.m_g"], att["linear_qkv.weight.m_g"])
    assert gamma_init(None, 9) == 0.0 and gamma_init("-1", 25) == -1.0
    assert gamma_init("9:1.5,25:0.5", 25) == 0.5 and gamma_init("9:1.5", 41) == 0.0
    # keys
    assert _key(9, "branch.linear_qkv.weight.m_u") == OPT.format(l=9, n="branch.linear_qkv.weight", s="m_u")
    assert _key(9, "branch_mix.gamma") == MODEL.format(l=9, n="branch_mix.gamma")
    _selftest_apply(src, heads, dc, hidden)
    _selftest_kvdup_chain(src, heads, dc, hidden)
    print("selftest ok")
    return 0


def _selftest_kvdup_chain(src, heads, dc, hidden):
    """kv8post4 (broad session): duplicate_kv_groups.py apply (2 -> 4 KV groups) on a tiny DCP checkpoint, then
    --init attn branches from the duplicated stack layers at the stack's new KV groups. The branch must be the
    duplicated source layer (verbatim + norm fold, its split moments), compute the 2-group source attention of
    RMSNorm(x) g_attn exactly (fp64 reference with gate, qk norm, XSA), and --groups 2 must be refused."""
    import tempfile

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import duplicate_kv_groups as kvd

    d, g_new, branch_layers, sources = 2 * dc, 4, (9, 25), (0, 17)
    sd = {}
    for layer in branch_layers:  # complex_kda layers: fp32 everywhere, so the function check is not bf16-limited
        for rel, t in src.items():
            sd[_key(layer, rel)] = t.clone()
        sd[LAYER_NORM.format(l=layer)] = torch.linspace(1.4, 0.6, hidden) + 0.01 * layer
    for layer in ATTN_LAYERS:
        g = torch.Generator().manual_seed(100 + layer)
        for rel, t in _fake_attention(heads, d, hidden, 2, g).items():
            sd[_key(layer, rel)] = t * (0.05 if rel == "linear_qkv.weight" else 1.0)
        sd[_key(layer, "k_layernorm.weight")] = torch.rand(d, generator=g) + 0.5
        for n in ("q_layernorm", "k_layernorm"):
            sd[_key(layer, f"{n}.weight.exp_avg")] = torch.randn(d, generator=g) * 1e-4
            sd[_key(layer, f"{n}.weight.exp_avg_sq")] = torch.rand(d, generator=g) * 1e-6
        sd[_key(layer, "linear_qkv.layer_norm_weight")] = torch.rand(hidden, generator=g) + 0.5
    sd["args"] = argparse.Namespace(num_attention_heads=heads, kv_channels=d, num_query_groups=2,
                                    group_query_attention=True, attention_output_gate=True)
    sd["iteration"] = 33200
    with tempfile.TemporaryDirectory() as tmp:
        dcp.save(sd, storage_writer=FileSystemWriter(f"{tmp}/src"), no_dist=True)
        assert kvd.main(["apply", "--src", f"{tmp}/src", "--dst", f"{tmp}/kv", "--groups", str(g_new)]) == 0
        assert kvd.main(["verify", "--dst", f"{tmp}/kv"]) == 0
        common = ["apply", "--src", f"{tmp}/kv", "--branch", "attention", "--layers", ",".join(map(str, branch_layers)),
                  "--init", "attn", "--kv-sources", ",".join(map(str, sources)), "--calibrate"]
        try:  # the source's KV groups only: 2 is the pre-duplication layout
            main(common[:3] + ["--dst", f"{tmp}/bad"] + common[3:] + ["--groups", "2"])
            raise AssertionError("--groups 2 on a 4-group stack was accepted")
        except SystemExit as e:
            assert "is not the source's KV groups" in str(e), e
        assert main(common[:3] + ["--dst", f"{tmp}/chain"] + common[3:] + ["--groups", str(g_new)]) == 0
        assert main(["verify", "--dst", f"{tmp}/chain"]) == 0
        smd = FileSystemReader(f"{tmp}/chain").read_metadata().state_dict_metadata
        kv_smd = FileSystemReader(f"{tmp}/kv").read_metadata().state_dict_metadata
        assert set(kv_smd) <= set(smd)
        for layer, s in zip(branch_layers, sources):
            names = [rel for rel in ATTN_COPY_SRC]
            dup = {rel: torch.empty(tuple(kv_smd[_key(s, rel)].size)) for rel in names}
            dcp.load({_key(s, r): t for r, t in dup.items()}, storage_reader=FileSystemReader(f"{tmp}/kv"), no_dist=True)
            new_rel = list(attention_branch_attn_copy(dup, sd[LAYER_NORM.format(l=layer)])[0])
            back = {rel: torch.empty(tuple(smd[_key(layer, rel)].size)) for rel in new_rel + ["branch_mix.calib"]}
            dcp.load({_key(layer, r): t for r, t in back.items()}, storage_reader=FileSystemReader(f"{tmp}/chain"),
                     no_dist=True)
            want, _ = attention_branch_attn_copy(dup, sd[LAYER_NORM.format(l=layer)])
            for rel in new_rel:
                assert torch.equal(back[rel], want[rel].to(back[rel].dtype)), (layer, rel)
            assert back["branch.linear_qkv.weight"].shape[0] == g_new * (2 * heads // g_new + 2) * d
            assert back["branch_mix.gamma"].item() == 0.0 and back["branch_mix.calib"].item() == 0.0
            # the moments of the K/V copies are the stack's split moments (kv8's), carried through the fold
            r = (sd[_key(s, "linear_qkv.layer_norm_weight")] / sd[LAYER_NORM.format(l=layer)])[None, :]
            torch.testing.assert_close(back["branch.linear_qkv.weight.m_u"] * r, dup["linear_qkv.weight.m_u"],
                                       rtol=1e-5, atol=1e-6)
            # function: branch(RMSNorm(x) g_layer) at 4 groups == source attention(RMSNorm(x) g_attn) at 2 groups
            x = torch.randn(7, hidden, dtype=torch.float64, generator=torch.Generator().manual_seed(layer))
            xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            ref = kvd.reference_attention(
                xn * sd[_key(s, "linear_qkv.layer_norm_weight")].double(), sd[_key(s, "linear_qkv.weight")].double(),
                sd[_key(s, "q_layernorm.weight")].double(), sd[_key(s, "k_layernorm.weight")].double(),
                sd[_key(s, "linear_proj.weight")].double(), heads=heads, groups=2, d=d)
            got = kvd.reference_attention(
                xn * sd[LAYER_NORM.format(l=layer)].double(), back["branch.linear_qkv.weight"].double(),
                back["branch.q_layernorm.weight"].double(), back["branch.k_layernorm.weight"].double(),
                back["branch.linear_proj.weight"].double(), heads=heads, groups=g_new, d=d)
            rel_err = ((got - ref).abs().max() / ref.abs().max()).item()
            assert rel_err < 1e-5, (layer, rel_err)  # fp32 storage of the folded weights
            print(f"kv-dup chain: branch {layer} (4 groups) <- attention {s} (2 groups): rel err {rel_err:.1e}")


def _selftest_apply(src, heads, dc, hidden):
    """apply() on a tiny DCP checkpoint: keys, dtypes, metadata and read-back."""
    import tempfile

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

    d = 2 * dc
    sd = {}
    for layer in (9, 25):
        for rel, t in src.items():
            # bf16 model weights, fp32 states: the dtypes must follow the source's
            sd[_key(layer, rel)] = t.to(torch.bfloat16) if _key(layer, rel).startswith("model.") else t.clone()
    for layer in ATTN_LAYERS:
        for rel, t in _fake_attention(heads, d, hidden, 2, torch.Generator().manual_seed(layer)).items():
            sd[_key(layer, rel)] = t.to(torch.bfloat16) if _key(layer, rel).startswith("model.") else t
        for n in ("q_layernorm", "k_layernorm"):
            sd[_key(layer, f"{n}.weight")] = torch.full((d,), 0.5 + layer / 100)
            sd[_key(layer, f"{n}.weight.exp_avg_sq")] = torch.full((d,), 1e-6)
            sd[_key(layer, f"{n}.weight.exp_avg")] = torch.full((d,), 1e-5)
        sd[_key(layer, "linear_qkv.layer_norm_weight")] = torch.linspace(0.5, 1.5, hidden).to(torch.bfloat16)
    for layer in (9, 25):
        sd[LAYER_NORM.format(l=layer)] = torch.linspace(1.5, 0.5, hidden).to(torch.bfloat16)
    sd["iteration"] = 1100
    with tempfile.TemporaryDirectory() as tmp:
        dcp.save(sd, storage_writer=FileSystemWriter(f"{tmp}/src"), no_dist=True)
        for branch, extra in (("attention", ["--groups", "2"]), ("kronlstm", []),
                              ("kvshare", ["--groups", "2", "--kv-sources", "0,17"]),
                              ("attention", ["--groups", "2", "--init", "random", "--kv-sources", "0,17"]),
                              ("attention", ["--groups", "2", "--init", "attn", "--kv-sources", "0,17",
                                             "--calibrate"])):
            dst = f"{tmp}/dst_{branch}" + ("_random" if "random" in extra else "_attn" if "attn" in extra else "")
            rc = main(["apply", "--src", f"{tmp}/src", "--dst", dst, "--branch", branch, "--layers", "9,25"] + extra)
            assert rc == 0
            smd = FileSystemReader(dst).read_metadata().state_dict_metadata
            assert set(sd) <= set(smd)
            if branch == "attention":
                k = _key(9, "branch.linear_qkv.weight")
                assert tuple(smd[k].size) == (2 * (heads // 2 * 2 + 2) * d, hidden), smd[k].size
                assert smd[k].properties.dtype == torch.bfloat16
                assert smd[_key(25, "branch.linear_qkv.weight.m_u")].properties.dtype == torch.float32
                assert smd[_key(9, "branch_mix.gamma")].properties.dtype == torch.bfloat16
                back = {_key(9, "branch.q_layernorm.weight"): torch.empty(d, dtype=torch.bfloat16)}
                dcp.load(back, storage_reader=FileSystemReader(dst), no_dist=True)
                if "random" in extra or "attn" in extra:  # the reference / source layer's (0 for layer 9) gains
                    want = sd[_key(0, "q_layernorm.weight")].float()
                else:  # the median gain of the stack's attention layers
                    want = torch.tensor([0.5 + x / 100 for x in ATTN_LAYERS]).median().expand(d)
                assert torch.allclose(back[_key(9, "branch.q_layernorm.weight")].float(), want, atol=1e-2)
            elif branch == "kvshare":
                k = _key(9, "branch.linear_qkv.weight")
                assert tuple(smd[k].size) == (2 * heads * d, hidden) and smd[k].properties.dtype == torch.bfloat16
                assert _key(9, "branch.k_layernorm.weight") not in smd
                back = {_key(25, "branch.q_layernorm.weight"): torch.empty(d, dtype=torch.bfloat16)}
                dcp.load(back, storage_reader=FileSystemReader(dst), no_dist=True)
                assert torch.equal(back[_key(25, "branch.q_layernorm.weight")],  # the source's gains (bf16 as A_log)
                                   sd[_key(17, "q_layernorm.weight")].to(torch.bfloat16))
            else:
                assert tuple(smd[_key(9, "branch.layer.qkv_proj.weight")].size) == (3 * heads * dc, hidden)
                assert tuple(smd[_key(9, "branch.layer.qkv_conv1d.weight")].size) == (3 * heads * dc, 1, 4)
            # the manifest, and a re-derivation from it reproduces every new tensor bit for bit
            import json

            doc = json.load(open(f"{dst}/{MANIFEST}"))
            assert doc["args"]["branch"] == branch and len(doc["new_tensors"]) == len(set(smd) - set(sd))
            assert main(["verify", "--dst", dst]) == 0
            # the source's own tensors are untouched (hard-linked shards, same metadata entries)
            back = {_key(9, "layer.q_proj.weight"): torch.empty(heads * dc, hidden, dtype=torch.bfloat16)}
            dcp.load(back, storage_reader=FileSystemReader(dst), no_dist=True)
            assert torch.equal(back[_key(9, "layer.q_proj.weight")], sd[_key(9, "layer.q_proj.weight")])


def gpucheck() -> int:
    """kronlstm_branch on a real complex_kda layer (the FLA fork, one GPU).

    The two recurrences differ (KronLSTM closes a scalar residual k^T S v along the Kronecker key k v^T,
    the delta rule the vector residual v - S^T k), so the layers agree exactly only on the first token,
    where the state is empty: that checks projections, convolution, gates and output path (normalize_v
    off). The difference over a sequence is reported, with and without the v normalization."""
    from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
    from fla.layers.kronlstm import KronLSTM

    torch.manual_seed(0)
    dev, hidden, heads, dc = "cuda", 512, 4, 128
    ckda = ComplexKimiDeltaAttention(hidden_size=hidden, head_dim=dc, num_heads=heads, allow_neg_eigval=True,
                                     gate="signed_sigmoid2", drop_silu=True, gate_init_style="shipped",
                                     output_gate="lowrank", layer_idx=0).to(dev)
    with torch.no_grad():  # trained-looking gates: random output-gate weights, spread decay
        for p in ckda.parameters():
            if p.dim() == 2:
                p.normal_(0, p.shape[1] ** -0.5)
    sd = {f"layer.{k}": v.detach().float().cpu() for k, v in ckda.state_dict().items()}
    for k in list(sd):  # optimizer states are not part of the check: zeros of the right shape
        if k.endswith(".weight") or k.endswith(".bias") or k in ("layer.A_log", "layer.dt_bias"):
            t = sd[k]
            for s in MUOWN:
                sd[f"{k}.{s}"] = torch.zeros(t.shape[0], *([1] if s != "m_u" else t.shape[1:]))
            for s in ADAM:
                sd[f"{k}.{s}"] = torch.zeros_like(t)
    new = kronlstm_branch(sd)
    for normalize_v in (False, True):
        kron = KronLSTM(hidden_size=hidden, k_head_dim=dc, v_head_dim=dc, num_k_heads=heads, num_v_heads=heads,
                        conv_size=4, normalize_v=normalize_v, allow_neg_eigval=True, use_target_gate=True,
                        decouple_erase=True, neg_eigval_init_bias=False, unit_write=False, drop_silu=True,
                        output_gate_rank="lowrank", decay_gate="vector", gate="signed_sigmoid2", layer_idx=0).to(dev)
        want = {k[len("branch.layer."):]: v for k, v in new.items()
                if k.startswith("branch.layer.") and k.rsplit(".", 1)[-1] not in MUOWN + ADAM}
        missing, unexpected = kron.load_state_dict({k: v.to(dev) for k, v in want.items()}, strict=False)
        assert not unexpected and not [m for m in missing if not m.endswith("_buffer")], (missing, unexpected)
        x = torch.randn(2, 256, hidden, device=dev)
        with torch.no_grad():
            a = ckda(x)[0].float()
            b = kron(x)[0].float()
        rel = ((a - b).norm() / a.norm()).item()
        rel0 = ((a[:, 0] - b[:, 0]).norm() / a[:, 0].norm()).item()
        print(f"normalize_v {normalize_v}: |kron - ckda| / |ckda| token 0 {rel0:.2e}, all tokens {rel:.2e} "
              f"(rms ckda {a.pow(2).mean().sqrt():.3e}, kron {b.pow(2).mean().sqrt():.3e})")
        if not normalize_v and rel0 > 1e-2:
            raise SystemExit("kronlstm_branch: KronLSTM's first token differs from complex_kda's")
    print("gpucheck ok")
    return 0


# ---------------------------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------------------------


def derive(a, md=None) -> dict:
    """The new tensors {checkpoint key: tensor} for the arguments ``a`` (deterministic: seed ``a.seed``)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    t0 = time.time()
    md = md if md is not None else FileSystemReader(a.src).read_metadata()
    smd = md.state_dict_metadata
    layers = [int(x) for x in a.layers.split(",")]
    gen = torch.Generator().manual_seed(a.seed)
    want = {}
    kv_sources = [int(x) for x in a.kv_sources.split(",")] if getattr(a, "kv_sources", None) else []
    random_init = getattr(a, "init", "copy") == "random"
    attn_copy = getattr(a, "init", "copy") == "attn"
    fold = attn_copy and not getattr(a, "no_fold_norm", False)
    if (random_init or attn_copy) and a.branch != "attention":
        raise SystemExit(f"--init {a.init}: attention branches only")
    if a.branch == "kvshare" or random_init or attn_copy:
        if len(kv_sources) != len(layers) or any(s not in ATTN_LAYERS or s >= l for s, l in zip(kv_sources, layers)):
            raise SystemExit(f"--kv-sources {kv_sources}: one earlier attention layer {ATTN_LAYERS} per layer {layers}")
        names = ATTN_COPY_SRC if attn_copy else RANDOM_ATTN_SRC if random_init else KVSHARE_SRC
        for src_layer in kv_sources:
            for rel in names:
                key = _key(src_layer, rel)
                want[key] = torch.empty(tuple(smd[key].size), dtype=smd[key].properties.dtype)
    if attn_copy:
        for layer in layers:
            key = LAYER_NORM.format(l=layer)
            if key not in smd or _key(layer, "layer.q_proj.weight") not in smd:
                raise SystemExit(f"layer {layer}: no complex_kda layer with an input norm ({key}) in {a.src}")
            want[key] = torch.empty(tuple(smd[key].size), dtype=smd[key].properties.dtype)
    for layer in (layers if a.branch != "kvshare" and not random_init and not attn_copy else []):
        for rel in _src_names(a.branch):
            key = _key(layer, rel)
            if key not in smd:
                raise SystemExit(f"{key}: not in {a.src} (is layer {layer} a complex_kda layer?)")
            want[key] = torch.empty(tuple(smd[key].size), dtype=smd[key].properties.dtype)
    if a.branch == "attention" and not attn_copy:
        for layer in ATTN_LAYERS:
            for n in ("q_layernorm", "k_layernorm"):
                for rel in (f"{n}.weight", f"{n}.weight.exp_avg_sq"):
                    want[_key(layer, rel)] = torch.empty(tuple(smd[_key(layer, rel)].size))
    dcp.load(want, storage_reader=FileSystemReader(a.src), no_dist=True)
    print(f"read {len(want)} tensors in {time.time() - t0:.0f} s", flush=True)
    replace = {}
    for i, layer in enumerate(layers):
        src = ({} if a.branch == "kvshare" or random_init or attn_copy else
               {rel: want[_key(layer, rel)].float() for rel in _src_names(a.branch)})
        g0 = gamma_init(getattr(a, "gamma_init", None), layer)
        if a.branch == "kvshare":
            d = smd[_key(ATTN_LAYERS[0], "q_layernorm.weight")].size[0]
            heads = smd[_key(ATTN_LAYERS[0], "linear_proj.weight")].size[1] // d
            att = {rel: want[_key(kv_sources[i], rel)].float() for rel in KVSHARE_SRC}
            new = kvshare_branch(att, heads=heads, d=d, groups=a.groups, eps=a.eps, gen=gen)
        elif attn_copy:
            att = {rel: want[_key(kv_sources[i], rel)].float() for rel in ATTN_COPY_SRC}
            if a.groups * (2 * (att["linear_proj.weight"].shape[1] // att["q_layernorm.weight"].shape[0]) // a.groups
                           + 2) * att["q_layernorm.weight"].shape[0] != att["linear_qkv.weight"].shape[0]:
                raise SystemExit(f"--init attn: --groups {a.groups} is not the source's KV groups "
                                 f"(linear_qkv {tuple(att['linear_qkv.weight'].shape)})")
            new, stats = attention_branch_attn_copy(att, want[LAYER_NORM.format(l=layer)] if fold else None)
            print(f"layer {layer} <- attention layer {kv_sources[i]}" + (f": norm fold {stats}" if stats else ""),
                  flush=True)
        elif random_init:
            d = smd[_key(ATTN_LAYERS[0], "q_layernorm.weight")].size[0]
            heads = smd[_key(ATTN_LAYERS[0], "linear_proj.weight")].size[1] // d
            att = {rel: want[_key(kv_sources[i], rel)].float() for rel in RANDOM_ATTN_SRC}
            new = attention_branch_random(att, heads=heads, d=d, groups=a.groups, eps=a.eps, gen=gen)
        elif a.branch == "attention":
            d = smd[_key(ATTN_LAYERS[0], "q_layernorm.weight")].size[0]
            heads = smd[_key(ATTN_LAYERS[0], "linear_proj.weight")].size[1] // d
            gains = {}
            for n in ("q_layernorm", "k_layernorm"):
                gains[n] = torch.cat([want[_key(x, f"{n}.weight")] for x in ATTN_LAYERS]).median().item()
                gains[f"{n}.exp_avg_sq"] = torch.cat([want[_key(x, f"{n}.weight.exp_avg_sq")]
                                                      for x in ATTN_LAYERS]).mean().item()
            new = attention_branch(src, gains, heads=heads, d=d, groups=a.groups, eps=a.eps, gen=gen)
        else:
            new = kronlstm_branch(src, b_bias=a.b_bias)
        if g0:
            new["branch_mix.gamma"] = torch.full((1,), g0)
        if getattr(a, "calibrate", False):  # ckda_branch_calibrate: measured on the first training forward
            new["branch_mix.calib"] = torch.zeros(1)
        for rel, t in new.items():
            key = _key(layer, rel)
            if key in smd:
                raise SystemExit(f"{key} already exists: the source has a branch already")
            replace[key] = t.to(_like_dtype(smd, layer, rel, t)).contiguous()
        print(f"layer {layer}: {len(new)} new tensors", flush=True)
    return replace


def apply(a) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dcp_surgery import write_surgery

    t0 = time.time()
    md = FileSystemReader(a.src).read_metadata()
    replace = derive(a, md)
    # every key is new: 64 MB chunks (the fallback, the source's median chunk, is tiny for small models and
    # split one test run's tensors into 86k chunks)
    write_surgery(a.src, a.dst, md, replace, tag="ckda_branch", chunk_bytes=64 << 20)
    # read back
    back = {k: torch.empty_like(v) for k, v in replace.items()}
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in back if not torch.equal(back[k], replace[k])]
    if bad:
        raise SystemExit(f"read-back mismatch: {bad[:5]}")
    manifest = write_manifest(a, replace)
    print(f"done in {time.time() - t0:.0f} s: {len(replace)} new keys in {a.dst}; manifest {manifest}", flush=True)
    return 0


MANIFEST = "ckda_branch_surgery.json"


def _tensor_sha256(t: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _code_state() -> dict:
    """The code that ran: the synced snapshot's SYNC_STAMP (commits + diff hashes of every repo), else git."""
    import subprocess

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    out = {"script": os.path.abspath(__file__), "root": root}
    stamp = os.path.join(root, "SYNC_STAMP")
    if os.path.exists(stamp):
        out["sync_stamp"] = open(stamp).read()
    try:
        out["git_head"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                         timeout=30).stdout.strip() or None
        out["git_dirty"] = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "scripts"],
                                               capture_output=True, text=True, timeout=60).stdout.strip())
    except Exception:  # no git on the node: the stamp is the record
        pass
    return out


def write_manifest(a, replace: dict) -> str:
    """<dst>/ckda_branch_surgery.json: what was done, from what, by which code, and the SHA-256 of every new
    tensor (a rerun with the same arguments and seed must reproduce them bit for bit)."""
    import hashlib
    import json
    import platform

    meta = os.path.join(a.src, ".metadata")
    h = hashlib.sha256()
    with open(meta, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    doc = {
        "tool": "add_ckda_branch.py apply",
        "args": {k: v for k, v in vars(a).items() if k != "cmd"},
        "source": os.path.abspath(a.src),
        "source_metadata_sha256": h.hexdigest(),
        "code": _code_state(),
        "torch": torch.__version__,
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "new_tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                            "sha256": _tensor_sha256(t)} for k, t in sorted(replace.items())},
    }
    path = os.path.join(a.dst, MANIFEST)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def verify(a) -> int:
    """Re-derive the new tensors from the manifest's source and arguments (no writing) and compare their
    SHA-256 with the manifest's: the surgery is reproducible when every one matches."""
    import json

    doc = json.load(open(os.path.join(a.dst, MANIFEST)))
    args = argparse.Namespace(**doc["args"])
    replace = derive(args)
    want = doc["new_tensors"]
    bad = [k for k, t in replace.items() if want.get(k, {}).get("sha256") != _tensor_sha256(t)]
    bad += [k for k in want if k not in replace]
    print(f"verify {a.dst}: {len(replace)} tensors, {len(bad)} differ" + (f": {bad[:5]}" if bad else ""))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("gpucheck", help="kronlstm mapping on a real complex_kda layer (FLA fork, one GPU)")
    v = sub.add_parser("verify", help="re-derive a surgery's new tensors from its manifest and compare hashes")
    v.add_argument("--dst", required=True, help="a surgery's output checkpoint (holds ckda_branch_surgery.json)")
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--branch", choices=["attention", "kronlstm", "kvshare"], required=True)
    p.add_argument("--groups", type=int, default=2,
                   help="attention: KV groups (8 = merged pairs, 2 = GQA); kvshare: the stack's (the source's)")
    p.add_argument("--init", choices=["copy", "random", "attn"], default="copy",
                   help="attention: copy = from complex_kda's projections; random = random rows at the reference "
                        "attention layers' (--kv-sources) row norms; attn = a copy of the attention layer "
                        "(--kv-sources) with all its optimizer states, its input-norm gain folded in")
    p.add_argument("--no-fold-norm", action="store_true",
                   help="--init attn: copy linear_qkv as is (do not fold g_attn / g_layer into its columns)")
    p.add_argument("--kv-sources", default=None,
                   help="kvshare: per --layers entry, the attention layer whose keys / values it reads (e.g. 0,17,33)")
    p.add_argument("--layers", required=True, help="comma-separated decoder layer indices (complex_kda layers)")
    p.add_argument("--eps", type=float, default=1e-3, help="relative norm of the new (noise) rows")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--b-bias", action="store_true", help="kronlstm: an erase-gate bias at 0 (neg_eigval_init_bias on)")
    p.add_argument("--calibrate", action="store_true",
                   help="write branch_mix.calib = 0 (Megatron ckda_branch_calibrate: measured on the first step)")
    p.add_argument("--gamma-init", default=None,
                   help="branch_mix.gamma at insertion: a value, or layer:value,... (crossfade: ratio - 1 matches "
                        "the complex_kda output RMS, -1 = weight 0)")
    a = ap.parse_args(argv)
    return {"selftest": selftest, "gpucheck": gpucheck, "verify": lambda: verify(a)}.get(a.cmd, lambda: apply(a))()


if __name__ == "__main__":
    sys.exit(main())
