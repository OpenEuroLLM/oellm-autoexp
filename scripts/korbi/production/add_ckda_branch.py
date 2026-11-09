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

**kronlstm** (``self_attention.branch.layer.*``): qkv_proj / qkv_conv1d = complex_kda's q / k / v
projections and convolution kernels stacked (exact: depthwise), f_proj / A_log / dt_bias / g_proj /
o_norm / o_proj copied, b_proj copied (erase s sigmoid(b), s = 2 as complex_kda's linear_beta_max 2; no bias:
neg_eigval_init_bias off), c_proj (write gate) = b_proj: with decouple_erase and without unit_write the write is
s c v, the update s sigmoid(b) (v - S k) -- complex_kda's delta rule, on the normalized v (the "hard break").

    python add_ckda_branch.py selftest
    PYTHONPATH=<fla fork> python add_ckda_branch.py gpucheck
    python add_ckda_branch.py apply --src <ckpt>/iter_0023300 --dst <new>/iter_0023300 \\
        --branch attention --groups 2 --layers 9,25,41 [--eps 1e-3] [--seed 1234]
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


def _gamma():
    return {"branch_mix.gamma": torch.zeros(1), "branch_mix.gamma.exp_avg": torch.zeros(1),
            "branch_mix.gamma.exp_avg_sq": torch.zeros(1)}


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
    # keys
    assert _key(9, "branch.linear_qkv.weight.m_u") == OPT.format(l=9, n="branch.linear_qkv.weight", s="m_u")
    assert _key(9, "branch_mix.gamma") == MODEL.format(l=9, n="branch_mix.gamma")
    _selftest_apply(src, heads, dc, hidden)
    print("selftest ok")
    return 0


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
        sd[_key(layer, "linear_proj.weight")] = torch.zeros(hidden, heads * d, dtype=torch.bfloat16)
        for n in ("q_layernorm", "k_layernorm"):
            sd[_key(layer, f"{n}.weight")] = torch.full((d,), 0.5 + layer / 100)
            sd[_key(layer, f"{n}.weight.exp_avg_sq")] = torch.full((d,), 1e-6)
    sd["iteration"] = 1100
    with tempfile.TemporaryDirectory() as tmp:
        dcp.save(sd, storage_writer=FileSystemWriter(f"{tmp}/src"), no_dist=True)
        for branch, extra in (("attention", ["--groups", "2"]), ("kronlstm", [])):
            dst = f"{tmp}/dst_{branch}"
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
                want = torch.tensor([0.5 + x / 100 for x in ATTN_LAYERS]).median()
                assert torch.allclose(back[_key(9, "branch.q_layernorm.weight")].float(), want.expand(d), atol=1e-2)
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
    for layer in layers:
        for rel in _src_names(a.branch):
            key = _key(layer, rel)
            if key not in smd:
                raise SystemExit(f"{key}: not in {a.src} (is layer {layer} a complex_kda layer?)")
            want[key] = torch.empty(tuple(smd[key].size), dtype=smd[key].properties.dtype)
    if a.branch == "attention":
        for layer in ATTN_LAYERS:
            for n in ("q_layernorm", "k_layernorm"):
                for rel in (f"{n}.weight", f"{n}.weight.exp_avg_sq"):
                    want[_key(layer, rel)] = torch.empty(tuple(smd[_key(layer, rel)].size))
    dcp.load(want, storage_reader=FileSystemReader(a.src), no_dist=True)
    print(f"read {len(want)} tensors in {time.time() - t0:.0f} s", flush=True)
    replace = {}
    for layer in layers:
        src = {rel: want[_key(layer, rel)].float() for rel in _src_names(a.branch)}
        if a.branch == "attention":
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
    p.add_argument("--branch", choices=["attention", "kronlstm"], required=True)
    p.add_argument("--groups", type=int, default=2, help="attention: KV groups (8 = merged pairs, 2 = GQA)")
    p.add_argument("--layers", required=True, help="comma-separated decoder layer indices (complex_kda layers)")
    p.add_argument("--eps", type=float, default=1e-3, help="relative norm of the new (noise) rows")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--b-bias", action="store_true", help="kronlstm: an erase-gate bias at 0 (neg_eigval_init_bias on)")
    a = ap.parse_args(argv)
    return {"selftest": selftest, "gpucheck": gpucheck, "verify": lambda: verify(a)}.get(a.cmd, lambda: apply(a))()


if __name__ == "__main__":
    sys.exit(main())
