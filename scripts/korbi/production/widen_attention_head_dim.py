#!/usr/bin/env python3
"""Widen the softmax-attention head dim of an fsdp_dtensor checkpoint (128 -> 256 or any multiple), function-preserving.

v4_2ph's 6 attention layers run 16 q heads x 128 (GQA 2, gated, qk RMSNorm, NoPE, XSA); Qwen3.5 uses 256.
This rewrites one checkpoint so that a model built with --kv-channels 256 computes the same function
(to O(eps^2)) and continues with the same optimizer state; the added channels carry small noise so they
get gradients and grow.

Per attention layer (fused linear_qkv rows per query group: [q heads | gate heads | k | v] x head dim):
  linear_qkv   every q / gate / k / v head gets d_new - d_old new rows: random unit directions with row
               norm eps x (median row norm of that head type). Muown classical row states: g = the new row
               norm, v_norm = median of the head type's old rows (same rotation rate), m_u / m_g / v_g = 0.
  q/k_layernorm  RMSNorm over the head dim: with ~zero new channels the normalized old channels grow by
               sqrt(r) (q.k by r = d_new / d_old) while the softmax scale drops by sqrt(r): old gammas x
               c = r^-1/4 (2^-1/4 for 128 -> 256) keep q.k / sqrt(d) exact; new gammas = mean(old) x c.
               Adam: exp_avg / c, exp_avg_sq / c^2 (the gradient's scale); new entries exp_avg 0,
               exp_avg_sq = mean(old).
  linear_proj  every head gets d_new - d_old new input columns, N(0, (eps * rms of the row)^2); g = the
               new row norm (signed as before); the spectral Muon scale grows sqrt(4096 / 2048), so v_norm
               x sqrt(2) and m_u x 1/sqrt(2) keep the rows' rotation rate; new m_u columns 0.
  value / gate rows of the new channels: tiny too, so the attention output (and XSA's projection
  coefficient) change only at O(eps^2).

The checkpoint is not copied tensor by tensor: the new tree hard-links every .distcp file of the source
(written once, never modified), adds one file with the rewritten tensors (and the args with kv_channels
256) and a merged .metadata whose entries for those keys point to it.

    # test the transform against a reference attention forward (CPU, any torch)
    python widen_attention_head_dim.py selftest
    # inspect / apply (inside the training container on MN5; login node is fine: ~1 GB read)
    python widen_attention_head_dim.py apply --src <ckpt>/iter_0009789 --dst <new ckpt>/iter_0009789 \\
        --layers 0,17,33,50,66,82 [--eps 1e-3] [--seed 1234]
"""

from __future__ import annotations

import argparse
import copy
import io
import math
import os
import pickle
import sys
import time

import torch

MODEL = "model.module.decoder.layers.{l}.self_attention.{n}"
OPT = "optimizer.0.state.module.module.module.decoder.layers.{l}.self_attention.{n}.{s}"
MUOWN_STATES = ("g", "m_u", "m_g", "v_g", "v_norm")


# ---------------------------------------------------------------------------------------------
# the transform (pure torch, CPU)
# ---------------------------------------------------------------------------------------------


def _qkv_view(t, ng, blocks, d):
    """[ng * blocks * d, X] -> [ng, blocks, d, X]."""
    return t.reshape(ng, blocks, d, t.shape[-1])


def widen_layer(t, *, heads, groups, d_old, d_new, eps, gen):
    """t: dict with keys qkv, proj, q_norm, k_norm, qkv.<state>, proj.<state>, q_norm.exp_avg(_sq),
    k_norm.exp_avg(_sq) (fp32). Returns the same keys at head dim d_new."""
    hpg = heads // groups
    blocks = 2 * hpg + 2  # q heads, gate heads, k, v
    kinds = ["q"] * hpg + ["gate"] * hpg + ["k", "v"]
    extra = d_new - d_old
    out = {}

    # ---- linear_qkv: new rows per head block
    w = _qkv_view(t["qkv"], groups, blocks, d_old)
    hidden = w.shape[-1]
    row_norm = w.norm(dim=-1)  # [ng, blocks, d_old]
    vn = _qkv_view(t["qkv.v_norm"], groups, blocks, d_old)[..., 0]
    new_w = torch.zeros(groups, blocks, d_new, hidden)
    new_w[:, :, :d_old] = w
    new_g = torch.zeros(groups, blocks, d_new, 1)
    new_g[:, :, :d_old] = _qkv_view(t["qkv.g"], groups, blocks, d_old)
    new_vn = torch.zeros(groups, blocks, d_new, 1)
    new_vn[:, :, :d_old] = vn[..., None]
    for kind in set(kinds):
        idx = [b for b, k in enumerate(kinds) if k == kind]
        scale = eps * row_norm[:, idx].median()
        vn_med = vn[:, idx].median()
        dirs = torch.randn(groups, len(idx), extra, hidden, generator=gen)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        new_w[:, idx, d_old:] = scale * dirs
        new_g[:, idx, d_old:] = scale
        new_vn[:, idx, d_old:] = vn_med
    out["qkv"] = new_w.reshape(groups * blocks * d_new, hidden)
    out["qkv.g"] = new_g.reshape(-1, 1)
    out["qkv.v_norm"] = new_vn.reshape(-1, 1)
    for s in ("m_u", "m_g", "v_g"):
        old = _qkv_view(t[f"qkv.{s}"], groups, blocks, d_old)
        new = torch.zeros(groups, blocks, d_new, old.shape[-1])
        new[:, :, :d_old] = old
        out[f"qkv.{s}"] = new.reshape(groups * blocks * d_new, old.shape[-1])

    # ---- q / k RMSNorm gammas (shared over heads, per head-dim channel)
    # RMS over d_new channels (the new ones ~0) grows the normalized old ones by sqrt(d_new / d_old),
    # q.k by d_new / d_old, while the softmax scale drops by sqrt(d_old / d_new): gammas x c with
    # c^2 = sqrt(d_old / d_new) (2^-1/4 for a doubling); the gammas' gradient scales by 1 / c
    c = (d_old / d_new) ** 0.25
    for n in ("q_norm", "k_norm"):
        gam = t[n]
        new = torch.empty(d_new)
        new[:d_old] = gam * c
        new[d_old:] = gam.mean() * c
        out[n] = new
        ea, eas = t[f"{n}.exp_avg"], t[f"{n}.exp_avg_sq"]
        new_ea = torch.zeros(d_new)
        new_ea[:d_old] = ea / c
        new_eas = torch.empty(d_new)
        new_eas[:d_old] = eas / c**2
        new_eas[d_old:] = (eas / c**2).mean()
        out[f"{n}.exp_avg"], out[f"{n}.exp_avg_sq"] = new_ea, new_eas

    # ---- linear_proj: new input columns per head
    p = t["proj"]
    rows = p.shape[0]
    pv = p.reshape(rows, heads, d_old)
    rms = p.pow(2).mean(dim=1, keepdim=True).sqrt()  # [rows, 1]
    new_p = torch.zeros(rows, heads, d_new)
    new_p[:, :, :d_old] = pv
    new_p[:, :, d_old:] = eps * rms[:, :, None] * torch.randn(rows, heads, extra, generator=gen)
    new_p = new_p.reshape(rows, heads * d_new)
    out["proj"] = new_p
    g_old = t["proj.g"]
    out["proj.g"] = torch.copysign(new_p.norm(dim=1, keepdim=True), g_old)
    spectral = math.sqrt(max(rows, heads * d_new)) / math.sqrt(max(rows, heads * d_old))
    out["proj.v_norm"] = t["proj.v_norm"] * spectral
    mu = torch.zeros(rows, heads, d_new)
    mu[:, :, :d_old] = t["proj.m_u"].reshape(rows, heads, d_old) / spectral
    out["proj.m_u"] = mu.reshape(rows, heads * d_new)
    out["proj.m_g"], out["proj.v_g"] = t["proj.m_g"].clone(), t["proj.v_g"].clone()
    return out


def reference_attention(x, w_qkv, g_q, g_k, w_proj, *, heads, groups, d, norm_eps=1e-6, xsa=True):
    """Megatron's gated GQA self attention (NoPE, qk RMSNorm, XSA, causal) on x [T, H], fp64."""
    hpg = heads // groups
    T = x.shape[0]
    mixed = (x @ w_qkv.T).reshape(T, groups, (2 * hpg + 2) * d)
    q, gate, k, v = torch.split(mixed, [hpg * d, hpg * d, d, d], dim=-1)
    q = q.reshape(T, heads, d)
    gate = gate.reshape(T, heads, d)
    rms = lambda a, g: a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + norm_eps) * g  # noqa: E731
    q, k = rms(q, g_q), rms(k, g_k)
    kk = k.repeat_interleave(hpg, dim=1)
    vv = v.repeat_interleave(hpg, dim=1)
    s = torch.einsum("thd,shd->hts", q, kk) / math.sqrt(d)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    o = torch.einsum("hts,shd->thd", s.softmax(-1), vv)
    if xsa:
        c = (o * vv).sum(-1, keepdim=True) / (vv.pow(2).sum(-1, keepdim=True) + 1e-6)
        o = o - c * vv
    o = o * torch.sigmoid(gate)
    return o.reshape(T, heads * d) @ w_proj.T


def selftest():
    torch.manual_seed(0)
    heads, groups, d_old, d_new, hidden, T = 16, 2, 128, 256, 512, 64
    hpg = heads // groups
    rows = groups * (2 * hpg + 2) * d_old
    t = {"qkv": torch.randn(rows, hidden) * 0.02 * 3, "proj": torch.randn(hidden, heads * d_old) * 0.02,
         "q_norm": 1 + 0.3 * torch.randn(d_old), "k_norm": 1 + 0.3 * torch.randn(d_old)}
    for n, r, c in (("qkv", rows, hidden), ("proj", hidden, heads * d_old)):
        t[f"{n}.g"] = t[n].norm(dim=1, keepdim=True)
        t[f"{n}.v_norm"] = t[f"{n}.g"] * (1 + torch.rand(r, 1))
        t[f"{n}.m_u"], t[f"{n}.m_g"], t[f"{n}.v_g"] = torch.randn(r, c), torch.randn(r, 1), torch.rand(r, 1)
    for n in ("q_norm", "k_norm"):
        t[f"{n}.exp_avg"], t[f"{n}.exp_avg_sq"] = torch.randn(d_old), torch.rand(d_old)
    x = torch.randn(T, hidden, dtype=torch.float64)
    ref = reference_attention(x, t["qkv"].double(), t["q_norm"].double(), t["k_norm"].double(),
                              t["proj"].double(), heads=heads, groups=groups, d=d_old)
    for dn in (512, d_new):  # 4x first, then the 2x case the layout checks below use
        for eps in (0.0, 1e-3, 1e-2):
            gen = torch.Generator().manual_seed(1234)
            n = widen_layer(t, heads=heads, groups=groups, d_old=d_old, d_new=dn, eps=max(eps, 1e-30), gen=gen)
            y = reference_attention(x, n["qkv"].double(), n["q_norm"].double(), n["k_norm"].double(),
                                    n["proj"].double(), heads=heads, groups=groups, d=dn)
            rel = ((y - ref).norm() / ref.norm()).item()
            print(f"{d_old} -> {dn}, eps {eps:g}: relative output change {rel:.2e}")
            assert rel < max(2e-6, 30 * eps**2), rel
    # layout: the old rows / columns are where Megatron reads them; row-state invariants
    hb = d_new
    blk = n["qkv"].reshape(groups, 2 * hpg + 2, d_new, hidden)
    assert torch.equal(blk[:, :, :d_old].reshape(rows, hidden), t["qkv"])
    assert torch.allclose(n["qkv.g"][:, 0], n["qkv"].norm(dim=1), rtol=1e-5)
    assert torch.allclose(n["proj.g"].abs()[:, 0], n["proj"].norm(dim=1), rtol=1e-5)
    assert torch.equal(n["proj"].reshape(hidden, heads, hb)[:, :, :d_old].reshape(hidden, -1), t["proj"])
    print("selftest passed")


# ---------------------------------------------------------------------------------------------
# checkpoint surgery (torch.distributed.checkpoint, single process)
# ---------------------------------------------------------------------------------------------

NAMES = {"qkv": "linear_qkv.weight", "proj": "linear_proj.weight", "q_norm": "q_layernorm.weight",
         "k_norm": "k_layernorm.weight"}


def layer_keys(layer):
    """our short name -> checkpoint key, for one attention layer."""
    m = {}
    for short, n in NAMES.items():
        m[short] = MODEL.format(l=layer, n=n)
    for short in ("qkv", "proj"):
        for s in MUOWN_STATES:
            m[f"{short}.{s}"] = OPT.format(l=layer, n=NAMES[short], s=s)
    for short in ("q_norm", "k_norm"):
        for s in ("exp_avg", "exp_avg_sq"):
            m[f"{short}.{s}"] = OPT.format(l=layer, n=NAMES[short], s=s)
    return m


def read_bytes_item(src, md, fqn):
    info = [v for k, v in md.storage_data.items() if k.fqn == fqn]
    assert len(info) == 1, (fqn, len(info))
    info = info[0]
    with open(os.path.join(src, info.relative_path), "rb") as f:
        f.seek(info.offset)
        raw = f.read(info.length)
    return torch.load(io.BytesIO(raw), weights_only=False)


def apply(a):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)  # noqa: E731
    reader = FileSystemReader(a.src)
    md = reader.read_metadata()
    smd = md.state_dict_metadata
    log("metadata", len(smd), "keys")
    layers = [int(x) for x in a.layers.split(",")]
    keymap = {l: layer_keys(l) for l in layers}
    want = {k: torch.empty(smd[k].size, dtype=smd[k].properties.dtype)
            for m in keymap.values() for k in m.values()}
    for k, v in want.items():
        assert v.dtype == torch.float32, (k, v.dtype)
    dcp.load(want, storage_reader=reader, no_dist=True)
    log("read", len(want), "tensors")

    args = read_bytes_item(a.src, md, "args")
    assert args.kv_channels == a.d_old, args.kv_channels
    heads, groups = args.num_attention_heads, args.num_query_groups
    gen = torch.Generator().manual_seed(a.seed)
    new_sd, report = {}, []
    for l, m in keymap.items():
        t = {short: want[k] for short, k in m.items()}
        n = widen_layer(t, heads=heads, groups=groups, d_old=a.d_old, d_new=a.d_new, eps=a.eps, gen=gen)
        for short, k in m.items():
            new_sd[k] = n[short].contiguous()
        report.append((l, t["qkv.g"].median().item(), t["qkv.v_norm"].median().item(),
                       t["q_norm"].mean().item(), t["k_norm"].mean().item(), t["proj.g"].abs().median().item()))
    for r in report:
        log("layer %d: qkv g med %.4f v_norm med %.4f | q/k gamma mean %.3f / %.3f | proj g med %.4f" % r)
    args.kv_channels = a.d_new
    new_sd["args"] = args

    os.makedirs(a.dst, exist_ok=False)
    tmp = os.path.join(a.dst, ".widen_tmp")
    dcp.save(new_sd, storage_writer=FileSystemWriter(tmp), no_dist=True)
    small = FileSystemReader(tmp).read_metadata()
    log("wrote the widened tensors")

    # merged metadata: the replaced keys point to the new file
    replaced = set(new_sd)
    out = copy.copy(md)
    out.state_dict_metadata = dict(smd)
    for k in replaced:
        out.state_dict_metadata[k] = small.state_dict_metadata[k]
    files = {}
    storage = {k: v for k, v in md.storage_data.items() if k.fqn not in replaced}
    for k, v in small.storage_data.items():
        name = files.setdefault(v.relative_path, f"widen{a.d_new}_{len(files)}{v.relative_path}")
        storage[k] = copy.copy(v)
        storage[k].relative_path = name
    out.storage_data = storage
    for old, name in files.items():
        os.rename(os.path.join(tmp, old), os.path.join(a.dst, name))
    os.remove(os.path.join(tmp, ".metadata"))
    os.rmdir(tmp)
    n_link = 0
    for f in os.listdir(a.src):
        if f.endswith(".distcp"):
            os.link(os.path.join(a.src, f), os.path.join(a.dst, f))
            n_link += 1
    with open(os.path.join(a.dst, ".metadata"), "wb") as f:
        pickle.dump(out, f)
    log(f"done: {n_link} hard links, {len(files)} new file(s), {len(replaced)} keys replaced")

    # check: the merged metadata reads back the new shapes / values
    chk = {k: torch.empty(out.state_dict_metadata[k].size) for k in list(replaced - {"args"})[:4]}
    dcp.load(chk, storage_reader=FileSystemReader(a.dst), no_dist=True)
    for k, v in chk.items():
        assert torch.equal(v, new_sd[k]), k
    log("read-back check passed:", ", ".join(f"{k.split('.self_attention.')[-1]} {tuple(v.shape)}"
                                             for k, v in chk.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--layers", default="0,17,33,50,66,82")
    p.add_argument("--d-old", type=int, default=128)
    p.add_argument("--d-new", type=int, default=256)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    else:
        apply(a)


if __name__ == "__main__":
    sys.exit(main())
