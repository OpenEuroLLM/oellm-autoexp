#!/usr/bin/env python3
"""Raise the KV groups of the softmax-attention layers of an fsdp_dtensor checkpoint (2 -> 8 / 16) by
duplicating K/V heads: function-preserving, the copies can then specialise in a continuation.

v4_2ph_hd256's 6 attention layers (0, 17, 33, 50, 66, 82) run 16 query heads x 256 with 2 KV groups,
output gate, qk RMSNorm (gains shared over heads, per head-dim channel), NoPE, XSA. Query head h reads
KV group h // (heads / groups) (flash / TE attention, Megatron's fused_xsa, HF / vLLM: repeat_interleave).

Megatron's fused linear_qkv (attention.py SelfAttention.get_query_key_value_tensors) stores, per query
group, d = kv_channels rows per head block:

    [q heads of the group | gate heads of the group | k | v]       (2 * heads/groups + 2 blocks)

so at groups G -> G' = r G the new group j holds q / gate heads j*h', ..., (j+1)*h' - 1 (h' = heads / G')
and k / v of old group j // r: the q and gate blocks are only re-ordered, the k / v blocks copied r times.
Nothing else depends on num_query_groups: q/k_layernorm are [d], linear_proj's input is heads x d in query
head order, linear_qkv.layer_norm_weight is [hidden], XSA projects each query head on its own group's
value (the same value, copied). The surgery rewrites per layer only linear_qkv.weight and its optimizer
state; every other tensor of the checkpoint is the source's file (hard links, byte-identical).

Optimizer state (muown_classical, W = diag(g) v / |v| row-wise; checkpoint keys
optimizer.0.state...linear_qkv.weight.{g, v_norm, m_u, m_g, v_g}, all [rows, 1] except m_u [rows, hidden];
muon_qkv_headwise orthogonalizes every d-row head block on its own, so block shapes -- and the
Newton-Schulz scale -- do not change):
  q / gate rows   re-ordered with their weights, values unchanged.
  k / v copies    g, v_norm copied (parameters of the row; same rotation rate). The moments are split
                  (--moments split, default) into what the copies would have accumulated over the same
                  history: dL/dk_old = sum of the gradients of its r copies (selftest), the copies are
                  exchangeable at insertion, so each copy's first moments m_u / m_g are the original's / r
                  (exact in expectation); its gradient is nearly uncorrelated with the other copies' (each
                  copy is read by other query heads: |grad copy| ~ |grad| / sqrt(r), measured in the
                  selftest), so Adam's second moment v_g is the original's / r. Newton-Schulz is scale-free
                  per head block, so m_u / r only keeps the momentum's weight against the new, smaller
                  gradients. --moments copy keeps all states verbatim (momentum r x too strong for
                  ~1/(1-beta) steps, Adam's g steps sqrt(r) x too small for ~1/(1-beta2) steps).

    python duplicate_kv_groups.py selftest           # CPU: layout + fp64 reference attention, states
    python duplicate_kv_groups.py megatron-test      # Megatron HybridModel ('*' layer), 2 vs 8 / 16 groups
    python duplicate_kv_groups.py apply --src <ckpt>/iter_0023400 --dst <new>/iter_0023400 --groups 8
    python duplicate_kv_groups.py verify --dst <new>/iter_0023400   # re-derive, compare the manifest hashes
    python duplicate_kv_groups.py gpucheck --src <ckpt>/iter_N --dst <new>/iter_N   # real weights, 1 GPU
    python duplicate_kv_groups.py compare-dumps <arm A token_dump> <arm B token_dump>  # eval-only check
    python duplicate_kv_groups.py export-test        # HF export split + HF model at 2 vs 4 KV heads (GPU)

apply writes <dst>/kv_dup_surgery.json (source + .metadata SHA-256, arguments, code, old / new shapes,
SHA-256 of every rewritten tensor). Production: scripts/korbi/production/kv_dup_surgery.sh through the
1-node pool job config/experiments/korbi/production/kv_dup_surgery_mn5.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

MODEL = "model.module.decoder.layers.{l}.self_attention.{n}"
OPT = "optimizer.0.state.module.module.module.decoder.layers.{l}.self_attention.{n}.{s}"
MUOWN_STATES = ("g", "v_norm", "m_u", "m_g", "v_g")
ATTN_LAYERS = (0, 17, 33, 50, 66, 82)
MANIFEST = "kv_dup_surgery.json"


# ---------------------------------------------------------------------------------------------
# the transform (pure torch)
# ---------------------------------------------------------------------------------------------


def block_map(heads: int, g_old: int, g_new: int, gate: bool = True):
    """Source block (old layout) of every block of the new layout, and its kind (q / gate / k / v).

    Blocks are kv_channels rows; per group the layout is [q heads | gate heads (if gate) | k | v]."""
    if g_new % g_old or heads % g_new:
        raise ValueError(f"groups {g_old} -> {g_new}: needs g_new = r * g_old and heads ({heads}) % g_new == 0")
    hpo, hpn, r = heads // g_old, heads // g_new, g_new // g_old
    nz = 1 if gate else 0
    bo = (1 + nz) * hpo + 2
    src, kinds = [], []
    for j in range(g_new):
        grp = j // r  # the old group of new group j (its query heads j*hpn.. are in it)
        local = [h - grp * hpo for h in range(j * hpn, (j + 1) * hpn)]
        src += [grp * bo + i for i in local]
        kinds += ["q"] * hpn
        if gate:
            src += [grp * bo + hpo + i for i in local]
            kinds += ["gate"] * hpn
        src += [grp * bo + (1 + nz) * hpo, grp * bo + (1 + nz) * hpo + 1]
        kinds += ["k", "v"]
    return src, kinds


def duplicate_layer(t: dict, *, heads: int, g_old: int, g_new: int, d: int, gate: bool = True,
                    moments: str = "split") -> dict:
    """t: {"qkv": weight [rows, hidden], "qkv.<state>": muown states} -> the same keys at g_new groups."""
    src, kinds = block_map(heads, g_old, g_new, gate)
    r = g_new // g_old
    idx = torch.tensor(src)
    kv = torch.tensor([k in ("k", "v") for k in kinds])
    out = {}
    for key, x in t.items():
        if not key.startswith("qkv"):
            raise KeyError(key)
        rows = x.shape[0]
        if rows % d or rows // d != len(block_map(heads, g_old, g_old, gate)[0]):
            raise ValueError(f"{key}: {rows} rows is not {g_old} groups of the layout")
        b = x.reshape(rows // d, d, *x.shape[1:])[idx].clone()
        state = key.split(".", 1)[1] if "." in key else None
        if moments == "split" and state in ("m_u", "m_g", "v_g"):
            b[kv] = b[kv] / r
        elif moments not in ("split", "copy"):
            raise ValueError(moments)
        out[key] = b.reshape(len(src) * d, *x.shape[1:])
    return out


def reference_attention(x, w_qkv, g_q, g_k, w_proj, *, heads, groups, d, norm_eps=1e-6, xsa=True):
    """Megatron's gated GQA self attention (NoPE, qk RMSNorm, XSA, causal) on x [T, H] (fp64)."""
    hpg = heads // groups
    T = x.shape[0]
    mixed = (x @ w_qkv.T).reshape(T, groups, (2 * hpg + 2) * d)
    q, gate, k, v = torch.split(mixed, [hpg * d, hpg * d, d, d], dim=-1)
    q, gate = q.reshape(T, heads, d), gate.reshape(T, heads, d)
    rms = lambda a, g: a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + norm_eps) * g  # noqa: E731
    q, k = rms(q, g_q), rms(k, g_k)
    kk, vv = k.repeat_interleave(hpg, dim=1), v.repeat_interleave(hpg, dim=1)
    s = torch.einsum("thd,shd->hts", q, kk) / math.sqrt(d)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    o = torch.einsum("hts,shd->thd", s.softmax(-1), vv)
    if xsa:
        c = (o * vv).sum(-1, keepdim=True) / (vv.pow(2).sum(-1, keepdim=True) + 1e-6)
        o = o - c * vv
    o = o * torch.sigmoid(gate)
    return o.reshape(T, heads * d) @ w_proj.T


def _fake_layer(heads, groups, d, hidden, gen):
    rows = groups * (2 * heads // groups + 2) * d
    w = torch.randn(rows, hidden, generator=gen) * 0.06
    t = {"qkv": w, "qkv.g": w.norm(dim=1, keepdim=True), "qkv.v_norm": 1 + torch.rand(rows, 1, generator=gen),
         "qkv.m_u": torch.randn(rows, hidden, generator=gen), "qkv.m_g": torch.randn(rows, 1, generator=gen),
         "qkv.v_g": torch.rand(rows, 1, generator=gen)}
    extra = {"proj": torch.randn(hidden, heads * d, generator=gen) * 0.02,
             "q_norm": 1 + 0.3 * torch.randn(d, generator=gen), "k_norm": 1 + 0.3 * torch.randn(d, generator=gen)}
    return t, extra


def selftest() -> int:
    gen = torch.Generator().manual_seed(0)
    heads, d, hidden, T = 16, 64, 256, 48
    t, e = _fake_layer(heads, 2, d, hidden, gen)
    x = torch.randn(T, hidden, dtype=torch.float64, generator=gen)
    ref = reference_attention(x, t["qkv"].double(), e["q_norm"].double(), e["k_norm"].double(),
                              e["proj"].double(), heads=heads, groups=2, d=d)
    hpo = heads // 2
    for g_new in (2, 4, 8, 16):
        r = g_new // 2
        n = duplicate_layer(t, heads=heads, g_old=2, g_new=g_new, d=d)
        y = reference_attention(x, n["qkv"].double(), e["q_norm"].double(), e["k_norm"].double(),
                                e["proj"].double(), heads=heads, groups=g_new, d=d)
        diff = ((y - ref).abs().max() / ref.abs().max()).item()
        print(f"2 -> {g_new:2d} groups: rows {t['qkv'].shape[0]} -> {n['qkv'].shape[0]}, "
              f"max |out - ref| / max |ref| = {diff:.1e}")
        assert diff < 1e-12, diff
        # layout: every query head's q / gate rows are the old ones, every k / v copy its group's
        hpn = heads // g_new
        old = t["qkv"].reshape(2, 2 * hpo + 2, d, hidden)
        new = n["qkv"].reshape(g_new, 2 * hpn + 2, d, hidden)
        for h in range(heads):
            go, gn = h // hpo, h // hpn
            assert torch.equal(new[gn, h % hpn], old[go, h % hpo])  # q
            assert torch.equal(new[gn, hpn + h % hpn], old[go, hpo + h % hpo])  # gate
            assert torch.equal(new[gn, 2 * hpn], old[go, 2 * hpo])  # k
            assert torch.equal(new[gn, 2 * hpn + 1], old[go, 2 * hpo + 1])  # v
        # states: q / gate rows moved verbatim; k / v: g, v_norm copied, moments split
        so = {s: t[f"qkv.{s}"].reshape(2, 2 * hpo + 2, d, -1) for s in MUOWN_STATES}
        sn = {s: n[f"qkv.{s}"].reshape(g_new, 2 * hpn + 2, d, -1) for s in MUOWN_STATES}
        for s in MUOWN_STATES:
            f = r if s in ("m_u", "m_g", "v_g") else 1
            for j in range(g_new):
                grp = j // r
                assert torch.equal(sn[s][j, :hpn], so[s][grp, (j % r) * hpn:(j % r + 1) * hpn])
                torch.testing.assert_close(sn[s][j, 2 * hpn:], so[s][grp, 2 * hpo:] / f, rtol=0, atol=0)
        # the split moments sum back to the original's (first moments) over a k row's r copies
        mk = sn["m_u"][:, 2 * hpn].reshape(2, r, d, hidden).sum(1)
        torch.testing.assert_close(mk, so["m_u"][:, 2 * hpo], rtol=1e-6, atol=1e-6)
        c = duplicate_layer(t, heads=heads, g_old=2, g_new=g_new, d=d, moments="copy")
        assert torch.equal(c["qkv.m_u"].reshape(g_new, 2 * hpn + 2, d, -1)[:, 2 * hpn],
                           so["m_u"][:, 2 * hpo].repeat_interleave(r, 0))
    # the gradient of the duplicated model's k copies sums to the original k's gradient (at insertion)
    n = duplicate_layer(t, heads=heads, g_old=2, g_new=8, d=d)
    grads = []
    for w0, g in ((t["qkv"], 2), (n["qkv"], 8)):
        w = w0.double().clone().requires_grad_(True)
        out = reference_attention(x, w, e["q_norm"].double(), e["k_norm"].double(), e["proj"].double(),
                                  heads=heads, groups=g, d=d)
        (out * torch.randn(out.shape, dtype=out.dtype, generator=torch.Generator().manual_seed(3))).sum().backward()
        grads.append(w.grad)
    gk_old = grads[0].reshape(2, 2 * hpo + 2, d, hidden)[:, 2 * hpo:]
    gk_new = grads[1].reshape(8, 2 * 2 + 2, d, hidden)[:, 4:].reshape(2, 4, 2, d, hidden)
    gsum = gk_new.sum(1)
    print("sum of the copies' k/v gradients vs the original's: max diff "
          f"{(gsum - gk_old).abs().max().item():.1e}; copy share of |grad|: "
          + " ".join(f"{(gk_new[0, i, 0].norm() / gk_old[0, 0].norm()).item():.2f}" for i in range(4)))
    torch.testing.assert_close(gsum, gk_old, rtol=1e-9, atol=1e-12)
    # no gate (the plain GQA layout) maps the same way
    src, kinds = block_map(4, 1, 2, gate=False)
    assert src == [0, 1, 4, 5, 2, 3, 4, 5] and kinds == ["q", "q", "k", "v"] * 2, (src, kinds)
    print("selftest passed")
    return 0


# ---------------------------------------------------------------------------------------------
# Megatron's own attention: a one-layer HybridModel ('*'), GQA 2 vs the duplicated model
# ---------------------------------------------------------------------------------------------


def _init_megatron_single():
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 2000))
        dist.init_process_group("gloo" if os.environ.get("KVDUP_GLOO", "1") == "1" else "nccl",
                                rank=0, world_size=1)
    if not ps.model_parallel_is_initialized():
        ps.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)


def _megatron_model(*, groups, heads, d, hidden, dtype, backend, fused_qk=False, vocab=512):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import AttnBackend

    for v in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):  # set per backend by the model
        os.environ.pop(v, None)
    cfg = TransformerConfig(
        num_layers=1, hidden_size=hidden, ffn_hidden_size=4 * hidden, num_attention_heads=heads,
        num_query_groups=groups, kv_channels=d, attention_output_gate=True, exclusive_self_attention=True,
        qk_layernorm=True, normalization="RMSNorm", add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, hidden_dropout=0.0, attention_dropout=0.0,
        attention_backend=getattr(AttnBackend, backend), is_hybrid_model=True,
        bf16=dtype == torch.bfloat16, params_dtype=dtype, fused_qk_norm_rope=fused_qk,
    )
    return HybridModel(config=cfg, hybrid_stack_spec=hybrid_stack_spec, vocab_size=vocab,
                       max_sequence_length=8192, hybrid_layer_pattern="*", position_embedding_type="none",
                       share_embeddings_and_output_weights=False).cuda()


def _attn_output(model, ids):
    """The attention layer's linear_proj output (before the residual) and the logits."""
    seen = {}
    attn = model.decoder.layers[0].self_attention
    h = attn.linear_proj.register_forward_hook(lambda m, a, o: seen.__setitem__("o", o[0].detach().float()))
    with torch.no_grad():
        logits = model(ids, position_ids=None, attention_mask=None).float()
    h.remove()
    return seen["o"], logits


def compare_megatron(qkv_old, extra, *, heads, d, hidden, g_new, dtype, backend, fused_qk=False, seq=512,
                     batch=2, label="", corrupt=False):
    """Load one layer into Megatron models with 2 and g_new groups (the transform applied to linear_qkv)
    and compare the attention output. extra: q_norm, k_norm, proj, ln (linear_qkv.layer_norm_weight)."""
    g_old = (qkv_old.shape[0] // d - 2 * heads) // 2  # blocks = g (2 heads / g + 2) = 2 heads + 2 g
    new = duplicate_layer({"qkv": qkv_old}, heads=heads, g_old=g_old, g_new=g_new, d=d)["qkv"]
    if corrupt:  # negative control: query head 0 / 1 swapped in new group 0 (a wrong layout must show)
        blocks = new.view(-1, d, new.shape[-1]).clone()
        blocks[[0, 1]] = blocks[[1, 0]]
        new = blocks.view_as(new)
    outs = []
    base = None
    for g, w in ((g_old, qkv_old), (g_new, new)):
        torch.manual_seed(7)
        m = _megatron_model(groups=g, heads=heads, d=d, hidden=hidden, dtype=dtype, backend=backend,
                            fused_qk=fused_qk)
        sd = m.state_dict()
        if base is None:  # the same embedding / head / norms in both models
            base = {k: v.clone() for k, v in sd.items() if "_extra_state" not in k}
        a = "decoder.layers.0.self_attention."
        load = dict(base)
        load[a + "linear_qkv.weight"] = w
        load[a + "linear_proj.weight"] = extra["proj"]
        load[a + "q_layernorm.weight"] = extra["q_norm"]
        load[a + "k_layernorm.weight"] = extra["k_norm"]
        if "ln" in extra and a + "linear_qkv.layer_norm_weight" in sd:
            load[a + "linear_qkv.layer_norm_weight"] = extra["ln"]
        load = {k: v.to(sd[k].dtype).to(sd[k].device) for k, v in load.items() if k in sd}
        missing, unexpected = m.load_state_dict(load, strict=False)
        missing = [k for k in missing if "_extra_state" not in k]
        assert not missing and not unexpected, (missing, unexpected)
        ids = torch.randint(0, 512, (batch, seq), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        outs.append(_attn_output(m, ids))
        del m
        torch.cuda.empty_cache()
    (o2, l2), (o8, l8) = outs
    rel = ((o8 - o2).norm() / o2.norm()).item()
    mx = ((o8 - o2).abs().max() / o2.abs().max()).item()
    lrel = ((l8 - l2).norm() / l2.norm()).item()
    print(f"{label}megatron {backend} {str(dtype).replace('torch.', '')}{' fused_qk' if fused_qk else ''}"
          f"{' NEGATIVE CONTROL' if corrupt else ''}: "
          f"{g_old} -> {g_new} groups, attention out rel L2 {rel:.2e} (max {mx:.2e}), logits rel L2 {lrel:.2e}",
          flush=True)
    return rel


def megatron_test(a) -> int:
    """Random layer, Megatron's attention (TE linear_qkv with the fused input norm, core attention, fused
    XSA, output gate): 2 groups vs 8 and 16 groups after the transform."""
    _init_megatron_single()
    gen = torch.Generator().manual_seed(0)
    heads, d, hidden = 16, a.head_dim, a.hidden
    t, e = _fake_layer(heads, 2, d, hidden, gen)
    e["ln"] = 1 + 0.2 * torch.randn(hidden, generator=gen)
    ok = True
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        for g_new in (8, 16):
            for backend in a.backends.split(","):
                rel = compare_megatron(t["qkv"], e, heads=heads, d=d, hidden=hidden, g_new=g_new, dtype=dtype,
                                       backend=backend)
                ok &= rel < tol
    if a.fused_qk:
        rel = compare_megatron(t["qkv"], e, heads=heads, d=d, hidden=hidden, g_new=8, dtype=torch.bfloat16,
                               backend=a.backends.split(",")[0], fused_qk=True)
        ok &= rel < 2e-2
    rel = compare_megatron(t["qkv"], e, heads=heads, d=d, hidden=hidden, g_new=8, dtype=torch.float32,
                           backend=a.backends.split(",")[0], corrupt=True)
    ok &= rel > 1e-2
    print("megatron-test", "passed" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# checkpoint surgery (torch.distributed.checkpoint, single process)
# ---------------------------------------------------------------------------------------------


def layer_keys(layer: int) -> dict:
    m = {"qkv": MODEL.format(l=layer, n="linear_qkv.weight")}
    for s in MUOWN_STATES:
        m[f"qkv.{s}"] = OPT.format(l=layer, n="linear_qkv.weight", s=s)
    return m


def read_bytes_item(src, md, fqn):
    import io

    (info,) = [v for k, v in md.storage_data.items() if k.fqn == fqn]
    with open(os.path.join(src, info.relative_path), "rb") as f:
        f.seek(info.offset)
        raw = f.read(info.length)
    return torch.load(io.BytesIO(raw), weights_only=False)


def derive(a, md=None):
    """The rewritten items {key: tensor | args} and a per-layer report (deterministic, no randomness)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    t0 = time.time()
    md = md if md is not None else FileSystemReader(a.src).read_metadata()
    smd = md.state_dict_metadata
    args = read_bytes_item(a.src, md, "args")
    heads, d = args.num_attention_heads, args.kv_channels
    g_old = args.num_query_groups if args.group_query_attention else heads
    gate = bool(args.attention_output_gate)
    if a.groups == g_old:
        raise SystemExit(f"the source already has {g_old} groups")
    layers = [int(x) for x in a.layers.split(",")]
    keymap = {l: layer_keys(l) for l in layers}
    want = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype)
            for m in keymap.values() for k in m.values()}
    dcp.load(want, storage_reader=FileSystemReader(a.src), no_dist=True)
    print(f"read {len(want)} tensors in {time.time() - t0:.0f} s (heads {heads}, groups {g_old}, d {d}, "
          f"gate {gate})", flush=True)
    replace, report = {}, {}
    for l, m in keymap.items():
        t = {short: want[k] for short, k in m.items()}
        n = duplicate_layer(t, heads=heads, g_old=g_old, g_new=a.groups, d=d, gate=gate, moments=a.moments)
        for short, k in m.items():
            replace[k] = n[short].to(smd[k].properties.dtype).contiguous()
        report[l] = {short: [list(t[short].shape), list(n[short].shape)] for short in m}
    args.num_query_groups = a.groups
    args.group_query_attention = True
    replace["args"] = args
    return replace, report, {"heads": heads, "kv_channels": d, "groups_old": g_old, "gate": gate}


def _tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _code_state() -> dict:
    import subprocess

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    out = {"script": os.path.abspath(__file__), "root": root}
    stamp = os.path.join(root, "SYNC_STAMP")
    if os.path.exists(stamp):
        out["sync_stamp"] = open(stamp).read()
    try:
        out["git_head"] = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                         timeout=30).stdout.strip() or None
    except Exception:  # no git on the node: the stamp is the record
        pass
    return out


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def apply(a) -> int:
    import platform

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dcp_surgery import write_surgery

    t0 = time.time()
    md = FileSystemReader(a.src).read_metadata()
    replace, report, shape = derive(a, md)
    for l, rep in report.items():
        print(f"layer {l}: linear_qkv {rep['qkv'][0]} -> {rep['qkv'][1]}", flush=True)
    # source-sized row chunks (dcp_surgery): each resuming rank reads its own rows only
    out = write_surgery(a.src, a.dst, md, replace, tag=f"kvdup{a.groups}", log=print)
    tensors = {k: v for k, v in replace.items() if isinstance(v, torch.Tensor)}
    back = {k: torch.empty(tuple(out.state_dict_metadata[k].size), dtype=v.dtype) for k, v in tensors.items()}
    back["args"] = None
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in tensors if not torch.equal(back[k], tensors[k])]
    if bad or back["args"].num_query_groups != a.groups:
        raise SystemExit(f"read-back mismatch: {bad[:5]} args groups {back['args'].num_query_groups}")
    untouched = sum(1 for k in out.state_dict_metadata if k not in replace)
    doc = {
        "tool": "duplicate_kv_groups.py apply",
        "args": {k: v for k, v in vars(a).items() if k != "cmd"},
        "transform": (f"per attention layer: fused linear_qkv [q heads | gate heads | k | v] per group, "
                      f"{shape['groups_old']} -> {a.groups} groups (block_map); q / gate rows re-ordered, k / v "
                      f"rows copied {a.groups // shape['groups_old']}x; muown states g, v_norm moved / copied, "
                      f"moments '{a.moments}' (split: m_u, m_g, v_g / r on the k / v copies); "
                      f"args.num_query_groups = {a.groups}; every other key unchanged (hard-linked shards)"),
        "model": shape,
        "source": os.path.abspath(a.src),
        "source_metadata_sha256": _file_sha256(os.path.join(a.src, ".metadata")),
        "code": _code_state(),
        "torch": torch.__version__,
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "layers": {str(l): rep for l, rep in report.items()},
        "untouched_keys": untouched,
        "new_tensors": {k: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                            "sha256": _tensor_sha256(t)} for k, t in sorted(tensors.items())},
    }
    with open(os.path.join(a.dst, MANIFEST), "w") as f:
        json.dump(doc, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s: {len(tensors)} tensors rewritten, {untouched} keys untouched; "
          f"manifest {os.path.join(a.dst, MANIFEST)}", flush=True)
    return 0


def verify(a) -> int:
    doc = json.load(open(os.path.join(a.dst, MANIFEST)))
    args = argparse.Namespace(**doc["args"])
    replace, _, _ = derive(args)
    tensors = {k: v for k, v in replace.items() if isinstance(v, torch.Tensor)}
    want = doc["new_tensors"]
    bad = [k for k, t in tensors.items() if want.get(k, {}).get("sha256") != _tensor_sha256(t)]
    bad += [k for k in want if k not in tensors]
    src_ok = _file_sha256(os.path.join(args.src, ".metadata")) == doc["source_metadata_sha256"]
    print(f"verify {a.dst}: {len(tensors)} tensors, {len(bad)} differ{': ' + str(bad[:5]) if bad else ''}; "
          f"source .metadata {'unchanged' if src_ok else 'CHANGED'}")
    return 1 if bad or not src_ok else 0


def gpucheck(a) -> int:
    """The real layers through Megatron's attention on one GPU: source checkpoint (its groups) vs the
    surgery checkpoint (read back from --dst, not re-derived), random token inputs."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    _init_megatron_single()
    mds = {p: FileSystemReader(p).read_metadata() for p in (a.src, a.dst)}
    args_src = read_bytes_item(a.src, mds[a.src], "args")
    args_dst = read_bytes_item(a.dst, mds[a.dst], "args")
    heads, d, hidden = args_src.num_attention_heads, args_src.kv_channels, args_src.hidden_size
    g_new = args_dst.num_query_groups
    worst = 0.0
    for l in [int(x) for x in a.layers.split(",")]:
        names = {"qkv": "linear_qkv.weight", "proj": "linear_proj.weight", "q_norm": "q_layernorm.weight",
                 "k_norm": "k_layernorm.weight", "ln": "linear_qkv.layer_norm_weight"}
        got = {}
        for p in (a.src, a.dst):
            smd = mds[p].state_dict_metadata
            sd = {MODEL.format(l=l, n=n): torch.empty(tuple(smd[MODEL.format(l=l, n=n)].size)) for n in names.values()}
            dcp.load(sd, storage_reader=FileSystemReader(p), no_dist=True)
            got[p] = {s: sd[MODEL.format(l=l, n=n)] for s, n in names.items()}
        for s in ("proj", "q_norm", "k_norm", "ln"):
            assert torch.equal(got[a.src][s], got[a.dst][s]), (l, s)
        derived = duplicate_layer({"qkv": got[a.src]["qkv"]}, heads=heads, g_old=args_src.num_query_groups,
                                  g_new=g_new, d=d)["qkv"]
        assert torch.equal(derived, got[a.dst]["qkv"]), f"layer {l}: dst linear_qkv != transform(src)"
        e = {s: got[a.src][s] for s in ("proj", "q_norm", "k_norm", "ln")}
        for dtype in (torch.float32, torch.bfloat16):
            rel = compare_megatron(got[a.src]["qkv"], e, heads=heads, d=d, hidden=hidden, g_new=g_new, dtype=dtype,
                                   backend=a.backend, fused_qk=a.fused_qk and dtype == torch.bfloat16,
                                   seq=a.seq, batch=1, label=f"layer {l}: ")
            if dtype == torch.bfloat16:
                worst = max(worst, rel)
    print(f"gpucheck: worst bf16 attention-output rel L2 {worst:.2e}")
    return 0 if worst < 2e-2 else 1


def export_test(a) -> int:
    """The HF export path at more KV groups (local GPU): (1) the exporter's split (export_q35.split_gated_qkv,
    used by export_ambitious) of the duplicated Megatron layout gives the same q / gate projections and k / v =
    the original heads repeated (HF / vLLM read num_key_value_heads from the export's config, which
    export_ambitious takes from the run's num_query_groups); (2) a tiny HF OellmAmbitious model with
    num_key_value_heads 2 vs 4 (repeated k / v projections) gives the same logits, and cached decoding at 4."""
    import tempfile
    from pathlib import Path

    from oellm_autoexp.hf_export import vllm_parity as vp
    from oellm_autoexp.hf_export.export_q35 import split_gated_qkv
    from transformers import AutoConfig, AutoModelForCausalLM

    heads, d, h = 16, 32, 64
    qkv = torch.randn(2 * (2 * 8 + 2) * d, h, generator=torch.Generator().manual_seed(0))
    qz2, k2, v2 = split_gated_qkv(qkv, heads, 2, d)
    qz8, k8, v8 = split_gated_qkv(duplicate_layer({"qkv": qkv}, heads=heads, g_old=2, g_new=8, d=d)["qkv"],
                                  heads, 8, d)
    assert torch.equal(qz2, qz8)
    assert torch.equal(k8, k2.view(2, d, h).repeat_interleave(4, 0).reshape(-1, h))
    assert torch.equal(v8, v2.view(2, d, h).repeat_interleave(4, 0).reshape(-1, h))
    print("export split: q / gate identical, k / v = the 2 heads repeated 4x")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        p2 = vp.build_tiny(out / "kv2", blocks=["attention", "moe", "attention", "moe"], residual="ldm",
                           sign_hash=False, quantile=True, pab=True)
        m2 = AutoModelForCausalLM.from_pretrained(p2, trust_remote_code=True, dtype=torch.float32).cuda().eval()
        cfg = json.loads((p2 / "config.json").read_text())
        hd = cfg["head_dim"]
        cfg["num_key_value_heads"] = 4
        (out / "kv4").mkdir()
        vp._copy_code(out / "kv4")
        (out / "kv4" / "config.json").write_text(json.dumps(cfg))
        m4 = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(out / "kv4", trust_remote_code=True),
                                              trust_remote_code=True, dtype=torch.float32).cuda().eval()
        sd = {}
        for k, v in m2.state_dict().items():
            if k.endswith(("k_proj.weight", "v_proj.weight")):
                v = v.view(2, hd, -1).repeat_interleave(2, 0).reshape(-1, v.shape[-1])
            sd[k] = v
        m4.load_state_dict(sd, strict=True)
        ids = torch.randint(0, 1024, (2, 64), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        with torch.no_grad():
            la, lb = m2(ids).logits, m4(ids).logits
            o = m4(ids[:, :32], use_cache=True)
            step = m4(ids[:, 32:33], past_key_values=o.past_key_values, use_cache=True).logits
    diff, cache = (la - lb).abs().max().item(), (step[:, 0] - lb[:, 32]).abs().max().item()
    print(f"HF tiny model, 2 vs 4 KV heads: max |dlogit| {diff:.1e} (logit scale {la.abs().max().item():.0f}); "
          f"cached decoding at 4 vs full forward {cache:.1e}")
    return 0 if diff < 1e-3 and cache < 1e-3 else 1


def compare_dumps(a) -> int:
    """Two MEGATRON_EVAL_TOKEN_DUMP directories of the same eval layout (fixedvalid_eval_kvdup_v4_mn5 arms):
    per-token loss differences on identical tokens, per validation set (seq_length) and overall."""
    import glob

    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.a, "r*_*.pt")))
    if not files:
        raise SystemExit(f"no dumps in {a.a}")
    stats, seen = {}, set()
    for name in files:
        pb = os.path.join(a.b, name)
        if not os.path.exists(pb):
            raise SystemExit(f"{name}: missing in {a.b}")
        x, y = torch.load(os.path.join(a.a, name)), torch.load(pb)
        if not torch.equal(x["tokens"], y["tokens"]):
            raise SystemExit(f"{name}: different tokens -- not the same eval layout")
        key = hashlib.sha256(x["tokens"].numpy().tobytes()).hexdigest()
        if key in seen:  # ranks of one data-parallel group write the same micro-batch
            continue
        seen.add(key)
        m = x["loss_mask"].reshape(x["losses"].shape).bool()
        la, lb = x["losses"][m].double(), y["losses"][m].double()
        s = stats.setdefault(x.get("seq_length"), {"n": 0, "sa": 0.0, "sb": 0.0, "max": 0.0, "sq": 0.0})
        s["n"] += la.numel()
        s["sa"] += la.sum().item()
        s["sb"] += lb.sum().item()
        s["sq"] += (lb - la).pow(2).sum().item()
        s["max"] = max(s["max"], (lb - la).abs().max().item())
    tot = {"n": 0, "sa": 0.0, "sb": 0.0, "sq": 0.0, "max": 0.0}
    for seq, s in sorted(stats.items(), key=lambda kv: str(kv[0])):
        print(f"seq {seq}: {s['n']} tokens, mean loss {s['sa'] / s['n']:.6f} vs {s['sb'] / s['n']:.6f} "
              f"(diff {(s['sb'] - s['sa']) / s['n']:+.2e}), per-token rms diff {math.sqrt(s['sq'] / s['n']):.2e}, "
              f"max {s['max']:.2e}")
        for k in ("n", "sa", "sb", "sq"):
            tot[k] += s[k]
        tot["max"] = max(tot["max"], s["max"])
    print(f"all: {len(seen)} unique micro-batches, {tot['n']} tokens, mean loss diff "
          f"{(tot['sb'] - tot['sa']) / tot['n']:+.2e}, per-token rms {math.sqrt(tot['sq'] / tot['n']):.2e}, "
          f"max {tot['max']:.2e}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    mt = sub.add_parser("megatron-test")
    mt.add_argument("--backends", default="unfused", help="AttnBackend names, e.g. unfused,flash,auto")
    mt.add_argument("--hidden", type=int, default=512)
    mt.add_argument("--head-dim", type=int, default=128)
    mt.add_argument("--fused-qk", action="store_true", help="also the fused_qk_norm_rope path (bf16)")
    v = sub.add_parser("verify")
    v.add_argument("--dst", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--groups", type=int, default=8, help="new num_query_groups (a multiple of the source's)")
    p.add_argument("--layers", default=",".join(map(str, ATTN_LAYERS)))
    p.add_argument("--moments", choices=["split", "copy"], default="split")
    g = sub.add_parser("gpucheck")
    g.add_argument("--src", required=True)
    g.add_argument("--dst", required=True)
    g.add_argument("--layers", default=",".join(map(str, ATTN_LAYERS)))
    g.add_argument("--backend", default="auto")
    g.add_argument("--seq", type=int, default=4096)
    g.add_argument("--fused-qk", action="store_true")
    sub.add_parser("export-test", help="HF export / modeling at more KV heads (local GPU)")
    c = sub.add_parser("compare-dumps", help="per-token eval losses of two arms (MEGATRON_EVAL_TOKEN_DUMP dirs)")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return {"megatron-test": megatron_test, "apply": apply, "verify": verify, "gpucheck": gpucheck,
            "compare-dumps": compare_dumps, "export-test": lambda _: export_test(a)}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
