#!/usr/bin/env python3
"""Turn complex_kda sublayers of an fsdp_dtensor checkpoint into plain (stack) softmax-attention sublayers with
RANDOM weights: the early 3:1 surgery (docs/recall_surgery_ambitious.md §13; user 2026-11-13: "replace many CKDA
with transformer layers ... hardly insert them from a random init, so that they are heavily trained").

The model then runs with the hybrid_layer_pattern's 'L' at those sublayers turned into '*' (7:1 -> 3:1 at
9/25/41/58/74/90). Per replaced sublayer L (reference = an existing attention sublayer R of the SAME checkpoint,
already at the target head dim, e.g. widened by widen_attention_head_dim.py; and the row statistics of R in the
--stats-src checkpoint, e.g. the unwidened one, whose rows are all trained):
  linear_qkv.weight   per KV group [q heads | gate heads | k | v] x d rows: random unit directions at R's median
                      row norm of that kind (gate rows x eps: sigmoid ~ 0.5); MuOwn classical states fresh
                      (g = row norm, v_norm = R's median of the kind, momenta 0)
  linear_proj.weight  random rows at R's median row norm, fresh MuOwn states
  q/k_layernorm       R's median gain (all d channels), Adam exp_avg 0, exp_avg_sq = R's mean
  linear_qkv.layer_norm_weight  = L's own complex_kda input-norm gain (the pre-norm of that position) and its
                      Adam moments
  the optimizer's per-parameter non-tensor items (param_to_group_meta.*, step): aliased to R's entries (same
                      group, lr / wd multipliers; no data written)
  mhc_attn_mixer.*    L's own (unchanged: the residual read / write of that position)
L's complex_kda tensors stay in the checkpoint unused (a strict load ignores extra keys); the first save of the
trained model drops them. The new tree hard-links every .distcp of the source (dcp_surgery.write_surgery).

    python replace_ckda_with_attention.py apply --src <hd256 ckpt>/iter_N --dst <out>/iter_N \\
        --stats-src <hd128 ckpt>/iter_N --layers 9,25,41,58,74,90 --refs 0,17,33,50,66,82 [--seed 1234]
    python replace_ckda_with_attention.py selftest     # synthetic 2-rank checkpoint, CPU
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch

M = "model.module.decoder.layers.{l}."
O = "optimizer.0.state.module.module.module.decoder.layers.{l}."
G = "optimizer.0.param_to_group_meta.module.module.module.decoder.layers.{l}."
MUOWN = ("g", "m_u", "m_g", "v_g", "v_norm")
ADAM = ("exp_avg", "exp_avg_sq")
SA = "self_attention."


def _noise_rows(n, cols, scale, gen):
    x = torch.randn(n, cols, generator=gen, dtype=torch.float64)
    return (x / x.norm(dim=1, keepdim=True) * float(scale)).float()


def _fresh_muown(w, v_norm_ref):
    n, c = w.shape
    return {"g": w.norm(dim=1, keepdim=True), "m_g": torch.zeros(n, 1), "v_g": torch.zeros(n, 1),
            "v_norm": torch.full((n, 1), float(v_norm_ref)), "m_u": torch.zeros(n, c)}


def ref_stats(t: dict, *, heads: int, groups: int, d: int) -> dict:
    """Median row norms / v_norm per row kind (q, gate, k, v of linear_qkv; linear_proj) and q/k norm gains of a
    reference attention layer with head dim ``d``. ``t``: names relative to self_attention."""
    qkv, vn = t["linear_qkv.weight"], t["linear_qkv.weight.v_norm"]
    hpg = heads // groups
    per = (2 * hpg + 2) * d
    if qkv.shape[0] != groups * per:
        raise ValueError(f"reference linear_qkv {tuple(qkv.shape)} is not {groups} groups x {2 * hpg + 2} x {d}")

    def kind(lo, hi):
        rows = torch.cat([qkv[g * per + lo:g * per + hi] for g in range(groups)]).float()
        vns = torch.cat([vn[g * per + lo:g * per + hi] for g in range(groups)]).float()
        return rows.norm(dim=1).median().item(), vns.median().item()

    s = {"q": kind(0, hpg * d), "gate": kind(hpg * d, 2 * hpg * d), "k": kind(2 * hpg * d, (2 * hpg + 1) * d),
         "v": kind((2 * hpg + 1) * d, (2 * hpg + 2) * d),
         "proj": (t["linear_proj.weight"].float().norm(dim=1).median().item(),
                  t["linear_proj.weight.v_norm"].float().median().item())}
    for n in ("q_layernorm", "k_layernorm"):
        s[n] = (t[f"{n}.weight"].float().median().item(), t[f"{n}.weight.exp_avg_sq"].float().mean().item())
    return s


def random_attention(stats: dict, *, hidden: int, heads: int, groups: int, d: int, eps: float,
                     gen: torch.Generator) -> dict:
    """Tensors of a random stack attention layer (names relative to self_attention.) at the reference stats."""
    hpg = heads // groups
    rows, states = [], []
    for _ in range(groups):
        for n_rows, kind, scale in ((hpg * d, "q", 1.0), (hpg * d, "q", eps), (d, "k", 1.0), (d, "v", 1.0)):
            norm, vref = stats[kind]
            w = _noise_rows(n_rows, hidden, norm * scale, gen)
            rows.append(w)
            states.append(_fresh_muown(w, vref))
    qkv = torch.cat(rows, 0)
    out = {"linear_qkv.weight": qkv}
    out.update({f"linear_qkv.weight.{s}": torch.cat([p[s] for p in states], 0) for s in MUOWN})
    proj = _noise_rows(hidden, heads * d, stats["proj"][0], gen)
    out["linear_proj.weight"] = proj
    out.update({f"linear_proj.weight.{s}": v for s, v in _fresh_muown(proj, stats["proj"][1]).items()})
    for n in ("q_layernorm", "k_layernorm"):
        gain, sq = stats[n]
        out[f"{n}.weight"] = torch.full((d,), gain)
        out[f"{n}.weight.exp_avg"] = torch.zeros(d)
        out[f"{n}.weight.exp_avg_sq"] = torch.full((d,), sq)
    return out


def _full_key(layer: int, rel: str) -> str:
    """Checkpoint key of a name relative to self_attention. (a parameter or <parameter>.<state>)."""
    for s in MUOWN + ADAM:
        if rel.endswith("." + s):
            return O.format(l=layer) + SA + rel
    return M.format(l=layer) + SA + rel


def derive(a, md, stats_md):
    """(replace {key: tensor}, alias {new key: reference key}, info) for the arguments."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    smd = md.state_dict_metadata
    layers = [int(x) for x in a.layers.split(",")]
    refs = [int(x) for x in a.refs.split(",")]
    if len(layers) != len(refs):
        raise SystemExit("--layers and --refs: one reference attention sublayer per replaced sublayer")
    gen = torch.Generator().manual_seed(a.seed)
    replace, alias, info = {}, {}, {}
    ssmd = stats_md.state_dict_metadata
    names = (["linear_qkv.weight", "linear_qkv.weight.v_norm", "linear_proj.weight", "linear_proj.weight.v_norm"]
             + [f"{n}.weight{x}" for n in ("q_layernorm", "k_layernorm") for x in ("", ".exp_avg_sq")])
    for layer, ref in zip(layers, refs):
        if M.format(l=layer) + SA + "layer.A_log" not in smd:
            raise SystemExit(f"sublayer {layer} is not a complex_kda sublayer of {a.src}")
        if M.format(l=ref) + SA + "linear_qkv.weight" not in smd:
            raise SystemExit(f"sublayer {ref} is not an attention sublayer of {a.src}")
    # ONE read per checkpoint (each dcp.load plans over the whole metadata: ~877k items, the login node's CPU limit)
    want = {_full_key(ref, n): torch.empty(tuple(ssmd[_full_key(ref, n)].size), dtype=ssmd[_full_key(ref, n)].properties.dtype)
            for ref in sorted(set(refs)) for n in names}
    dcp.load(want, storage_reader=FileSystemReader(a.stats_src), no_dist=True)
    ln_keys = {}
    for layer in layers:
        ln_keys[(layer, "")] = M.format(l=layer) + "input_layernorm.weight"
        for s in ADAM:
            ln_keys[(layer, s)] = O.format(l=layer) + f"input_layernorm.weight.{s}"
    ln_keys = {k: v for k, v in ln_keys.items() if v in smd}
    lnv = {v: torch.empty(tuple(smd[v].size), dtype=smd[v].properties.dtype) for v in ln_keys.values()}
    dcp.load(lnv, storage_reader=FileSystemReader(a.src), no_dist=True)
    print(f"read {len(want)} reference + {len(lnv)} norm tensors", flush=True)
    for layer, ref in zip(layers, refs):
        # every key of the reference attention's self_attention (model, states, group meta, step) -> layer
        for pre in (M, O, G):
            src_pre, dst_pre = pre.format(l=ref) + SA, pre.format(l=layer) + SA
            for k in smd:
                if k.startswith(src_pre):
                    new = dst_pre + k[len(src_pre):]
                    if new in smd:
                        raise SystemExit(f"{new} exists already in {a.src}")
                    alias[new] = k
        # target shapes (the reference at the target head dim)
        qkv_shape = tuple(smd[M.format(l=ref) + SA + "linear_qkv.weight"].size)
        proj_shape = tuple(smd[M.format(l=ref) + SA + "linear_proj.weight"].size)
        d = smd[M.format(l=ref) + SA + "q_layernorm.weight"].size[0]
        hidden, heads = qkv_shape[1], proj_shape[1] // d
        groups = (qkv_shape[0] // d - 2 * heads) // 2
        # the reference's trained row statistics (in --stats-src: e.g. before widening, where every row is trained)
        d_ref = ssmd[_full_key(ref, "q_layernorm.weight")].size[0]
        st = ref_stats({n: want[_full_key(ref, n)] for n in names}, heads=heads, groups=groups, d=d_ref)
        new = random_attention(st, hidden=hidden, heads=heads, groups=groups, d=d, eps=a.eps, gen=gen)
        # the position's own pre-norm gain (complex_kda input_layernorm) and its Adam moments
        g0 = lnv[ln_keys[(layer, "")]].float()
        new["linear_qkv.layer_norm_weight"] = g0
        for s in ADAM:
            k = ln_keys.get((layer, s))
            new[f"linear_qkv.layer_norm_weight.{s}"] = lnv[k].float() if k else torch.zeros_like(g0)
        for rel, t in new.items():
            key = _full_key(layer, rel)
            if key not in alias:
                raise SystemExit(f"{key}: not a key of the reference attention layout")
            ref_meta = smd[alias[key]]
            if tuple(t.shape) != tuple(ref_meta.size):
                raise SystemExit(f"{key}: shape {tuple(t.shape)} != reference {tuple(ref_meta.size)}")
            replace[key] = t.to(ref_meta.properties.dtype).contiguous()
        info[layer] = {"ref": ref, "d": d, "d_ref_stats": d_ref, "heads": heads, "groups": groups,
                       "stats": {k: list(v) for k, v in st.items()}}
        print(f"sublayer {layer} <- random attention (reference {ref}, d {d}, {heads} heads, {groups} KV groups)",
              flush=True)
    # every aliased TENSOR is rewritten; only non-tensor items (group meta, step) are shared with the reference
    left = [k for k in alias if k not in replace and hasattr(smd[alias[k]], "size")]
    if left:
        raise SystemExit(f"aliased tensors not rewritten: {left[:5]}")
    for k in replace:
        del alias[k]
    return replace, alias, info


def with_aliases(md, alias):
    """A copy of the metadata ``md`` where each key of ``alias`` reads its reference key's stored items."""
    from torch.distributed.checkpoint.metadata import MetadataIndex

    out = copy.copy(md)
    out.state_dict_metadata = dict(md.state_dict_metadata)
    out.storage_data = dict(md.storage_data)
    by_fqn = {}
    for idx, inf in md.storage_data.items():
        by_fqn.setdefault(idx.fqn, []).append((idx, inf))
    for new, ref in alias.items():
        out.state_dict_metadata[new] = md.state_dict_metadata[ref]
        for idx, inf in by_fqn[ref]:
            out.storage_data[MetadataIndex(fqn=new, offset=idx.offset, index=idx.index)] = inf
    return out


def apply(a) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dcp_surgery import write_surgery

    t0 = time.time()
    md = FileSystemReader(a.src).read_metadata()
    stats_md = md if a.stats_src == a.src else FileSystemReader(a.stats_src).read_metadata()
    replace, alias, info = derive(a, md, stats_md)
    if getattr(a, "pattern", None):
        # the saved args: the new layer pattern (export tools read the model layout from them)
        args = {"args": None}
        dcp.load(args, storage_reader=FileSystemReader(a.src), no_dist=True)
        args = args["args"]
        old = getattr(args, "hybrid_layer_pattern", None)
        if old is None or len(old) != len(a.pattern):
            raise SystemExit(f"--pattern: the checkpoint's hybrid_layer_pattern is {old!r}")
        for layer in (int(x) for x in a.layers.split(",")):
            if old[layer] != "L" or a.pattern[layer] != "*":
                raise SystemExit(f"--pattern: sublayer {layer} must go 'L' -> '*' ({old[layer]!r} -> {a.pattern[layer]!r})")
        if [i for i, (x, y) in enumerate(zip(old, a.pattern)) if x != y] != sorted(int(x) for x in a.layers.split(",")):
            raise SystemExit("--pattern differs from the checkpoint's beyond --layers")
        args.hybrid_layer_pattern = a.pattern
        replace["args"] = args
        info["pattern"] = {"old": old, "new": a.pattern}
    print(f"{len(replace)} items new, {len(alias)} non-tensor items aliased ({time.time() - t0:.0f} s)", flush=True)
    write_surgery(a.src, a.dst, with_aliases(md, alias), replace, tag="ckda2attn", chunk_bytes=64 << 20)
    back = {k: torch.empty_like(v) for k, v in replace.items() if isinstance(v, torch.Tensor)}
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in back if not torch.equal(back[k], replace[k])]
    if bad:
        raise SystemExit(f"read-back mismatch: {bad[:5]}")
    # the aliased non-tensor items read back as the reference's
    if alias:
        sample = dict(list(alias.items())[:50])
        objs = {k: None for k in list(sample) + list(sample.values())}  # the references are unchanged in dst
        dcp.load(objs, storage_reader=FileSystemReader(a.dst), no_dist=True)
        diff = [k for k, r in sample.items() if objs[k] != objs[r]]
        if diff:
            raise SystemExit(f"aliased items differ from their references: {diff[:5]}")
    manifest = {"src": a.src, "stats_src": a.stats_src, "layers": a.layers, "refs": a.refs, "seed": a.seed,
                "eps": a.eps, "new_tensors": len([v for v in replace.values() if isinstance(v, torch.Tensor)]), "aliased_items": len(alias), "info": info,
                "seconds": round(time.time() - t0)}
    with open(os.path.join(a.dst, "ckda2attn_surgery.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s -> {a.dst}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# selftest: a synthetic sharded checkpoint (2 gloo ranks), one attention + one complex_kda sublayer
# ---------------------------------------------------------------------------------------------


def _synthetic(d=8, heads=4, groups=2, hidden=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    rows = groups * (2 * heads // groups + 2) * d
    att = {"linear_qkv.weight": torch.randn(rows, hidden, generator=g), "linear_proj.weight": torch.randn(hidden, heads * d, generator=g),
           "q_layernorm.weight": torch.rand(d, generator=g) + 0.5, "k_layernorm.weight": torch.rand(d, generator=g) + 0.5,
           "linear_qkv.layer_norm_weight": torch.rand(hidden, generator=g) + 0.5}
    for n, t in att.items():
        sd[M.format(l=0) + SA + n] = t
        if t.dim() == 2:
            for s in MUOWN:
                sd[O.format(l=0) + SA + f"{n}.{s}"] = torch.rand(t.shape[0], t.shape[1] if s == "m_u" else 1, generator=g)
        else:
            for s in ADAM:
                sd[O.format(l=0) + SA + f"{n}.{s}"] = torch.rand(t.shape, generator=g)
        sd[O.format(l=0) + SA + f"{n}.step"] = 1000
        sd[G.format(l=0) + SA + f"{n}.lr_mult"] = 1.0
    for n in ("layer.A_log", "layer.q_proj.weight"):
        sd[M.format(l=1) + SA + n] = torch.randn(4, hidden, generator=g)
    sd[M.format(l=1) + "input_layernorm.weight"] = torch.rand(hidden, generator=g) + 0.5
    for s in ADAM:
        sd[O.format(l=1) + f"input_layernorm.weight.{s}"] = torch.rand(hidden, generator=g)
    return sd


def selftest() -> int:
    import tempfile

    import torch.distributed.checkpoint as dcp

    with tempfile.TemporaryDirectory() as root:
        src, dst = os.path.join(root, "src"), os.path.join(root, "dst")
        sd = _synthetic()
        dcp.save(sd, checkpoint_id=src, no_dist=True)
        a = argparse.Namespace(src=src, dst=dst, stats_src=src, layers="1", refs="0", seed=1234, eps=1e-3, pattern=None)
        apply(a)
        from torch.distributed.checkpoint import FileSystemReader

        out = FileSystemReader(dst).read_metadata().state_dict_metadata
        want = {}
        for k in sd:
            if k.startswith((M.format(l=0) + SA, O.format(l=0) + SA, G.format(l=0) + SA)):
                nk = k.replace("layers.0.", "layers.1.")
                assert nk in out, nk
                want[nk] = torch.empty_like(sd[k]) if isinstance(sd[k], torch.Tensor) else None
        for k in sd:  # everything of the source still there, unchanged
            assert k in out, k
        dcp.load(want, checkpoint_id=dst, no_dist=True)
        assert want[O.format(l=1) + SA + "linear_qkv.weight.step"] == 1000
        assert want[G.format(l=1) + SA + "linear_qkv.weight.lr_mult"] == 1.0
        qkv = want[M.format(l=1) + SA + "linear_qkv.weight"]
        assert qkv.shape == sd[M.format(l=0) + SA + "linear_qkv.weight"].shape
        assert not torch.allclose(qkv, sd[M.format(l=0) + SA + "linear_qkv.weight"])
        assert torch.equal(want[M.format(l=1) + SA + "linear_qkv.layer_norm_weight"], sd[M.format(l=1) + "input_layernorm.weight"])
        g_ = want[O.format(l=1) + SA + "linear_qkv.weight.g"]
        assert torch.allclose(g_.squeeze(1), qkv.norm(dim=1), rtol=1e-5)
        assert torch.count_nonzero(want[O.format(l=1) + SA + "linear_qkv.weight.m_u"]) == 0
        orig = {k: torch.empty_like(v) for k, v in sd.items() if isinstance(v, torch.Tensor)}
        dcp.load(orig, checkpoint_id=dst, no_dist=True)
        assert all(torch.equal(orig[k], sd[k]) for k in orig)
    print("selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("apply")
    p.add_argument("--src", required=True, help="checkpoint iter_* dir at the target head dim (e.g. widened)")
    p.add_argument("--dst", required=True)
    p.add_argument("--stats-src", help="where the reference rows' statistics are read (default --src)")
    p.add_argument("--layers", default="9,25,41,58,74,90")
    p.add_argument("--refs", default="0,17,33,50,66,82")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eps", type=float, default=1e-3, help="gate rows' norm relative to the q rows")
    p.add_argument("--pattern", help="the new hybrid_layer_pattern, written into the saved args (checked: exactly "
                   "--layers go 'L' -> '*')")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    a.stats_src = a.stats_src or a.src
    return apply(a)


if __name__ == "__main__":
    sys.exit(main())
