#!/usr/bin/env python3
"""Insert six first-class V shared-KV attention layers into the 35900 ambitious checkpoint.

98-layer source -> 104-layer destination: V after the old CKDA layers 9/25/41/58/74/90.
``--init random`` (default) is the v4a surgery: the V layers start with o_proj = 0 (exactly
function-preserving), random q / gate, unit q-norm and input norm, and zero optimizer moments.
``--init copy`` (v4b) copies each V layer's attention from its source attention layer instead:
q / gate rows, output projection and q-norm gain (with their MuOwn / Adam state), with the
source's input-norm gain folded into the ``linear_qkv`` columns, ``W' = W diag(g_attn / g_layer)``
(``g_layer`` is the V layer's own norm gain, ones). The branch is then a working attention of the
source layer's kind the instant its gate opens, and MuOwn has the row directions it needs: a
zero output projection leaves ``u = W / g`` with ``g = 0``, i.e. no direction, no magnitude
gradient, and the tensor cannot move at all. Both inits write the two learnable gates
(``v_gate.out_gate`` / ``v_gate.res_gate``, 0 = exact identity, ``v_block_learnable_gate``).

Existing 1.1 TB chunks are hard-linked; only metadata FQNs shift (dcp_surgery rename).

Usage:
  python insert_v_layers.py --src <ckpt>/iter_0035900 --dst <new>/ckpt/iter_0035900 [--dry-run]
  python insert_v_layers.py --init copy --src <ckpt>/iter_0035900 --dst <new>/ckpt/iter_0035900
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp  # noqa: F401  (metadata types)
from torch.distributed.checkpoint import FileSystemReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcp_surgery import write_surgery  # noqa: E402

ORIG_PATTERN = "*ENLELELELELELELE*ELELELELELELELE*ELELELELELELELEN*ELELELELELELELE*ELELELELELELELE*ELELELELELELELE"
INSERT_AFTER = [9, 25, 41, 58, 74, 90]
# the source attention layer of each V layer, by its 0-based index in the 104-layer pattern
# (kv_share_source_layers of the v4a / v4b configs)
SOURCES = [0, 18, 35, 53, 70, 87]
HEADS, D, GROUPS, HIDDEN = 16, 256, 2, 2048
QKV_ROWS = HEADS * D * 2  # two query groups worth of [q | gate]
PROJ_COLS = HEADS * D
MUOWN = ("g", "m_g", "v_g", "v_norm", "m_u")
ADAM = ("exp_avg", "exp_avg_sq")
MODEL = "model.module.decoder.layers.{l}.self_attention.{n}"
OPT = "optimizer.0.state.module.module.module.decoder.layers.{l}.self_attention.{n}.{s}"


def new_index(old: int) -> int:
    return old + sum(1 for p in INSERT_AFTER if p < old)


def old_index(new: int) -> int:
    """Inverse of :func:`new_index`: a 104-layer index -> the 98-layer source index.

    The source attention layers of the V layers are 0 / 18 / 35 / 53 / 70 / 87 in the 104-layer
    pattern (``kv_share_source_layers``) and 0 / 17 / 33 / 50 / 66 / 82 in the source checkpoint.
    """
    return new - sum(1 for p in INSERT_AFTER if p < new)


def build_pattern() -> tuple[str, list[int]]:
    out, v_layers = [], []
    for old, ch in enumerate(ORIG_PATTERN):
        out.append(ch)
        if old in INSERT_AFTER:
            if ch != "L":
                raise SystemExit(f"insert after {old}: pattern char {ch!r} is not L")
            v_layers.append(len(out))
            out.append("V")
    if len(out) != 104:
        raise SystemExit(f"pattern length {len(out)} != 104")
    return "".join(out), v_layers


def _state(rows: int, cols: int | None, kind: str) -> dict[str, torch.Tensor]:
    if kind == "adam":
        return {"exp_avg": torch.zeros(rows), "exp_avg_sq": torch.zeros(rows)}
    return {
        "g": torch.zeros(rows, 1),
        "m_g": torch.zeros(rows, 1),
        "v_g": torch.zeros(rows, 1),
        "v_norm": torch.ones(rows, 1),
        "m_u": torch.zeros(rows, cols),
    }


def _rows_state(w: torch.Tensor, v_norm: torch.Tensor) -> dict[str, torch.Tensor]:
    """MuOwn state of a copied weight: row norms as ``g``, the source's retained ``v_norm``, zero
    momenta. ``W = g * v / ||v||``, so this makes the copied rows' directions exactly ``W / ||W||``."""
    rows, cols = w.shape
    return {
        "g": w.norm(dim=1, keepdim=True).float(),
        "m_g": torch.zeros(rows, 1),
        "v_g": torch.zeros(rows, 1),
        "v_norm": v_norm.reshape(rows, 1).float(),
        "m_u": torch.zeros(rows, cols),
    }


def source_keys(sources: list[int]) -> list[str]:
    """The source attention tensors ``--init copy`` reads, per source layer.

    The source attention layer's input norm is TE-fused into its ``linear_qkv``
    (``linear_qkv.layer_norm_weight``), not a separate ``input_layernorm``: that fused gain is the one
    the fold needs. The V layer's own input norm is a separate ``TENorm`` (ones), so
    ``W' = W diag(g_attn / 1)`` reproduces the source's normalized input exactly.
    """
    keys: list[str] = []
    for s in sources:
        keys.append(MODEL.format(l=s, n="linear_qkv.weight"))
        keys += [OPT.format(l=s, n="linear_qkv.weight", s=k) for k in MUOWN]
        keys.append(MODEL.format(l=s, n="linear_proj.weight"))
        keys += [OPT.format(l=s, n="linear_proj.weight", s=k) for k in MUOWN]
        keys.append(MODEL.format(l=s, n="q_layernorm.weight"))
        keys.append(OPT.format(l=s, n="q_layernorm.weight", s="exp_avg_sq"))
        keys.append(MODEL.format(l=s, n="linear_qkv.layer_norm_weight"))
    return keys


def v_tensors_copy(layer: int, src: int, att: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """New tensors of V layer ``layer``, copied from its source attention layer ``src``.

    ``att`` holds that source's tensors (:func:`source_keys`). q / gate rows are the source's
    per-KV-group ``[q | gate | k | v]`` blocks minus k / v (the V layer reads the source's keys /
    values at run time), with the source's input-norm gain folded into the columns
    ``W' = W diag(g_attn / g_layer)``. The source's norm is TE-fused into its ``linear_qkv``
    (``linear_qkv.layer_norm_weight``); the V layer's own input norm is a separate ``TENorm`` at ones,
    so ``g_layer = 1`` and the fold is exactly ``diag(g_attn)`` -- the branch reads the same normalized
    input the source did. The output projection and the q-norm gain are copied whole. MuOwn state
    follows the fold: ``g`` = the new row norms, ``v_norm`` the source's rows, momenta zero (a fresh
    direction is a fresh momentum); the q-norm keeps the source's ``exp_avg_sq``.
    """
    qkv, proj = att["linear_qkv.weight"], att["linear_proj.weight"]
    hpg = HEADS // GROUPS
    per = (2 * hpg + 2) * D  # the source's rows per KV group: [q | gate | k | v]
    if qkv.shape != (GROUPS * per, HIDDEN) or proj.shape != (HIDDEN, PROJ_COLS):
        raise SystemExit(
            f"--init copy: source attention {src} linear_qkv {tuple(qkv.shape)} / linear_proj "
            f"{tuple(proj.shape)} is not {HEADS} gated heads x {D} in {GROUPS} groups"
        )
    g_attn = att["linear_qkv.layer_norm_weight"].float()  # the source's TE-fused input-norm gain
    g_layer = torch.ones(HIDDEN)  # the V layer's own input norm, written below
    r = g_attn / g_layer
    # the V layer's linear_qkv is per query group [q | gate] (KVShareSelfAttention splits the groups
    # first), so the copied rows interleave per group, not as all-q then all-gate
    rows: list[torch.Tensor] = []
    vns: list[torch.Tensor] = []
    for g in range(GROUPS):
        rows.append(qkv[g * per:g * per + hpg * D].float() * r[None, :])
        rows.append(qkv[g * per + hpg * D:g * per + 2 * hpg * D].float() * r[None, :])
        vns.append(att["linear_qkv.weight.v_norm"][g * per:g * per + hpg * D])
        vns.append(att["linear_qkv.weight.v_norm"][g * per + hpg * D:g * per + 2 * hpg * D])
    qkv_new = torch.cat(rows, 0)
    v_norm_new = torch.cat(vns, 0)
    if qkv_new.shape != (QKV_ROWS, HIDDEN):
        raise SystemExit(f"--init copy: built linear_qkv {tuple(qkv_new.shape)} != {(QKV_ROWS, HIDDEN)}")
    proj_new = proj.float().clone()
    mp = f"model.module.decoder.layers.{layer}."
    op = f"optimizer.0.state.module.module.module.decoder.layers.{layer}."
    out: dict[str, torch.Tensor] = {
        mp + "self_attention.linear_qkv.weight": qkv_new,
        mp + "self_attention.linear_proj.weight": proj_new,
        mp + "self_attention.q_layernorm.weight": att["q_layernorm.weight"].clone(),
        mp + "input_layernorm.weight": torch.ones(HIDDEN),
    }
    for rel, w, vn in (("self_attention.linear_qkv.weight", qkv_new, v_norm_new),
                       ("self_attention.linear_proj.weight", proj_new,
                        att["linear_proj.weight.v_norm"])):
        for k, t in _rows_state(w, vn).items():
            out[op + rel + "." + k] = t
    out[op + "self_attention.q_layernorm.weight.exp_avg"] = torch.zeros(D)
    out[op + "self_attention.q_layernorm.weight.exp_avg_sq"] = \
        att["q_layernorm.weight.exp_avg_sq"].clone()
    for rel, n in (("self_attention.q_layernorm.weight", D), ("input_layernorm.weight", HIDDEN)):
        for k, t in _state(n, None, "adam").items():
            out.setdefault(op + rel + "." + k, t)
    out.update(v_gate_tensors(layer))
    return out


def v_gate_tensors(layer: int) -> dict[str, torch.Tensor]:
    """The inserted V layer's two learnable gates, 0 at insertion (``v_block_learnable_gate``).

    ``out_gate`` is g_v on the attention write, ``res_gate`` is rho on the ``h_res`` stream mixing.
    Both are one-element parameters on the scalar (Adam) optimizer, so they carry exp_avg /
    exp_avg_sq. A config without the flag ignores them (they are extra checkpoint keys, as the
    ``input_layernorm`` of the pre-2026-11-17 spec was), so the copy-init arm sets the flag.
    """
    mp = f"model.module.decoder.layers.{layer}.v_gate."
    op = f"optimizer.0.state.module.module.module.decoder.layers.{layer}.v_gate."
    out: dict[str, torch.Tensor] = {}
    for name in ("out_gate", "res_gate"):
        out[mp + name] = torch.zeros(1)
        for k, t in _state(1, None, "adam").items():
            out[op + name + "." + k] = t
    return out


def v_tensors(layer: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + layer)
    qkv = torch.randn(QKV_ROWS, HIDDEN, generator=g) * 0.02
    proj = torch.zeros(HIDDEN, PROJ_COLS)
    qn = torch.ones(D)
    inp = torch.ones(HIDDEN)
    mp = f"model.module.decoder.layers.{layer}."
    op = f"optimizer.0.state.module.module.module.decoder.layers.{layer}."
    out = {
        mp + "self_attention.linear_qkv.weight": qkv,
        mp + "self_attention.linear_proj.weight": proj,
        mp + "self_attention.q_layernorm.weight": qn,
        mp + "input_layernorm.weight": inp,
    }
    for rel, rows, cols in (("self_attention.linear_qkv.weight", QKV_ROWS, HIDDEN),
                            ("self_attention.linear_proj.weight", HIDDEN, PROJ_COLS)):
        for k, v in _state(rows, cols, "muown").items():
            out[op + rel + "." + k] = v
    for rel, n in (("self_attention.q_layernorm.weight", D), ("input_layernorm.weight", HIDDEN)):
        for k, v in _state(n, None, "adam").items():
            out[op + rel + "." + k] = v
    out.update(v_gate_tensors(layer))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init", choices=("random", "copy"), default="random",
                    help="random (v4a): o_proj 0, random q / gate; copy (v4b): the source attention "
                         "layer's q / gate / o / q-norm, input-norm gain folded into linear_qkv")
    ap.add_argument("--sources", default=",".join(map(str, SOURCES)),
                    help="per V layer, the 0-based source attention layer (--init copy)")
    a = ap.parse_args()

    pattern, v_layers = build_pattern()
    print("pattern:", pattern)
    print("V layers:", v_layers)
    print("old->new:", {i: new_index(i) for i in range(98)})
    sources = [int(x) for x in a.sources.split(",") if x]
    if a.init == "copy" and len(sources) != len(v_layers):
        raise SystemExit(f"--sources {sources}: one source per V layer {v_layers}")
    md = FileSystemReader(a.src).read_metadata()
    rename = {}
    for k in md.state_dict_metadata:
        m = re.search(r"decoder\.layers\.(\d+)", k)
        if not m:
            continue
        old = int(m.group(1))
        new = new_index(old)
        if new != old:
            rename[k] = k.replace(f"decoder.layers.{old}", f"decoder.layers.{new}", 1)
    att = {}
    if a.init == "copy":
        olds = [old_index(s) for s in sources]
        keys = source_keys(olds)
        missing = [k for k in keys if k not in md.state_dict_metadata]
        if missing:
            raise SystemExit(f"--init copy: {len(missing)} source tensors not in {a.src}: {missing[:4]}")
        want = {k: torch.empty(tuple(md.state_dict_metadata[k].size)) for k in keys}
        dcp.load(want, storage_reader=FileSystemReader(a.src), no_dist=True)
        print(f"read {len(want)} source attention tensors of {sources} (source layers {olds})")
        by_old = {o: {} for o in olds}
        for k, t in want.items():
            src_layer = int(re.search(r"decoder\.layers\.(\d+)\.", k).group(1))
            rel = k.split(f"decoder.layers.{src_layer}.", 1)[1]
            if rel.startswith("self_attention."):
                rel = rel[len("self_attention."):]
            by_old[src_layer][rel] = t
        att = {s: by_old[o] for s, o in zip(sources, olds)}
    replace = {}
    for v, s in zip(v_layers, sources if a.init == "copy" else [None] * len(v_layers)):
        replace.update(v_tensors(v, a.seed) if a.init == "random" else v_tensors_copy(v, s, att[s]))
    print(f"rename: {len(rename)} FQNs; new V tensors: {len(replace)} (init {a.init})")
    if a.dry_run:
        print("dry run: no checkpoint written")
        return
    write_surgery(a.src, a.dst, md, replace, tag="v_insert", rename=rename, thread_count=16)
    (a.dst.parent / "latest_checkpointed_iteration.txt").write_text("35900\n")
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
