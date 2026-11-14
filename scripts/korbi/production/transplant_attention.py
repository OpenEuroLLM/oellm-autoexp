#!/usr/bin/env python3
"""Transplant trained attention sublayers of a DONOR checkpoint into the SAME sublayer positions of a BASE checkpoint,
where the base has complex_kda, as crossfade ckda_branch attention branches (docs/recall_surgery_ambitious.md §14,
Phase 2 M / Mp: "train attention early, merge late"). Per sublayer L:

  donor  L = attention (linear_qkv / linear_proj / q,k_layernorm + MuOwn / Adam states, its fused input norm)
  base   L = complex_kda (kept: under ckda_branch_crossfade the layer is (alpha) ckda + (1 - alpha)(1 + gamma) branch,
         alpha ramping 1 -> 0, so the model is exactly the base at the start and the donor's attention at the end)
  new    self_attention.branch.* = the donor's attention, its input-norm gain folded into linear_qkv against the base
         layer's own input norm (add_ckda_branch.attention_branch_attn_copy), optimizer states copied (folded), gamma 0

The donor and base must have the same width / heads / KV groups (the small31 arms do). The new tree hard-links every
.distcp of the base (dcp_surgery.write_surgery) and holds a manifest transplant_attention.json.

    transplant_attention.py apply --base <b7>/ckpt/iter_N --donor <E>/ckpt/iter_M --dst <M>/ckpt/iter_N \\
        --layers 9,25,42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from add_ckda_branch import ATTN_COPY_SRC, LAYER_NORM, _key, attention_branch_attn_copy  # noqa: E402


def apply(a) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    from dcp_surgery import write_surgery

    t0 = time.time()
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists (a partial attempt? remove it after checking)")
    bmd = FileSystemReader(a.base).read_metadata()
    dmd = FileSystemReader(a.donor).read_metadata()
    bsmd, dsmd = bmd.state_dict_metadata, dmd.state_dict_metadata
    layers = [int(x) for x in a.layers.split(",")]
    donor, norms = {}, {}
    for layer in layers:
        if _key(layer, "layer.A_log") not in bsmd:
            raise SystemExit(f"base sublayer {layer} is not complex_kda")
        if _key(layer, "linear_qkv.weight") not in dsmd:
            raise SystemExit(f"donor sublayer {layer} is not attention")
        if _key(layer, "branch.linear_qkv.weight") in bsmd:
            raise SystemExit(f"base sublayer {layer} has a branch already")
        for rel in ATTN_COPY_SRC:
            k = _key(layer, rel)
            donor[k] = torch.empty(tuple(dsmd[k].size), dtype=dsmd[k].properties.dtype)
        k = LAYER_NORM.format(l=layer)
        norms[k] = torch.empty(tuple(bsmd[k].size), dtype=bsmd[k].properties.dtype)
    dcp.load(donor, storage_reader=FileSystemReader(a.donor), no_dist=True)
    dcp.load(norms, storage_reader=FileSystemReader(a.base), no_dist=True)
    replace, stats = {}, {}
    for layer in layers:
        att = {rel: donor[_key(layer, rel)].float() for rel in ATTN_COPY_SRC}
        new, st = attention_branch_attn_copy(att, norms[LAYER_NORM.format(l=layer)])
        stats[layer] = st
        for rel, t in new.items():
            k = _key(layer, rel)
            like = dsmd.get(_key(layer, rel.replace("branch.", "", 1)))
            dtype = like.properties.dtype if like is not None and rel.startswith("branch.") else torch.float32
            replace[k] = t.to(dtype).contiguous()
    print(f"{len(replace)} new tensors for sublayers {layers} ({time.time() - t0:.0f} s)", flush=True)
    write_surgery(a.base, a.dst, bmd, replace, tag="transplant", chunk_bytes=64 << 20)
    back = {k: torch.empty_like(v) for k, v in replace.items()}
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in back if not torch.equal(back[k], replace[k])]
    if bad:
        raise SystemExit(f"read-back mismatch: {bad[:5]}")
    with open(os.path.join(a.dst, "transplant_attention.json"), "w") as f:
        json.dump({"base": a.base, "donor": a.donor, "layers": layers, "fold_stats": stats,
                   "new_tensors": len(replace), "seconds": round(time.time() - t0)}, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s -> {a.dst}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--base", required=True)
    p.add_argument("--donor", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--layers", required=True)
    a = ap.parse_args(argv)
    return apply(a)


if __name__ == "__main__":
    sys.exit(main())
