#!/usr/bin/env python3
"""Layer-swap compatibility test (docs/recall_surgery_ambitious.md §11): does a sublayer trained in an EARLIER
checkpoint of the same run still fit a later one?

Motivation: the "train new attention layers early, merge them into the late model" plan (user 2026-11-13) only
works if a layer from step d still reads / writes the residual streams of step b > d. Here a base checkpoint gets
sublayers of a donor checkpoint (same run, same architecture), or those sublayers' outputs zeroed (the ablation
reference), and the eval-only fixed-validation loss (fixedvalid_eval_layer_swap_mn5) says how much each costs:
swap ~ 0 -> layers transfer; swap ~ zero-ablation (or worse) -> a transplanted layer is no better than none.

    layer_swap.py build --base <ckpt>/iter_B --dst <out>/ckpt/iter_B --layers attn6 [--donor <ckpt>/iter_D]
        --layers  attn6 (the 6 attention sublayers) | attn1 (attention 17) | ckda1 (CKDA 35) |
                  mixall (every attention + CKDA sublayer) | a comma list of sublayer indices
        --donor   swap: every model tensor of those sublayers (mixer, its mHC residual parameters, input norm)
                  from the donor; without --donor: zero the sublayers' output projections (linear_proj / o_proj)

Only model tensors are written (the eval runs with no_load_optim); the optimizer states stay the base's, so an
output is for evaluation, not for resuming training. The output hard-links every base file (dcp_surgery.py) and
holds a manifest layer_swap.json; the tracker is the caller's (layer_swap_build.sh), written after the read-back.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

PREFIX = "model.module.decoder.layers."
ATTN1, CKDA1 = 17, 35


def sublayer_kinds(smd) -> dict[int, str]:
    """{sublayer index: 'attn' | 'ckda' | 'other'} from the checkpoint keys."""
    kinds: dict[int, str] = {}
    for k in smd:
        m = re.match(r"model\.module\.decoder\.layers\.(\d+)\.(.*)", k)
        if not m:
            continue
        i, rel = int(m.group(1)), m.group(2)
        if rel == "self_attention.linear_qkv.weight":
            kinds[i] = "attn"
        elif rel == "self_attention.layer.A_log":
            kinds[i] = "ckda"
        else:
            kinds.setdefault(i, "other")
    return kinds


def resolve_layers(spec: str, kinds: dict[int, str]) -> list[int]:
    attn = sorted(i for i, k in kinds.items() if k == "attn")
    ckda = sorted(i for i, k in kinds.items() if k == "ckda")
    named = {"attn6": attn, "attn1": [ATTN1], "ckda1": [CKDA1], "mixall": sorted(attn + ckda)}
    layers = named[spec] if spec in named else [int(x) for x in spec.split(",")]
    bad = [i for i in layers if kinds.get(i) not in ("attn", "ckda")]
    if bad:
        raise SystemExit(f"sublayers {bad} are not attention / CKDA sublayers")
    if spec == "attn6" and len(attn) != 6:
        raise SystemExit(f"expected 6 attention sublayers, found {attn}")
    return layers


def model_keys(smd, layers: list[int]) -> list[str]:
    pre = tuple(f"{PREFIX}{i}." for i in layers)
    return sorted(k for k in smd if k.startswith(pre) and hasattr(smd[k], "size"))


def out_proj(i: int, kind: str) -> str:
    return f"{PREFIX}{i}." + ("self_attention.linear_proj.weight" if kind == "attn" else "self_attention.layer.o_proj.weight")


def build(a) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dcp_surgery import write_surgery

    t0 = time.time()
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists (a partial attempt? remove it after checking)")
    md = FileSystemReader(a.base).read_metadata()
    smd = md.state_dict_metadata
    kinds = sublayer_kinds(smd)
    layers = resolve_layers(a.layers, kinds)
    if a.donor:
        keys = model_keys(smd, layers)
        dmd = FileSystemReader(a.donor).read_metadata().state_dict_metadata
        for k in keys:
            if k not in dmd or tuple(dmd[k].size) != tuple(smd[k].size) or dmd[k].properties.dtype != smd[k].properties.dtype:
                raise SystemExit(f"{k}: donor shape / dtype differs from the base (not the same architecture)")
        replace = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in keys}
        dcp.load(replace, storage_reader=FileSystemReader(a.donor), no_dist=True)
        base_vals = {k: torch.empty_like(v) for k, v in replace.items()}
        dcp.load(base_vals, storage_reader=FileSystemReader(a.base), no_dist=True)
        rel_change = {k: ((replace[k].float() - base_vals[k].float()).norm() / base_vals[k].float().norm().clamp_min(1e-30)).item()
                      for k in keys}
    else:
        keys = [out_proj(i, kinds[i]) for i in layers]
        replace = {k: torch.zeros(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in keys}
        rel_change = {k: 1.0 for k in keys}
    print(f"{len(layers)} sublayers {layers}: {len(replace)} tensors ({'donor ' + a.donor if a.donor else 'zeroed'}),"
          f" read in {time.time() - t0:.0f} s", flush=True)
    write_surgery(a.base, a.dst, md, replace, tag="layer_swap", chunk_bytes=64 << 20)
    back = {k: torch.empty_like(v) for k, v in replace.items()}
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in back if not torch.equal(back[k], replace[k])]
    if bad:
        raise SystemExit(f"read-back mismatch: {bad[:5]}")
    manifest = {"base": a.base, "donor": a.donor, "mode": "swap" if a.donor else "zero", "layers_spec": a.layers,
                "layers": layers, "kinds": {str(i): kinds[i] for i in layers}, "tensors": len(replace),
                "rel_change_median": sorted(rel_change.values())[len(rel_change) // 2],
                "rel_change": rel_change, "seconds": round(time.time() - t0)}
    with open(os.path.join(a.dst, "layer_swap.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s; median relative change {manifest['rel_change_median']:.3f}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--base", required=True, help="the base checkpoint's iter_* directory")
    p.add_argument("--dst", required=True, help="the output iter_* directory (must not exist)")
    p.add_argument("--layers", required=True)
    p.add_argument("--donor", help="the donor checkpoint's iter_* directory (else: zero the output projections)")
    a = ap.parse_args(argv)
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
