"""Depth profile of a training run from two consecutive torch_dist checkpoints (no forward pass):
per decoder layer and matrix kind, the realized update / weight ratio |W_b - W_a| / |W_a| over
the interval (the effective lr / weight-norm ratio), the weight RMS, and the share of the total
gradient (from the optimizer moments at b: Adam E[g^2], Muown ||m_u||^2 -- the moments are of the
clipped gradient, so only shares are meaningful). Loads one layer at a time (fp32 master weights
from the optimizer state).

    python depth_profile.py <ckpt iter_a dir> <ckpt iter_b dir> <out.json>
"""

import json
import re
import sys
from collections import defaultdict

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

A, B_, OUT = sys.argv[1:4]
MD_A = FileSystemReader(A).read_metadata().state_dict_metadata
MD_B = FileSystemReader(B_).read_metadata().state_dict_metadata
KEY = re.compile(r"chained_\d+\.optimizer\.state\.(\w+?)\.((?:decoder|embedding|output_layer)\..*)$")


def index(md):
    out = defaultdict(dict)
    for k in md:
        m = KEY.match(k)
        if m:
            out[m.group(2)][m.group(1)] = k
    return out


IA, IB = index(MD_A), index(MD_B)


def load(path, md, keys):
    sd = {k: torch.empty(md[k].size, dtype=md[k].properties.dtype) for k in keys}
    if sd:
        dcp.load(sd, storage_reader=FileSystemReader(path), no_dist=True)
    return sd


def layer_of(name):
    m = re.search(r"decoder\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def kind_of(name):
    n = re.sub(r"decoder\.layers\.\d+\.", "", name)
    return n.replace(".weight", "")


groups = defaultdict(list)
for name in IB:
    if "fp32_param" in IB[name] and "fp32_param" in IA.get(name, {}):
        groups[layer_of(name)].append(name)
rows = []
for layer in sorted(groups):
    names = groups[layer]
    sa = load(A, MD_A, [IA[n]["fp32_param"] for n in names])
    want = [IB[n][s] for n in names for s in ("fp32_param", "m_u", "exp_avg_sq") if s in IB[n]]
    sb = load(B_, MD_B, want)
    for n in names:
        wa = sa[IA[n]["fp32_param"]].float()
        wb = sb[IB[n]["fp32_param"]].float()
        if "m_u" in IB[n]:
            g2 = float(sb[IB[n]["m_u"]].float().square().sum())
        elif "exp_avg_sq" in IB[n]:
            g2 = float(sb[IB[n]["exp_avg_sq"]].float().sum())
        else:
            g2 = float("nan")
        rows.append({"name": n, "layer": layer, "kind": kind_of(n), "numel": wa.numel(),
                     "w_rms": float(wa.square().mean().sqrt()),
                     "update_ratio": float((wb - wa).norm() / wa.norm().clamp_min(1e-30)),
                     "grad2": g2})
    del sa, sb
    print(f"layer {layer}: {len(names)} params", flush=True)
json.dump({"a": A, "b": B_, "rows": rows}, open(OUT, "w"))
print("wrote", OUT)
