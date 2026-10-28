"""Muown row magnitudes across checkpoints: under muown_classical W = g v / |v| row by row, so the
optimizer state's g (one per output row) IS the row norm of W, and v_norm the norm of the
unnormalized direction (its growth anneals the rotation rate). Reads only these [rows, 1] tensors
(plus the routers' weights and expert biases) from torch_dist checkpoints -- no model weights, no
forward -- and writes per parameter: |g| mean / rms / p99 / max, v_norm mean / max, rows.

    python muown_g_stats.py <out.json> <tag>=<checkpoint iter dir> ...
"""

import json
import re
import sys

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

# per-parameter optimizer layout (the production runs: optimizer.<i>.state.<param>.<key>) and the
# chained distributed-optimizer one (screens: chained_<i>.optimizer.state.<key>.<param>, experts
# stacked [E, rows, ...]); g / v_norm are [.., rows, 1] or broadcast to the parameter's shape
STATE = re.compile(r"^optimizer\.\d+\.state\.(?:module\.)*(decoder\..+|embedding\..+|output_layer\..+)\.(g|v_norm)$")
CHAINED = re.compile(r"^chained_\d+\.optimizer\.state\.(g|v_norm)\.(decoder\..+|embedding\..+|output_layer\..+)$")
ROUTER = re.compile(r"^(?:model\.(?:module\.)*)?(decoder\.layers\.\d+\.mlp\.router\.(?:weight|expert_bias))$")
CHUNK = 4000


def stats(t: torch.Tensor) -> list:
    a = t.float().abs().flatten().numpy()
    return [float(a.mean()), float(np.sqrt((a * a).mean())), float(np.quantile(a, 0.99)),
            float(a.max()), int(a.size)]


def per_row(t: torch.Tensor) -> torch.Tensor:
    """[.., rows, 1] or the parameter-shaped broadcast -> [.., rows]."""
    return t[..., 0]


def expert_quantiles(rows: torch.Tensor) -> dict:
    """[E, rows]: the distribution over experts of each expert's max and mean |g|."""
    a = rows.float().abs()
    q = lambda x: [float(v) for v in np.quantile(x.numpy(), [0.5, 0.9, 0.99, 1.0])]  # noqa: E731
    return {"expert_max_q": q(a.max(dim=1).values), "expert_mean_q": q(a.mean(dim=1))}


def scan(path: str) -> dict:
    md = FileSystemReader(path).read_metadata().state_dict_metadata
    want = {}
    for k, v in md.items():
        m = STATE.match(k)
        if m:
            want[k] = (m.group(1), m.group(2))
            continue
        m = CHAINED.match(k)
        if m:
            want[k] = (m.group(2), m.group(1))
            continue
        m = ROUTER.match(k)
        if m:
            want[k] = (m.group(1), "router")
    out = {}
    keys = sorted(want)
    for i in range(0, len(keys), CHUNK):
        part = keys[i:i + CHUNK]
        sd = {k: torch.empty(md[k].size, dtype=md[k].properties.dtype) for k in part}
        dcp.load(sd, storage_reader=FileSystemReader(path), no_dist=True)
        for k in part:
            name, kind = want[k]
            t = sd[k]
            if kind == "router" and name.endswith(".weight"):  # per expert: the router row norm
                out.setdefault(name, {})["router_row_norm"] = stats(t.float().norm(dim=-1))
            elif kind == "router":
                out.setdefault(name, {})["value"] = stats(t)
            else:
                rows = per_row(t)
                out.setdefault(name, {})[kind] = stats(rows)
                if rows.dim() == 2 and ".experts.experts." in name:
                    out[name][kind + "_experts"] = expert_quantiles(rows)
        print(f"{path}: {min(i + CHUNK, len(keys))} / {len(keys)}", flush=True)
    return out


def main():
    res = {}
    for spec in sys.argv[2:]:
        tag, path = spec.split("=", 1)
        res[tag] = {"path": path, "params": scan(path)}
        json.dump(res, open(sys.argv[1], "w"))
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
