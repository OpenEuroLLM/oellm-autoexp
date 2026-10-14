#!/usr/bin/env python3
"""Per-output-neuron weight norms: are individual units dying inside a layer?

`scan_weight_stats.py` reports one number per layer, so a matrix can keep a
healthy overall rms while individual output neurons collapse to zero. That is
the gap this closes. For each weight [num_layers, out, in] it takes the rms
along the input dimension, giving one norm per output neuron, and reports how
far the weakest neurons have fallen relative to the layer's median.

One checkpoint per invocation; see scan_weight_stats.py's header for the
libcuda stub the pre-swap checkpoints need.

  apptainer exec <sif> python3 scripts/scan_dead_neurons.py <iter_dir> \\
      --run flagship --csv docs/64k-debug/data/dead_neurons.csv
"""

import argparse
import csv
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

TENSORS = [
    "decoder.layers.self_attention.linear_qkv.weight",
    "decoder.layers.self_attention.linear_proj.weight",
    "decoder.layers.mlp.linear_fc2.weight",
    "decoder.layers.mlp.linear_fc1.weight",  # 33.6 GB, last by default
]


def rows(ckpt, keys):
    reader = dcp.FileSystemReader(ckpt)
    meta = reader.read_metadata().state_dict_metadata
    for key in keys:
        if key not in meta:
            continue
        m = meta[key]
        sd = {key: torch.empty(m.size, dtype=m.properties.dtype)}
        dcp.load(sd, storage_reader=reader)
        t = sd.pop(key)
        short = key.replace("decoder.layers.", "").replace(".weight", "")
        for layer in range(t.shape[0]):
            # rms over the input dim -> one norm per output neuron
            n = t[layer].float().pow(2).mean(dim=-1).sqrt()
            med = n.median()
            yield {
                "tensor": short,
                "layer": layer,
                "n_neurons": n.numel(),
                "row_rms_med": med.item(),
                "row_rms_min": n.min().item(),
                "row_rms_p01": n.quantile(0.01).item(),
                # "Dead" is relative to the layer's own median, so it does not
                # depend on the absolute scale, which grows all run.
                "frac_below_10pct_med": (n < 0.10 * med).float().mean().item(),
                "frac_below_25pct_med": (n < 0.25 * med).float().mean().item(),
                "frac_below_50pct_med": (n < 0.50 * med).float().mean().item(),
            }
            del n
        del t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--rotate", type=int, default=0)
    args = ap.parse_args()

    keys = TENSORS
    if args.rotate:
        r = args.rotate % len(keys)
        keys = keys[r:] + keys[:r]

    it = int(re.search(r"\d+", args.ckpt.name).group())
    out = [{"iter": it, "run": args.run, **r} for r in rows(args.ckpt, keys)]
    if not out:
        raise SystemExit(f"no weight tensors read from {args.ckpt}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    new = not args.csv.exists()
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        if new:
            w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    # NB plain min over the ratio. An earlier version wrote
    # `r["frac_below_10pct_med"] and <ratio>`, which returns 0.0 whenever the
    # fraction is 0.0 -- so it printed "0.0000x" for healthy layers.
    worst = min(r["row_rms_min"] / r["row_rms_med"] for r in out)
    print(
        f"{args.run} {it}: {len(out)} layer-tensors, weakest neuron "
        f"{worst:.4f}x its layer median -> {args.csv}"
    )


if __name__ == "__main__":
    main()
