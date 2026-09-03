#!/usr/bin/env python3
"""Measure production delayed-scaling FP8 weight quantization error.

The existing ``scan_fp8_amax.py`` validates amax/scale bookkeeping, but a
correct scale can still give poor resolution to typical values when a tensor
contains outliers.  This probe reads the BF16 checkpoint weights and each
layer's saved Transformer Engine forward weight scale, performs the E4M3
quantize/dequantize operation used by the HYBRID recipe, and reports error by
layer.  It is read-only and performs no model forward or parameter update.

One checkpoint per invocation.  The stacked tensors are read one at a time and
individual layers are quantized on the GPU, keeping peak GPU memory small.
"""

import argparse
import csv
import io
import math
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from scan_fp8_amax import decode
from scan_weight_stats import TENSORS


FP8_MAX = 448.0  # E4M3 forward format used by Transformer Engine HYBRID


def saved_weight_scales(reader, meta):
    """Return {(module suffix, layer): saved forward weight scale}."""
    pat = re.compile(
        r"decoder\.layers\.(?P<module>self_attention\.linear_(?:qkv|proj)|"
        r"mlp\.linear_fc[12])\._extra_state/shard_(?P<layer>\d+)_\d+$"
    )
    keys = [(k, pat.match(k)) for k in meta]
    keys = [(k, m) for k, m in keys if m]
    state = {k: io.BytesIO() for k, _ in keys}
    dcp.load(state, storage_reader=reader)

    out = {}
    for key, match in keys:
        payload = decode(state[key])
        scale = payload.get("scale_fwd") if isinstance(payload, dict) else None
        if scale is None or scale.numel() < 2:
            continue
        out[(match["module"], int(match["layer"]))] = float(scale.float()[1])
    return out


def quant_stats(weight, scale):
    w = weight.cuda().float()
    scaled = w * scale
    clipped = (scaled.abs() > FP8_MAX).float().mean().item()
    # Transformer Engine uses a saturating FP8 cast.  PyTorch's plain E4M3FN
    # conversion may produce NaN for values above the finite maximum.
    q = scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    zero = (q.float() == 0).float().mean().item()
    dq = q.float() / scale
    err = dq - w
    signal = w.pow(2).sum()
    noise = err.pow(2).sum()
    rel_l2 = (noise / signal.clamp_min(1e-30)).sqrt().item()
    sqnr_db = 10.0 * math.log10(max(signal.item(), 1e-30) / max(noise.item(), 1e-30))
    cosine = torch.nn.functional.cosine_similarity(w.flatten(), dq.flatten(), dim=0).item()
    return {
        "scale_wgt": scale,
        "amax_scaled": scaled.abs().max().item(),
        "rel_l2": rel_l2,
        "sqnr_db": sqnr_db,
        "cosine": cosine,
        "zero_frac": zero,
        "clip_frac": clipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="one iter_* torch_dist checkpoint")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument(
        "--tensors", default="all", help="comma-separated suffixes to include, or 'all'"
    )
    args = ap.parse_args()

    reader = dcp.FileSystemReader(str(args.ckpt))
    meta = reader.read_metadata().state_dict_metadata
    scales = saved_weight_scales(reader, meta)
    keys = (
        TENSORS
        if args.tensors == "all"
        else [k for k in TENSORS if any(s in k for s in args.tensors.split(","))]
    )
    iteration = int(re.search(r"\d+", args.ckpt.name).group())
    rows = []

    for key in keys:
        if key not in meta:
            continue
        md = meta[key]
        state = {key: torch.empty(md.size, dtype=md.properties.dtype)}
        dcp.load(state, storage_reader=reader)
        stacked = state[key]
        module = key.replace("decoder.layers.", "").replace(".weight", "")
        for layer in range(stacked.shape[0]):
            scale = scales.get((module, layer))
            if scale is None:
                continue
            rows.append(
                {
                    "iter": iteration,
                    "tensor": module,
                    "layer": layer,
                    "numel": stacked[layer].numel(),
                    **quant_stats(stacked[layer], scale),
                }
            )
        del stacked, state
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit(f"no matching FP8 weights/scales in {args.ckpt}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.8g}" if isinstance(v, float) else v) for k, v in row.items()}
            )

    for module in sorted({r["tensor"] for r in rows}):
        group = [r for r in rows if r["tensor"] == module]
        sqnr = sorted(r["sqnr_db"] for r in group)
        rel = sorted(r["rel_l2"] for r in group)
        zero = sorted(r["zero_frac"] for r in group)
        clip = max(r["clip_frac"] for r in group)
        print(
            f"{iteration} {module}: median SQNR={sqnr[len(sqnr) // 2]:.2f} dB, "
            f"median relL2={rel[len(rel) // 2]:.5f}, "
            f"median zeros={zero[len(zero) // 2]:.5f}, max clip={clip:.3e}"
        )
    print(f"{len(rows)} layer tensors -> {args.csv}")


if __name__ == "__main__":
    main()
