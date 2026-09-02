#!/usr/bin/env python3
"""Read RMSNorm gains straight out of the torch_dist checkpoints.

Answers "are the norm gains drifting away from their init of 1.0, and does the
drift knee at the same iteration the loss turned?" without running any training.
Loads only the five gain tensors per checkpoint, so it is seconds per iter_*
even though the checkpoints are ~450 GB each.

Run it inside the training container (login node is fine, no GPU needed):

  apptainer exec /e/project1/e-sta-openeurollm/container/\
MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif \
    python3 scripts/scan_norm_gains.py \
      /e/scratch/e-sta-openeurollm/production_training/\
oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4/checkpoints \
      --csv docs/fp8-loss-turn/data/norm_gains.csv
"""

import argparse
import csv
import re
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

# The only 1-D non-bias params in the model. Megatron stacks the per-layer ones
# into a single [num_layers, hidden] tensor, so dim 0 is the layer index.
GAINS = [
    "decoder.layers.self_attention.linear_qkv.layer_norm_weight",  # input_layernorm
    "decoder.layers.mlp.linear_fc1.layer_norm_weight",  # pre_mlp_layernorm
    "decoder.layers.self_attention.q_layernorm.weight",
    "decoder.layers.self_attention.k_layernorm.weight",
    "decoder.final_layernorm.weight",  # 1-D, no layer dim
]

# "...linear_qkv.layer_norm_weight" -> "linear_qkv"; "...final_layernorm.weight"
# -> "final_layernorm". Same rule either way.
SHORT = {k: k.split(".")[-2] for k in GAINS}


def load_gains(ckpt):
    """Pull just the gain tensors out of one iter_* directory."""
    reader = dcp.FileSystemReader(ckpt)
    meta = reader.read_metadata().state_dict_metadata

    sd = {}
    for key in GAINS:
        if key not in meta:
            continue  # tolerate configs without qk-norm
        m = meta[key]
        sd[key] = torch.empty(m.size, dtype=m.properties.dtype)

    dcp.load(sd, storage_reader=reader)
    return {k: v.float() for k, v in sd.items()}


def rows_for(it, gains):
    """One row per (tensor, layer).

    Layer is -1 for the unstacked final norm.
    """
    for key, t in gains.items():
        t = t.unsqueeze(0) if t.ndim == 1 else t
        layers = range(t.shape[0]) if t.shape[0] > 1 else [-1]
        for i, layer in enumerate(layers):
            g = t[i]
            yield {
                "iter": it,
                "tensor": SHORT[key],
                "layer": layer,
                "mean": g.mean().item(),
                "rms": g.pow(2).mean().sqrt().item(),
                "max": g.max().item(),
                "min": g.min().item(),
            }


def _fmt(row):
    """Trim float noise at write time.

    These come from bf16 tensors upcast to float32, so 17-digit repr is
    pure expansion noise -- 6 significant figures is far more than the
    source has, and keeps the committed CSVs under the 500 KB pre-commit
    limit. Formatting happens here, not in the row dicts, so in-memory
    arithmetic is unaffected.
    """
    return {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_root", type=Path, help="dir holding the iter_* checkpoints")
    ap.add_argument("--csv", type=Path, help="write per-layer rows here")
    ap.add_argument("--first", type=int, default=0, help="skip iterations below this")
    args = ap.parse_args()

    ckpts = sorted(
        args.ckpt_root.glob("iter_*"), key=lambda p: int(re.search(r"\d+", p.name).group())
    )
    ckpts = [c for c in ckpts if int(re.search(r"\d+", c.name).group()) >= args.first]
    if not ckpts:
        raise SystemExit(f"no iter_* checkpoints under {args.ckpt_root}")

    all_rows = []
    names = None

    for ckpt in ckpts:
        it = int(re.search(r"\d+", ckpt.name).group())
        try:
            gains = load_gains(ckpt)
        except Exception as e:
            # Incomplete saves, and the odd early checkpoint whose metadata
            # pickle wants megatron.core (needs a GPU node to unpickle).
            print(f"{it:<8} skipped: {type(e).__name__}")
            continue
        rows = list(rows_for(it, gains))
        all_rows += rows

        # Summary line: mean gain per tensor, averaged over layers.
        by_tensor = {}
        for r in rows:
            by_tensor.setdefault(r["tensor"], []).append(r["mean"])
        by_tensor = {k: sum(v) / len(v) for k, v in by_tensor.items()}

        if names is None:
            names = list(by_tensor)
            print("iter    " + "  ".join(f"{n:>14}" for n in names))
        print(f"{it:<8}" + "  ".join(f"{by_tensor[n]:>14.4f}" for n in names))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(_fmt(r) for r in all_rows)
        print(f"\nwrote {len(all_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
