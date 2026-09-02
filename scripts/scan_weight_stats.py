#!/usr/bin/env python3
"""Per-layer max|W| and rms|W| for the four FP8 GEMM weights, read straight out
of a torch_dist checkpoint.

Why this exists: `fp8_amax.csv`'s `amax_wgt` comes from TE's delayed-scaling
`_extra_state`, so it only exists in FP8 runs -- the bf16 control (`cont4`,
fp8: null) records none, and cannot be compared against. This recomputes the
same quantity from the weight tensors themselves, which every run has, and adds
the RMS so outlier growth can be told apart from bulk growth.

One checkpoint per invocation, so several can run in parallel (the reader is
single-threaded at ~0.14 GB/s); concatenate the per-checkpoint files into
docs/fp8-loss-turn/data/weight_stats.csv afterwards. Weights are stacked [num_layers, out, in], so
dim 0 is the layer index; slices are cast one layer at a time to keep the fp32
copy small.

  apptainer exec <sif> python3 scripts/scan_weight_stats.py <iter_dir> \
      --run flagship --csv docs/fp8-loss-turn/data/weight_stats.csv

Checkpoints written before the 2026-08-28 stack swap carry a metadata pickle
that imports megatron.core -> transformer_engine -> libcuda.so.1, which a login
node does not have; read_metadata() dies with OSError before any tensor is
touched. Same fix as scan_fp8_amax.py -- hand it a stub so the dlopen resolves:

  echo 'void __cuda_stub(void){}' > stub.c
  gcc -shared -fPIC -x c stub.c -o stub/libcuda.so.1
  APPTAINERENV_LD_LIBRARY_PATH=$PWD/stub:/usr/local/lib apptainer exec <sif> ...
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
    "decoder.layers.mlp.linear_fc1.weight",  # 33.6 GB, keep it last by default
]


def stats(ckpt, keys):
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
            w = t[layer].float()
            yield {
                "tensor": short,
                "layer": layer,
                "wmax": w.abs().max().item(),
                "wrms": w.pow(2).mean().sqrt().item(),
            }
            del w
        del t


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
    ap.add_argument("ckpt", type=Path, help="one iter_* directory")
    ap.add_argument("--run", required=True, help="label for this run, e.g. flagship")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument(
        "--tensors", default="all", help="comma-separated suffixes to include, or 'all'"
    )
    ap.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="rotate tensor order, so parallel workers do not all "
        "hit the 33 GB linear_fc1 at the same moment",
    )
    args = ap.parse_args()

    keys = (
        TENSORS
        if args.tensors == "all"
        else [k for k in TENSORS if any(s in k for s in args.tensors.split(","))]
    )
    if args.rotate:
        r = args.rotate % len(keys)
        keys = keys[r:] + keys[:r]

    it = int(re.search(r"\d+", args.ckpt.name).group())
    rows = [{"iter": it, "run": args.run, **r} for r in stats(args.ckpt, keys)]
    if not rows:
        raise SystemExit(f"no weight tensors read from {args.ckpt}")

    # One file per invocation; the driver concatenates.
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "run", "tensor", "layer", "wmax", "wrms"])
        w.writeheader()
        w.writerows(_fmt(r) for r in rows)
    print(f"{args.run} {it}: {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
