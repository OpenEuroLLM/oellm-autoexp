#!/usr/bin/env python3
"""Write an ablated copy of an HF export with one Engram layer's output zeroed.

The Engram layer is exported as ``model.layers.<L>.mixer`` with ``value_proj.weight`` as the
output projection (no bias). Zeroing that weight makes the layer's residual contribution exactly
zero while leaving the rest of the checkpoint untouched, so the ablation needs no model-code
change and no re-export.

The copy is cheap: every file is symlinked, and only the shard that holds the target tensor is
rewritten. Run inside the training image (safetensors is available there):

    python scripts/korbi/hf_export/zero_engram.py \
        --src /gpfs/.../hf_exports/ambitious_v4b_learned_fp8_47000 \
        --dst /gpfs/.../hf_exports/ambitious_v4b_learned_fp8_47000_eng52off \
        --layer 52
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source export directory")
    ap.add_argument("--dst", required=True, help="destination (must not exist or be empty)")
    ap.add_argument("--layer", type=int, required=True, help="transformer layer index of the Engram")
    ap.add_argument("--suffix", default="value_proj.weight", help="tensor suffix to zero")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"no index at {index_path}", file=sys.stderr)
        return 1
    weight_map = json.loads(index_path.read_text())["weight_map"]

    prefix = f"model.layers.{args.layer}.mixer."
    targets = [k for k in weight_map if k.startswith(prefix) and k.endswith(args.suffix)]
    if not targets:
        print(f"no tensors matching {prefix}*{args.suffix}", file=sys.stderr)
        return 1
    shards = sorted({weight_map[k] for k in targets})
    print(f"zeroing {targets} in shard(s) {shards}")
    if args.dry_run:
        return 0

    if dst.exists() and any(dst.iterdir()):
        print(f"{dst} is not empty", file=sys.stderr)
        return 1
    dst.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file, save_file  # noqa: PLC0415
    import torch  # noqa: PLC0415

    for entry in sorted(src.iterdir()):
        if entry.name in shards:
            continue
        os.symlink(entry.resolve(), dst / entry.name)

    for shard in shards:
        tensors = load_file(src / shard)
        for key in targets:
            if key not in tensors:
                print(f"{key} missing from {shard}", file=sys.stderr)
                return 1
            tensor = tensors[key]
            print(f"  {key}: {tuple(tensor.shape)} {tensor.dtype} -> zeros")
            tensors[key] = torch.zeros_like(tensor)
        save_file(tensors, dst / shard, metadata={"format": "pt"})
        del tensors

    # verify
    check = load_file(dst / shards[0])
    for key in targets:
        if float(check[key].abs().sum()) != 0.0:
            print(f"verification failed: {key} is not zero", file=sys.stderr)
            return 1
    print(f"wrote {dst} (ablated layer {args.layer}, {args.suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
