#!/usr/bin/env python3
"""Add Megatron's attention_head_temperature parameters to an fsdp_dtensor checkpoint (docs/recall_surgery_ambitious.md
§14.8, small31 arm T): per softmax-attention layer ``self_attention.head_temperature`` = ones(heads) (the identity:
the model computes exactly what it computed before) and zero Adam states. The fsdp_dtensor load is strict on model
keys, so a checkpoint without them does not load into a model with ``attention_head_temperature``. The new tree
hard-links every .distcp of the source (dcp_surgery.write_surgery); the new keys are read back.

    add_head_temperature.py --src <run>/ckpt/iter_N --dst <run2>/ckpt/iter_N
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = "model.module.decoder.layers.{l}.self_attention.{n}"
OPT = "optimizer.0.state.module.module.module.decoder.layers.{l}.self_attention.{n}.{s}"


def main(argv=None) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    from dcp_surgery import write_surgery

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists")
    md = FileSystemReader(a.src).read_metadata()
    smd = md.state_dict_metadata
    # softmax-attention layers: the ones with a fused linear_qkv (q heads = linear_proj columns / head dim)
    layers = sorted({int(m.group(1)) for k in smd
                     if (m := re.match(r"model\.module\.decoder\.layers\.(\d+)\.self_attention\.linear_proj\.weight$", k))})
    replace = {}
    for l in layers:
        if MODEL.format(l=l, n="head_temperature") in smd:
            raise SystemExit(f"layer {l} has head_temperature already")
        d = smd[MODEL.format(l=l, n="q_layernorm.weight")].size[0]
        heads = smd[MODEL.format(l=l, n="linear_proj.weight")].size[1] // d
        replace[MODEL.format(l=l, n="head_temperature")] = torch.ones(heads, dtype=torch.float32)
        for s in ("exp_avg", "exp_avg_sq"):
            replace[OPT.format(l=l, n="head_temperature", s=s)] = torch.zeros(heads, dtype=torch.float32)
    print(f"attention layers {layers}: {len(replace)} new tensors", flush=True)
    write_surgery(a.src, a.dst, md, replace, tag="headtemp")
    back = {k: torch.empty_like(v) for k, v in replace.items()}
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in back if not torch.equal(back[k], replace[k])]
    if bad:
        raise SystemExit(f"read-back mismatch: {bad[:5]}")
    print(f"done -> {a.dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
