#!/usr/bin/env python3
"""Update the checkpointed ``args`` for a 98 -> 104 layer V insertion.

Megatron's ``check_checkpoint_args`` compares ``state_dict['args'].num_layers`` with
the live model's 104 layers.  The tensor surgery did not rewrite that pickled object,
so a fully loadable V checkpoint still stops with::

    AssertionError: num_layers value from checkpoint (98) is not equal to
    the input argument value (104).

This script replaces only the ``args`` object; every tensor chunk is hard-linked.

Usage:
  python fix_v_args.py --src <v5>/ckpt/iter_0035900 --dst <v6>/ckpt/iter_0035900
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submodules" / "Megatron-LM"))

from dcp_surgery import write_surgery  # noqa: E402
from insert_v_layers import build_pattern  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    pattern, v_layers = build_pattern()
    md = FileSystemReader(a.src).read_metadata()
    if "args" not in md.state_dict_metadata:
        raise SystemExit("checkpoint has no args object")
    sd = {"args": None}
    dcp.load(sd, storage_reader=FileSystemReader(a.src))
    args = sd["args"]
    before = getattr(args, "num_layers", None)
    args.num_layers = len(pattern)
    args.hybrid_layer_pattern = pattern
    args.kv_share_layers = v_layers
    args.kv_share_source_layers = [0, 18, 35, 53, 70, 87]
    args.ckda_branch = None
    args.ckda_branch_layers = None
    args.ckda_branch_serial = False
    args.attention_head_temperature = False
    print(f"args.num_layers {before} -> {args.num_layers}")
    print("args.hybrid_layer_pattern:", args.hybrid_layer_pattern)
    print("args.kv_share_layers:", args.kv_share_layers)
    if a.dry_run:
        return

    write_surgery(a.src, a.dst, md, {"args": args}, tag="v_args", thread_count=8)
    (a.dst.parent / "latest_checkpointed_iteration.txt").write_text("35900\n")
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
