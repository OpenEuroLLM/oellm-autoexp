#!/usr/bin/env python3
"""Insert six first-class V shared-KV attention layers into the 35900 ambitious checkpoint.

98-layer source -> 104-layer destination: V after the old CKDA layers 9/25/41/58/74/90.
The V layers start with o_proj = 0 (exactly function-preserving), random q / gate, unit q-norm
and input norm, and zero optimizer moments. Existing 1.1 TB chunks are hard-linked; only
metadata FQNs shift (dcp_surgery rename). No source attention tensors are read.

Usage:
  python insert_v_layers.py --src <ckpt>/iter_0035900 --dst <new>/ckpt/iter_0035900 [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp  # noqa: F401  (metadata types)
from torch.distributed.checkpoint import FileSystemReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcp_surgery import write_surgery  # noqa: E402

ORIG_PATTERN = "*ENLELELELELELELE*ELELELELELELELE*ELELELELELELELEN*ELELELELELELELE*ELELELELELELELE*ELELELELELELELE"
INSERT_AFTER = [9, 25, 41, 58, 74, 90]
HEADS, D, GROUPS, HIDDEN = 16, 256, 2, 2048
QKV_ROWS = HEADS * D * 2  # two query groups worth of [q | gate]
PROJ_COLS = HEADS * D
MUOWN = ("g", "m_g", "v_g", "v_norm", "m_u")
ADAM = ("exp_avg", "exp_avg_sq")


def new_index(old: int) -> int:
    return old + sum(1 for p in INSERT_AFTER if p < old)


def build_pattern() -> tuple[str, list[int]]:
    out, v_layers = [], []
    for old, ch in enumerate(ORIG_PATTERN):
        out.append(ch)
        if old in INSERT_AFTER:
            if ch != "L":
                raise SystemExit(f"insert after {old}: pattern char {ch!r} is not L")
            v_layers.append(len(out))
            out.append("V")
    if len(out) != 104:
        raise SystemExit(f"pattern length {len(out)} != 104")
    return "".join(out), v_layers


def _state(rows: int, cols: int | None, kind: str) -> dict[str, torch.Tensor]:
    if kind == "adam":
        return {"exp_avg": torch.zeros(rows), "exp_avg_sq": torch.zeros(rows)}
    return {
        "g": torch.zeros(rows, 1),
        "m_g": torch.zeros(rows, 1),
        "v_g": torch.zeros(rows, 1),
        "v_norm": torch.ones(rows, 1),
        "m_u": torch.zeros(rows, cols),
    }


def v_tensors(layer: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + layer)
    qkv = torch.randn(QKV_ROWS, HIDDEN, generator=g) * 0.02
    proj = torch.zeros(HIDDEN, PROJ_COLS)
    qn = torch.ones(D)
    inp = torch.ones(HIDDEN)
    mp = f"model.module.decoder.layers.{layer}."
    op = f"optimizer.0.state.module.module.module.decoder.layers.{layer}."
    out = {
        mp + "self_attention.linear_qkv.weight": qkv,
        mp + "self_attention.linear_proj.weight": proj,
        mp + "self_attention.q_layernorm.weight": qn,
        mp + "input_layernorm.weight": inp,
    }
    for rel, rows, cols in (("self_attention.linear_qkv.weight", QKV_ROWS, HIDDEN),
                            ("self_attention.linear_proj.weight", HIDDEN, PROJ_COLS)):
        for k, v in _state(rows, cols, "muown").items():
            out[op + rel + "." + k] = v
    for rel, n in (("self_attention.q_layernorm.weight", D), ("input_layernorm.weight", HIDDEN)):
        for k, v in _state(n, None, "adam").items():
            out[op + rel + "." + k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    pattern, v_layers = build_pattern()
    print("pattern:", pattern)
    print("V layers:", v_layers)
    print("old->new:", {i: new_index(i) for i in range(98)})
    md = FileSystemReader(a.src).read_metadata()
    rename = {}
    for k in md.state_dict_metadata:
        m = re.search(r"decoder\.layers\.(\d+)", k)
        if not m:
            continue
        old = int(m.group(1))
        new = new_index(old)
        if new != old:
            rename[k] = k.replace(f"decoder.layers.{old}", f"decoder.layers.{new}", 1)
    replace = {}
    for v in v_layers:
        replace.update(v_tensors(v, a.seed))
    print(f"rename: {len(rename)} FQNs; new V tensors: {len(replace)}")
    if a.dry_run:
        print("dry run: no checkpoint written")
        return
    write_surgery(a.src, a.dst, md, replace, tag="v_insert", rename=rename, thread_count=16)
    (a.dst.parent / "latest_checkpointed_iteration.txt").write_text("35900\n")
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
