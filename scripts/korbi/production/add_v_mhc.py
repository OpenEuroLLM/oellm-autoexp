#!/usr/bin/env python3
"""Add the missing mHC attention-mixer tensors to the six first-class V layers.

The V checkpoint from insert_v_layers.py carries the new V attention tensors but not
``mhc_attn_mixer``, which every TransformerLayer with an attention sublayer owns when
``mhc`` is enabled.  Strict DCP load therefore fails at layer 10 with::

    Missing key in checkpoint state_dict:
      model.module.decoder.layers.10.mhc_attn_mixer.phi

This script appends the model and optimizer-state tensors for the V layers, hard-linking
the existing 1.1 TB chunks.  The mixer is initialized exactly like MHCMixer.reset_parameters:
phi = 0, b_pre = logit(1/n), b_post = 0, b_res = identity_bias * I, alpha = 0.01.  Together
with the V layer's zero output projection this keeps the inserted layer function-preserving.

Usage:
  python add_v_mhc.py --src <v1>/ckpt/iter_0035900 --dst <v2>/ckpt/iter_0035900 [--dry-run]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dcp_surgery import write_surgery  # noqa: E402

V_LAYERS = (10, 27, 44, 62, 79, 96)
STREAMS = 4
HIDDEN = 2048
IDENTITY_BIAS = 5.0
ALPHA_INIT = 0.01
ADAM = ("exp_avg", "exp_avg_sq")
MUOWN = ("g", "m_g", "v_g", "v_norm", "m_u")


def _adam(*shape: int) -> dict[str, torch.Tensor]:
    return {"exp_avg": torch.zeros(shape), "exp_avg_sq": torch.zeros(shape)}


def _muown(rows: int, cols: int | None) -> dict[str, torch.Tensor]:
    return {
        "g": torch.zeros(rows, 1),
        "m_g": torch.zeros(rows, 1),
        "v_g": torch.zeros(rows, 1),
        "v_norm": torch.ones(rows, 1),
        "m_u": torch.zeros(rows, cols),
    }


def mixer_tensors(layer: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(model tensors, optimizer-state tensors) for one V layer's attention mixer."""
    n = STREAMS
    flat = n * HIDDEN
    model = f"model.module.decoder.layers.{layer}.mhc_attn_mixer."
    opt = f"optimizer.0.state.module.module.module.decoder.layers.{layer}.mhc_attn_mixer."
    tensors: dict[str, torch.Tensor] = {
        model + "phi": torch.zeros(flat, n + n + n * n),
        model + "b_pre": torch.full((n,), math.log((1.0 / n) / (1.0 - 1.0 / n))),
        model + "b_post": torch.zeros(n),
        model + "b_res": torch.eye(n) * IDENTITY_BIAS,
        model + "alpha_pre": torch.full((1,), ALPHA_INIT),
        model + "alpha_post": torch.full((1,), ALPHA_INIT),
        model + "alpha_res": torch.full((1,), ALPHA_INIT),
    }
    states: dict[str, torch.Tensor] = {}
    # phi is the one mHC parameter Adam owns (use_adam_with_emerging_optimizers).
    for k, v in _adam(flat, n + n + n * n).items():
        states[opt + "phi." + k] = v
    for name in ("b_pre", "b_post"):
        for k, v in _adam(n).items():
            states[opt + name + "." + k] = v
    for name in ("alpha_pre", "alpha_post", "alpha_res"):
        for k, v in _adam(1).items():
            states[opt + name + "." + k] = v
    for k, v in _muown(n, n).items():
        states[opt + "b_res." + k] = v
    return tensors, states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--layers", default=",".join(map(str, V_LAYERS)))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",") if x]

    md = FileSystemReader(a.src).read_metadata()
    replace: dict[str, torch.Tensor] = {}
    for layer in layers:
        model, states = mixer_tensors(layer)
        replace.update(model)
        replace.update(states)
    print(f"V layers: {layers}")
    print(f"new mHC tensors: {len(replace)}")
    for k in list(replace)[:8]:
        print(" ", k, tuple(replace[k].shape))
    if a.dry_run:
        print("dry run: no checkpoint written")
        return
    # The output contains the six V layers' self-attention tensors already, so no rename.
    write_surgery(a.src, a.dst, md, replace, tag="v_mhc", thread_count=16)
    (a.dst.parent / "latest_checkpointed_iteration.txt").write_text("35900\n")
    print(f"wrote {a.dst}")


if __name__ == "__main__":
    main()
