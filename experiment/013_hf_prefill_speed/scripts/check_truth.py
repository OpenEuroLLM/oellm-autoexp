"""Fast and reference mixers against a float64 copy of the same module (the "truth"): both bf16
paths should have errors of the same size if the fast one only changes accumulation order.

    CC=/usr/bin/gcc PYTHONPATH=flash-linear-attention python experiment/013_hf_prefill_speed/scripts/check_truth.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_model import build  # noqa: E402


def rel(a, t):
    a, t = a.double(), t.double()
    return ((a - t).norm() / t.norm()).item()


@torch.no_grad()
def main():
    model, cfg = build(["attention", "moe", "engram"])
    b, s = 2, 384
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(3, cfg.vocab_size, (b, s), generator=g).cuda()
    x = torch.randn(b, s, cfg.hidden_size, generator=g).cuda().bfloat16()
    out = {}
    for layer in model.model.layers:
        mix = layer.mixer
        args = (ids, None, None) if layer.kind == "engram" else ()
        xin = x if layer.kind == "engram" else layer.input_layernorm(x)
        res = {}
        for m in ("reference", "fast"):
            os.environ["OELLM_HF_IMPL"] = m
            y = mix(xin, *args)
            res[m] = y[0] if isinstance(y, tuple) else y
        os.environ["OELLM_HF_IMPL"] = "reference"
        m64 = copy.deepcopy(mix).double()
        if layer.kind == "engram":  # the table rows as the bf16 model has them
            m64.table.data = mix.table.data.double()
        y = m64(xin.double(), *args)
        truth = y[0] if isinstance(y, tuple) else y
        out[layer.kind] = {"ref_vs_truth": rel(res["reference"], truth), "fast_vs_truth": rel(res["fast"], truth),
                           "fast_vs_ref": rel(res["fast"], res["reference"])}
        print(layer.kind, json.dumps(out[layer.kind]), flush=True)
        del m64


if __name__ == "__main__":
    main()
