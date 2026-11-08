"""Fast inference path vs the reference on the reduced v4_2ph model (random weights).

Per component (same input to both paths): the mHC read (+ input norm) and write, attention, the
MoE layer, the Engram layer (table on the GPU and pinned in host memory) -- bitwise equality or
the largest difference; then the whole forward (logits, NLL of random targets) and a check that
the fast forward never synchronizes with the host (torch.cuda.set_sync_debug_mode).

    CC=/usr/bin/gcc PYTHONPATH=flash-linear-attention python experiment/013_hf_prefill_speed/scripts/check_fast.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_model import build  # noqa: E402


def mode(m: str):
    os.environ["OELLM_HF_IMPL"] = m


def diff(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {"equal": bool(torch.equal(a, b)), "max_abs": d.max().item(), "frac_diff": (d > 0).float().mean().item(),
            "ref_rms": b.pow(2).mean().sqrt().item()}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", nargs="+", default=["attention", "moe", "engram", "ckda", "moe", "ckda", "moe"])
    p.add_argument("--shape", default="4x512")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    model, cfg = build(a.layers)
    b, s = map(int, a.shape.split("x"))
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, cfg.vocab_size, (b, s), generator=g).cuda()
    res = {}
    base = model.model
    streams = base._enter(base._embed(ids, None, None))
    # perturb the streams so the 4 copies differ (as after a few layers)
    streams = (streams.float() + 0.3 * torch.randn(streams.shape, generator=g).cuda()).to(streams.dtype)
    for i, layer in enumerate(base.layers):
        kind = layer.kind
        norm = None if kind == "engram" else layer.input_layernorm
        mode("reference")
        xr, pr = layer.mhc.read(streams)
        xin_r = layer._norm(xr)
        mode("fast")
        xin_f, pf = layer.mhc.read(streams, norm)
        res[f"L{i}.{kind}.mhc_read_in"] = diff(xin_f, xin_r)
        res[f"L{i}.{kind}.h_post"] = diff(pf[0].view_as(pr[0]), pr[0])
        res[f"L{i}.{kind}.h_res"] = diff(pf[1].view_as(pr[1]), pr[1])
        for m in ("reference", "fast"):
            mode(m)
            out, _ = layer._mix(xin_r, None, ids, None, None, False)
            res.setdefault(f"L{i}.{kind}.mixer", {})[m] = out
        res[f"L{i}.{kind}.mixer"] = diff(res[f"L{i}.{kind}.mixer"]["fast"], res[f"L{i}.{kind}.mixer"]["reference"])
        mode("reference")
        wr = layer.mhc.write(streams, out, pr)
        mode("fast")
        wf = layer.mhc.write(streams, out, pr)
        res[f"L{i}.{kind}.mhc_write"] = diff(wf, wr)
        mode("reference")
        streams, _ = layer(streams, None, ids, None, None, False)
    # Engram with the table pinned in host memory
    eng = [layer for layer in base.layers if layer.kind == "engram"]
    if eng:
        mix = eng[0].mixer
        x = torch.randn(b, s, cfg.hidden_size, generator=g).cuda().bfloat16()
        mode("fast")
        on_gpu, _ = mix(x, ids, None, None)
        mix.offload_table("cpu")
        assert mix.pin_table()
        pinned, _ = mix(x, ids, None, None)
        res["engram.pinned_vs_gpu_table"] = diff(pinned, on_gpu)
    # whole forward
    labels = torch.randint(3, cfg.vocab_size, (b, s), generator=g).cuda()
    logits = {}
    for m in ("reference", "fast"):
        mode(m)
        logits[m] = model(input_ids=ids).logits.float()
    res["logits"] = diff(logits["fast"], logits["reference"])
    nll = {m: torch.nn.functional.cross_entropy(v[:, :-1].reshape(-1, v.shape[-1]), labels[:, 1:].reshape(-1)).item()
           for m, v in logits.items()}
    res["nll"] = {**nll, "delta": nll["fast"] - nll["reference"]}
    # no host synchronization in the fast forward
    mode("fast")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(input_ids=ids)
    torch.cuda.set_sync_debug_mode("default")
    syncs = [str(w.message).splitlines()[0] for w in caught if "synchroniz" in str(w.message).lower()]
    res["host_syncs"] = {"count": len(syncs), "first": syncs[:5]}
    for k, v in res.items():
        print(k, json.dumps(v))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
