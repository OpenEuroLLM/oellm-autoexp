"""Standalone check of one FLA layer on a saved input (debug_fla_paths --save): determinism of
each kernel path, weights unchanged by a call, and agreement with an fp32 naive recurrence.

    python -m oellm_autoexp.hf_export.fla_layer_check <saved.pt> [--paths a,b] [--warm 781,1682]
"""

from __future__ import annotations

import argparse
import importlib

import torch

PATHS = {
    "hf": dict(cache=True, cu=False),     # the HF wrapper: fresh Cache, use_cache, no cu_seqlens
    "nocache": dict(cache=False, cu=False),
    "train": dict(cache=False, cu=True),  # Megatron: fused projections, cu_seqlens
    "cache_cu": dict(cache=True, cu=True),
}


def build(saved, dtype, device="cuda"):
    module, name = saved["spec"]["class"].rsplit(".", 1)
    layer = getattr(importlib.import_module(module), name)(**saved["spec"]["kwargs"], layer_idx=0)
    layer.load_state_dict(saved["state_dict"])
    return layer.to(device, dtype).eval()


GUARD = False


def call(layer, x, path):
    import contextlib

    from fla.models.utils import Cache

    ctx = torch.cuda.device(x.device) if GUARD else contextlib.nullcontext()
    with ctx:
        return _call(layer, x, path)


def _call(layer, x, path):
    from fla.models.utils import Cache

    kw = {}
    if PATHS[path]["cache"]:
        kw.update(past_key_values=Cache(), use_cache=True)
    else:
        kw.update(use_cache=False)
    if PATHS[path]["cu"]:
        kw["cu_seqlens"] = torch.tensor([0, x.shape[1]], device=x.device, dtype=torch.int32)
    with torch.no_grad():
        return layer(hidden_states=x, **kw)[0].float()


def checksum(layer):
    return {k: float(v.float().abs().sum()) for k, v in layer.state_dict().items()}


def nonfinite(o):
    bad = (~torch.isfinite(o)).reshape(o.shape[1], -1).any(-1)
    pos = torch.nonzero(bad).flatten()
    return f"{int(bad.sum())} non-finite (first {pos[:3].tolist()})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("saved")
    p.add_argument("--paths", default="hf,nocache,train,cache_cu")
    p.add_argument("--warm", default="", help="comma-separated lengths run first (random input, hf path)")
    p.add_argument("--no-ref", action="store_true")
    p.add_argument("--device", default="cuda:0", help="the layer's GPU; the current device stays cuda:0")
    p.add_argument("--guard", action="store_true", help="calls under torch.cuda.device(x.device)")
    a = p.parse_args()
    global GUARD
    GUARD = a.guard
    torch.cuda.set_device(0)
    saved = torch.load(a.saved, weights_only=False)
    torch.manual_seed(0)
    layer = build(saved, torch.bfloat16, a.device)
    x = saved["x"].to(a.device, torch.bfloat16)
    print(f"{saved['spec']['class']}, x {tuple(x.shape)}", flush=True)
    w0 = checksum(layer)
    for n in [int(t) for t in a.warm.split(",") if t]:
        xw = x[:, :n].clone()
        o = call(layer, xw, "hf")
        print(f"warm {n}: {nonfinite(o)}", flush=True)
    outs = {}
    for path in a.paths.split(","):
        o1, o2 = call(layer, x, path), call(layer, x, path)
        same = torch.equal(torch.nan_to_num(o1, 7.0), torch.nan_to_num(o2, 7.0))
        changed = [k for k, v in checksum(layer).items() if v != w0[k]]
        print(f"{path:9s} {nonfinite(o1)}; repeat identical {same}; weights changed {changed[:4]}", flush=True)
        outs[path] = o1
    if "train" in outs:
        for path, o in outs.items():
            ok = torch.isfinite(o) & torch.isfinite(outs["train"])
            print(f"{path:9s} max |o - train| {float((o - outs['train']).abs()[ok].max()):.3g}", flush=True)
    if not a.no_ref:
        ref_layer = build(saved, torch.float32, "cuda:0")
        ref_layer.backend = "naive_recurrent"
        ref = call(ref_layer, x.float().to("cuda:0"), "nocache").to(a.device)
        print(f"fp32 naive ref: {nonfinite(ref)}, |ref| max {float(ref.abs().max()):.3g}", flush=True)
        for path, o in outs.items():
            ok = torch.isfinite(o)
            err = (o - ref).abs()[ok]
            print(f"{path:9s} vs fp32 ref: max {float(err.max()):.3g}, mean {float(err.mean()):.3g}", flush=True)


if __name__ == "__main__":
    main()
