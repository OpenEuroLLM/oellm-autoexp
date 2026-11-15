#!/usr/bin/env python3
"""Widen the dense SwiGLU MLPs of an fsdp_dtensor checkpoint, function-preserving (docs/recall_surgery_ambitious.md
§14.8, small31 arm W: the dense analogue of "one more active expert"). Per MLP layer, F -> F_new hidden units:

  linear_fc1.weight_w / weight_v   (F, H) -> (F_new, H): the new rows random N(0, init_std) (the model's init)
  linear_fc2.weight                (H, F) -> (H, F_new): the new columns ZERO, so the layer's output is unchanged
  optimizer states (MuOwn)         m_u of fc1 gets zero rows, of fc2 zero columns; per-row states of fc1 (g / m_g /
                                   v_g / v_norm, when the checkpoint has them) get fresh rows (g = the new row's norm,
                                   v_norm = the median of the existing rows, momenta 0); fc2's per-row states keep
                                   their shape

Megatron has one ffn_hidden_size for every MLP layer, so all MLP layers are widened (run the widened model with
ffn_hidden_size = F_new). The new tree hard-links every .distcp of the source (dcp_surgery.write_surgery) and holds a
manifest widen_ffn.json; the rewritten keys are read back and compared.

    widen_ffn.py --src <b7>/ckpt/iter_38146 --dst <W>/ckpt/iter_38146 --ffn 4608 [--init-std 0.02] [--seed 1234]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MLP = re.compile(r"^(model\.module|optimizer\.0\.state\.module\.module\.module)\.decoder\.layers\.(\d+)\.mlp\."
                 r"(linear_fc1\.weight_[wv]|linear_fc2\.weight)(?:\.(\w+))?$")


def widen(key: str, t: torch.Tensor, f_new: int, std: float, gen: torch.Generator, fc1_new: dict) -> torch.Tensor:
    m = MLP.match(key)
    _, layer, name, state = m.groups()
    t = t.float()
    if name.startswith("linear_fc1"):
        f, h = t.shape
        add = f_new - f
        if state is None:  # the weight: new random rows
            new = torch.randn(add, h, generator=gen) * std
            fc1_new[(layer, name)] = new
            return torch.cat([t, new], 0)
        if state == "m_u" or state in ("m_g", "v_g"):
            return torch.cat([t, torch.zeros((add,) + tuple(t.shape[1:]))], 0)
        if state == "g":
            return torch.cat([t, fc1_new[(layer, name)].norm(dim=1, keepdim=True)], 0)
        if state == "v_norm":
            return torch.cat([t, torch.full((add,) + tuple(t.shape[1:]), float(t.median()))], 0)
        raise SystemExit(f"{key}: unknown fc1 state {state}")
    # linear_fc2: (H, F) -> (H, F_new) zero columns; per-row states (H, 1) unchanged
    if state is None or state == "m_u":
        h, f = t.shape
        return torch.cat([t, torch.zeros(h, f_new - f)], 1)
    return t


def apply(a) -> int:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    from dcp_surgery import write_surgery

    t0 = time.time()
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists (a partial attempt? remove it after checking)")
    md = FileSystemReader(a.src).read_metadata()
    smd = md.state_dict_metadata
    keys = sorted(k for k in smd if MLP.match(k) and hasattr(smd[k], "size"))
    # the weights first (the fc1 per-row g state needs the new rows)
    keys.sort(key=lambda k: (MLP.match(k).group(4) is not None, k))
    src = {k: torch.empty(tuple(smd[k].size), dtype=smd[k].properties.dtype) for k in keys}
    dcp.load(src, storage_reader=FileSystemReader(a.src), no_dist=True)
    f_old = {tuple(v.shape)[0] for k, v in src.items() if k.startswith("model.") and "linear_fc1" in k}
    if len(f_old) != 1 or a.ffn <= next(iter(f_old)):
        raise SystemExit(f"fc1 widths {f_old}: expected one width below --ffn {a.ffn}")
    gen = torch.Generator().manual_seed(a.seed)
    fc1_new, replace = {}, {}
    for k in keys:
        replace[k] = widen(k, src[k], a.ffn, a.init_std, gen, fc1_new).to(src[k].dtype).contiguous()
    layers = sorted({int(MLP.match(k).group(2)) for k in keys})
    print(f"{len(replace)} tensors widened {next(iter(f_old))} -> {a.ffn} in {len(layers)} MLP layers "
          f"({time.time() - t0:.0f} s)", flush=True)
    write_surgery(a.src, a.dst, md, replace, tag="widen", chunk_bytes=64 << 20)
    back = {k: torch.empty_like(v) for k, v in replace.items()}
    dcp.load(back, storage_reader=FileSystemReader(a.dst), no_dist=True)
    bad = [k for k in back if not torch.equal(back[k], replace[k])]
    if bad:
        raise SystemExit(f"read-back mismatch: {bad[:5]}")
    old_rows = [src[k].float().norm(dim=1).median().item() for k in keys
                if k.startswith("model.") and "linear_fc1" in k]
    new_rows = [v.norm(dim=1).median().item() for v in fc1_new.values()]
    with open(os.path.join(a.dst, "widen_ffn.json"), "w") as f:
        json.dump({"src": a.src, "ffn_old": next(iter(f_old)), "ffn_new": a.ffn, "init_std": a.init_std,
                   "seed": a.seed, "layers": layers, "tensors": len(replace),
                   "fc1_row_norm_median_old": sorted(old_rows)[len(old_rows) // 2],
                   "fc1_row_norm_median_new": sorted(new_rows)[len(new_rows) // 2],
                   "seconds": round(time.time() - t0)}, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s -> {a.dst} (fc1 row norm median: trained "
          f"{sorted(old_rows)[len(old_rows) // 2]:.3f}, new {sorted(new_rows)[len(new_rows) // 2]:.3f})", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ffn", type=int, required=True, help="the new ffn_hidden_size")
    ap.add_argument("--init-std", type=float, default=0.02, help="std of the new fc1 rows (the model's init_method_std)")
    ap.add_argument("--seed", type=int, default=1234)
    return apply(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
