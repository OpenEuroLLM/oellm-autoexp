#!/usr/bin/env python3
"""Check of the attn_kv@<n> surgery (surgery.pool_kv) on fla's Attention -- the hybrid's attention class, gated,
NoPE (docs/recall_surgery_ambitious.md section 3.7). Run in the surgery job before any attn_kv arm (GPU: the
flash-attn path the training and the readout use; CPU: the SDPA fallback):

 1. shapes: the pooled k_proj / v_proj load STRICTLY into Attention(num_kv_heads=n)
 2. identical heads inside each group -> the pooled GQA layer equals the original MHA layer
 3. arbitrary weights: the pooled GQA layer equals the MHA layer with every head's k / v replaced by its group's mean
    (the intended function)
 4. grouping: perturbing KV head 0 changes exactly query heads 0 .. G-1 (contiguous groups)
 + the size of the function change (pooled vs original) on random weights, for the record

    PYTHONPATH=<code dir> python kv_pool_check.py --surgery <code dir>/lm_scaling/megatron_ext/surgery.py [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surgery", required=True, help="the surgery.py whose pool_kv is checked")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv", type=int, default=2)
    a = ap.parse_args(argv)

    import torch
    from fla.layers import attn as fla_attn
    from fla.layers.attn import Attention

    spec = importlib.util.spec_from_file_location("_surgery", a.surgery)
    surgery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(surgery)
    dev = a.device
    flash = fla_attn.flash_attn_func is not None and dev.startswith("cuda")
    torch.manual_seed(0)

    def build(hidden, heads, kv, dtype):
        return Attention(hidden_size=hidden, num_heads=heads, num_kv_heads=kv, qk_norm=False, output_gate=True,
                         use_rope=False, layer_idx=0).to(dev, dtype)

    def pooled_copy(mha, kv):
        gqa = build(mha.hidden_size, mha.num_heads, kv, next(mha.parameters()).dtype)
        sd = mha.state_dict()
        for k in ("k_proj.weight", "v_proj.weight"):
            sd[k] = surgery.pool_kv(sd[k], mha.num_heads, kv, mha.head_dim)
        gqa.load_state_dict(sd, strict=True)
        return gqa

    # flash-attn takes half precision only; the CPU / SDPA path runs float64 / float32
    cases = ([(256, 8, a.kv, 128, torch.bfloat16), (2048, 32, a.kv, 512, torch.bfloat16)] if flash else
             [(256, 8, a.kv, 33, torch.float64), (2048, 32, a.kv, 64, torch.float32)])
    for hidden, heads, kv, L, dtype in cases:
        mha = build(hidden, heads, heads, dtype)
        hd, G = hidden // heads, heads // kv
        x = torch.randn(2, L, hidden, device=dev, dtype=dtype)
        with torch.no_grad():
            gqa = pooled_copy(mha, kv)
            assert gqa.k_proj.weight.shape == (kv * hd, hidden) and gqa.num_kv_heads == kv
            ident = copy.deepcopy(mha)
            for p in (ident.k_proj.weight, ident.v_proj.weight):
                w = p.view(kv, G, hd, hidden)
                w.copy_(w[:, :1].expand_as(w).clone())
            y0 = ident(x)[0]
            d2 = (y0 - pooled_copy(ident, kv)(x)[0]).abs().max().item()
            ref = copy.deepcopy(mha)
            for p in (ref.k_proj.weight, ref.v_proj.weight):
                w = p.view(kv, G, hd, hidden)
                w.copy_(w.double().mean(1, keepdim=True).expand_as(w).to(w.dtype).clone())
            d3 = (ref(x)[0] - gqa(x)[0]).abs().max().item()
            cap = {}
            h = gqa.o_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("o", i[0].detach().clone()))
            gqa(x)
            o_a = cap["o"]
            gqa.v_proj.weight[:hd] += 1.0
            gqa(x)
            o_b = cap["o"]
            h.remove()
            touched = [i for i, c in enumerate(((o_a - o_b).float().abs().view(2, L, heads, hd)
                                                .amax(dim=(0, 1, 3)) > 0).tolist()) if c]
            y_orig = mha(x)[0]
            dchg = (y_orig - pooled_copy(mha, kv)(x)[0]).abs().max().item()
        scale = y0.abs().max().item()
        tol = {torch.float64: 1e-12, torch.float32: 1e-5}.get(dtype, 2e-2 * max(scale, 1e-3))
        ok = d2 <= tol and d3 <= tol and touched == list(range(G))
        print(f"[kv_pool_check] {'PASS' if ok else 'FAIL'} hidden {hidden} heads {heads}->{kv} KV ({G} per group) "
              f"{str(dtype)[6:]} {dev} {'flash-attn' if flash else 'sdpa'}: identical heads |d| {d2:.2e}, "
              f"GQA vs group-mean MHA |d| {d3:.2e} (tol {tol:.1e}), KV head 0 feeds query heads "
              f"{touched[0] if touched else '-'}..{touched[-1] if touched else '-'}; pooled vs original "
              f"|d| {dchg:.2e} (output scale {y_orig.abs().max().item():.2e})", flush=True)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
