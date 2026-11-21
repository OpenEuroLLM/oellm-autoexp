#!/usr/bin/env python3
"""Memory of the pure per-token rowmask path vs dense, one attention layer (v4b shapes: 16 / 2 heads, head dim 256),
forward + backward on packed phase-2-like layouts (doclen_sample.py sequences). Every measurement runs in a fresh
subprocess, since a kernel that spills registers grows the CUDA context's local memory for the rest of the process:
  torch_peak   -- max_memory_allocated above the inputs (q / k / v and their gradients' buffers excluded)
  nontorch     -- device memory gone beyond PyTorch's reserved pool after the run (local memory of spilled kernels,
                  module images), vs a process that only created the inputs
  ms           -- one fwd + bwd after a warm-up
Variants: dense (flash_attn varlen, document-masked), block (page_maxpool, one selection per 64 queries, flex),
token (pure per-token, rowmask head-packed), token_spans (the same with the per-span selection). --configs sweeps the
head-packed kernels' launch configs (ROWMASK_HP_CONFIG) at the single-document layout and reports regs / spills.

    token_memory_bench.py --seqs seqs_64k.json [--layouts single,many,mid] [--configs] [--out json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def child(a):
    import time

    import torch
    from megatron.core.transformer.experimental_attention_variant import page_sparse as ps

    dev = "cuda"
    torch.cuda.init()
    torch.zeros(1, device=dev)
    free0 = torch.cuda.mem_get_info()[0]
    res0 = torch.cuda.memory_reserved()
    segs = json.loads(a.segs)
    L = sum(segs)
    torch.manual_seed(0)
    q = torch.randn(1, a.heads, L, a.dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, a.kv, L, a.dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(1, a.heads, L, a.dim, device=dev, dtype=torch.bfloat16)
    for t in (q, k, v):
        t.grad = torch.zeros_like(t)
    scale = a.dim**-0.5
    base = dict(selector="page_maxpool", page=64, topk=32, window=512, sink_pages=1, per_group=True)
    bounds = [0]
    for s_ in segs:
        bounds.append(bounds[-1] + s_)

    def run():
        if a.variant == "dense":
            from flash_attn import flash_attn_varlen_func

            cu = torch.tensor(bounds, dtype=torch.int32, device=dev)
            o = flash_attn_varlen_func(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1), cu, cu,
                                       max(segs), max(segs), causal=True, softmax_scale=scale)
            o.backward(dout[0].transpose(0, 1))
            return
        if a.variant == "block":
            cfg, kern, packed = ps.PageSparseConfig(**base, select_block=64), "flex", True
        else:
            cfg, kern = ps.PageSparseConfig(**base, select_block=1), "rowmask"
            packed = a.variant == "token"
        spans = [(x, y, z, 0, None) for x, y, z in zip(q.split(segs, 2), k.split(segs, 2), v.split(segs, 2))]
        outs = ps.flex_packed_page_attention(spans, cfg, scale, 64, kernel=kern, packed_select=packed)
        torch.cat(outs, 2).backward(dout)

    run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    alloc0 = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    run()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    peak = torch.cuda.max_memory_allocated() - alloc0
    torch.cuda.empty_cache()
    free1 = torch.cuda.mem_get_info()[0]
    res1 = torch.cuda.memory_reserved()
    nontorch = (free0 - free1) - (res1 - res0)
    out = {"ms": ms, "torch_peak_gb": peak / 2**30, "nontorch_gb": nontorch / 2**30}
    if a.variant.startswith("token"):  # the head-packed kernels' register use at the launched configs
        from megatron.core.transformer.experimental_attention_variant import rowmask_attention as ra

        kinfo = {}
        for kern in (ra._fwd_hp_kernel, ra._dq_hp_kernel, ra._dkv_hp_kernel):
            for dc in kern.device_caches.values():
                for c in dc[0].values():
                    try:
                        c._init_handles()
                        kinfo.setdefault(kern.fn.__name__, []).append(
                            {"warps": c.metadata.num_warps, "stages": c.metadata.num_stages, "regs": c.n_regs,
                             "spills": c.n_spills})
                    except Exception as e:  # noqa: BLE001
                        kinfo.setdefault(kern.fn.__name__, []).append(str(e)[:80])
        out["kernels"] = kinfo
    print("RESULT " + json.dumps(out), flush=True)


def spawn(a, variant, segs, env=None):
    cmd = [sys.executable, __file__, "--child", "--variant", variant, "--segs", json.dumps(segs), "--heads",
           str(a.heads), "--kv", str(a.kv), "--dim", str(a.dim)]
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": (p.stderr or p.stdout)[-600:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--variant", default="token")
    ap.add_argument("--segs", default="[65536]")
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--layouts", default="single,many,mid")
    ap.add_argument("--variants", default="dense,block,token,token_spans")
    ap.add_argument("--configs", action="store_true")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.child:
        return child(a)
    layouts = {"single": [65536]}
    if a.seqs:
        seqs = json.load(open(a.seqs))["seqs"]
        by_docs = sorted(seqs, key=len)
        layouts["many"] = by_docs[-1]  # the most documents
        layouts["mid"] = by_docs[len(by_docs) // 2]  # the median document count
        layouts["long_tail"] = max(seqs, key=lambda s: (len([x for x in s if x > 4096]), -max(s)))
    res = {"layouts": {}, "configs": {}}
    for name in a.layouts.split(","):
        segs = layouts[name]
        row = {"docs": len(segs), "longest": max(segs)}
        for var in a.variants.split(","):
            row[var] = spawn(a, var, segs)
            print(name, var, json.dumps(row[var])[:400], flush=True)
        res["layouts"][name] = row
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    if a.configs:
        grid = []
        for fwd in ("4,1", "4,2", "8,1", "8,2", "8,3"):
            grid.append(f"fwd={fwd};dq=8,1;dkv=8,1,64")
        for dq in ("4,1", "4,2", "8,2"):
            grid.append(f"fwd=8,2;dq={dq};dkv=8,1,64")
        for dkv in ("4,1,32", "4,1,64", "8,1,32", "8,1,128", "8,2,64", "8,2,32"):
            grid.append(f"fwd=8,2;dq=8,1;dkv={dkv}")
        for c in grid:
            r = spawn(a, "token", [65536], {"ROWMASK_HP_CONFIG": c})
            res["configs"][c] = r
            print("config", c, json.dumps(r)[:500], flush=True)
            if a.out:
                json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
