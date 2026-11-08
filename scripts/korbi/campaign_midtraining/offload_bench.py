"""P2 of the midtraining campaign (docs/campaign_3b_hybrid_midtraining.md): can v4 decode with its experts, KV pages
and Engram tables offloaded to host memory on a 24 GB consumer GPU? Microbenchmarks on synthetic tensors of v4's
shapes (no model needed), then a decode-rate estimate that combines them with the routing-trace cache simulation of
spec_gate.py (--gate spec_gate.json).

Measured: device copy bandwidth; pinned host->GPU copies (single / batched expert-sized buffers); copies running
concurrently with compute on another stream (overlap); one W8A16 expert FFN (FP8 / int8 weights dequantized to bf16
+ GLU matmuls, vs bf16-resident weights) at 1 / 16 / 64 tokens; gathering KV pages and Engram rows from pinned host
memory.

    python offload_bench.py --out <dir> [--gate <dir>/spec_gate.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

H, FFN = 2048, 768
DEV = torch.device("cuda")


def timed(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def device_bw():
    a = torch.empty(2**30 // 2, dtype=torch.bfloat16, device=DEV)
    b = torch.empty_like(a)
    t = timed(lambda: b.copy_(a), 10)
    return 2 * a.numel() * 2 / t / 1e9  # read + write


def h2d(sizes_mb=(1, 4, 4.87, 16, 64, 256)):
    out = {}
    for mb in sizes_mb:
        n = int(mb * 2**20)
        src = torch.empty(n, dtype=torch.uint8).pin_memory()
        dst = torch.empty(n, dtype=torch.uint8, device=DEV)
        t = timed(lambda: dst.copy_(src, non_blocking=True), 20)
        out[f"single_{mb}MB"] = n / t / 1e9
    # batched: N expert-sized buffers per "layer" as separate async copies (the prefetch pattern)
    e = int(4.87 * 2**20)
    for nexp in (5, 10, 50):
        srcs = [torch.empty(e, dtype=torch.uint8).pin_memory() for _ in range(nexp)]
        dsts = [torch.empty(e, dtype=torch.uint8, device=DEV) for _ in range(nexp)]

        def go():
            for s, d in zip(srcs, dsts):
                d.copy_(s, non_blocking=True)

        t = timed(go, 10)
        out[f"batched_{nexp}x4.87MB"] = nexp * e / t / 1e9
    return out


def overlap():
    """Host->GPU copies on a side stream while the main stream computes: (a) compute-bound bf16 matmuls,
    (b) memory-bound reads (decode-like). Effective = how much of the copy hides behind the compute."""
    side = torch.cuda.Stream()
    src = torch.empty(256 * 2**20, dtype=torch.uint8).pin_memory()
    dst = torch.empty_like(src, device=DEV)
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=DEV)
    big = torch.empty(2**30 // 2, dtype=torch.bfloat16, device=DEV)
    big2 = torch.empty_like(big)
    res = {}

    def copy():
        with torch.cuda.stream(side):
            dst.copy_(src, non_blocking=True)

    for name, work in (("matmul", lambda: [a @ a for _ in range(8)]), ("memread", lambda: [big2.copy_(big) for _ in range(6)])):
        t_c = timed(copy, 5)
        t_w = timed(work, 5)

        def both():
            copy()
            work()
            torch.cuda.current_stream().wait_stream(side)

        t_b = timed(both, 5)
        res[name] = {"copy_ms": t_c * 1e3, "work_ms": t_w * 1e3, "both_ms": t_b * 1e3,
                     "copy_gbs_alone": src.numel() / t_c / 1e9,
                     "hidden_share": max(0.0, min(1.0, (t_c + t_w - t_b) / min(t_c, t_w)))}
    return res


def ffn():
    """One expert's GLU FFN (W1, W3: 2048 -> 768, W2: 768 -> 2048) at n tokens."""
    res = {}
    w1 = torch.randn(FFN, H, device=DEV, dtype=torch.bfloat16) * 0.02
    w3 = torch.randn(FFN, H, device=DEV, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(H, FFN, device=DEV, dtype=torch.bfloat16) * 0.02
    f8 = [w.to(torch.float8_e4m3fn) for w in (w1, w3, w2)]
    sc = [torch.ones(w.shape[0], w.shape[1] // 128, device=DEV) for w in (w1, w3, w2)]  # 1x128 block scales
    i8 = [w.mul(127 / 0.1).round().clamp(-127, 127).to(torch.int8) for w in (w1, w3, w2)]

    def deq_f8(w, s):
        return (w.to(torch.bfloat16).view(w.shape[0], -1, 128) * s[..., None].to(torch.bfloat16)).view(w.shape)

    def glu(x, a, b, c):
        return (F.silu(x @ a.t()) * (x @ b.t())) @ c.t()

    for n in (1, 16, 64):
        x = torch.randn(n, H, device=DEV, dtype=torch.bfloat16)
        res[f"bf16_n{n}_us"] = timed(lambda: glu(x, w1, w3, w2), 50) * 1e6
        res[f"fp8deq_n{n}_us"] = timed(lambda: glu(x, *(deq_f8(w, s) for w, s in zip(f8, sc))), 50) * 1e6
        res[f"int8deq_n{n}_us"] = timed(lambda: glu(x, *(w.to(torch.bfloat16) * (0.1 / 127) for w in i8)), 50) * 1e6
    # 5 experts of one layer at batch 1 (sequential, as a plain loop would run them)
    x = torch.randn(1, H, device=DEV, dtype=torch.bfloat16)
    res["fp8deq_5experts_b1_us"] = timed(lambda: [glu(x, *(deq_f8(w, s) for w, s in zip(f8, sc))) for _ in range(5)], 30) * 1e6
    res["bf16_5experts_b1_us"] = timed(lambda: [glu(x, w1, w3, w2) for _ in range(5)], 30) * 1e6
    return res


def kv_gather(head_dim=256, pages=32, page=64, kv_heads=2, ctx_tokens=1 << 20):
    """One attention layer's selected KV pages ([K, V] x kv_heads x page tokens x head_dim bf16, 128 KB / page
    at head_dim 256) from a pinned host pool of ctx_tokens: per-page async copies vs a CPU gather into a pinned
    staging buffer + one copy."""
    P = ctx_tokens // page
    page_elems = 2 * kv_heads * page * head_dim
    pool = torch.empty(P, page_elems, dtype=torch.bfloat16).pin_memory()
    staging = torch.empty(pages, page_elems, dtype=torch.bfloat16).pin_memory()
    dst = torch.empty(pages, page_elems, dtype=torch.bfloat16, device=DEV)
    sel = torch.randperm(P)[:pages].sort().values

    def per_page():
        for i, p in enumerate(sel.tolist()):
            dst[i].copy_(pool[p], non_blocking=True)

    def staged():
        torch.index_select(pool, 0, sel, out=staging)
        dst.copy_(staging, non_blocking=True)

    mb = pages * page_elems * 2 / 2**20
    t1, t2 = timed(per_page, 20), timed(staged, 20)
    return {"mb_per_layer": mb, "per_page_copies_us": t1 * 1e6, "per_page_gbs": mb * 2**20 / t1 / 1e9,
            "cpu_gather_then_copy_us": t2 * 1e6, "cpu_gather_gbs": mb * 2**20 / t2 / 1e9}


def engram_gather(rows=4 * 2**20, dim=64, lookups=48):
    """Engram rows (64 bf16 each) of one token: 2 Engram layers x 24 lookups, from a pinned host table."""
    table = torch.empty(rows, dim, dtype=torch.bfloat16).pin_memory()
    staging = torch.empty(lookups, dim, dtype=torch.bfloat16).pin_memory()
    dst = torch.empty(lookups, dim, dtype=torch.bfloat16, device=DEV)
    idx = torch.randint(0, rows, (lookups,))

    def go():
        torch.index_select(table, 0, idx, out=staging)
        dst.copy_(staging, non_blocking=True)

    return {"lookups": lookups, "us_per_token": timed(go, 50) * 1e6}


def estimate(meas, gate, out):
    """Decode tokens / s at batch 1 and 16 for 64k / 1M / 10M context. Per step: GPU time (resident weights +
    the step's experts read at device bandwidth, + compute) overlapped with PCIe time (expert misses with the
    prefetch factor + KV pages beyond what stays resident); the slower one bounds the step [est.]."""
    dev = meas["device_gbs"]
    pcie = meas["h2d"]["batched_10x4.87MB"]
    hide = meas["overlap"]["memread"]["hidden_share"]
    kv_layer_mb = meas["kv_gather"]["mb_per_layer"]
    kv_gbs = max(meas["kv_gather"]["per_page_gbs"], meas["kv_gather"]["cpu_gather_gbs"])
    exp_mb = 4.87
    resident_gb = 2.6  # FP8 trunk + shared experts + bf16 head, read every step [est.]
    rows = {}
    sims = gate["sims"]
    pick = {1: [k for k in sims if k.endswith("|w1") and "_b16" not in k],
            16: [k for k in sims if k.endswith("|w1") and "_b16" in k]}
    for bs, keys in pick.items():
        for cap in gate["caps"]:
            union = sum(sims[k]["union"] for k in keys) / len(keys)
            miss = sum(sims[k][f"miss@{cap}"] for k in keys) / len(keys)
            for ctx, kv_offload in (("64k", False), ("1M", True), ("10M", True)):
                gpu_ms = (resident_gb * 1024 + union * exp_mb) / (dev * 1024) * 1e3 + 0.02 * bs
                exp_ms = miss * 2.0 * exp_mb / (pcie * 1024) * 1e3  # x2: top-10 prefetch false positives
                kv_ms = (6 * kv_layer_mb * bs / (kv_gbs * 1024) * 1e3) if kv_offload else 0.0
                pcie_ms = exp_ms + kv_ms
                step = max(gpu_ms, pcie_ms) + (1 - hide) * min(gpu_ms, pcie_ms)
                rows[f"b{bs}|cap{cap}|{ctx}"] = {"gpu_ms": gpu_ms, "expert_pcie_ms": exp_ms, "kv_pcie_ms": kv_ms,
                                                 "step_ms": step, "tok_s": bs * 1e3 / step, "union": union,
                                                 "miss": miss}
    (out / "estimate.json").write_text(json.dumps(rows, indent=1))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gate", type=Path, help="spec_gate.json (cache simulation) for the decode estimate")
    p.add_argument("--skip-bench", action="store_true", help="reuse <out>/bench.json")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.skip_bench:
        meas = json.loads((args.out / "bench.json").read_text())
    else:
        meas = {"gpu": torch.cuda.get_device_name()}
        meas["device_gbs"] = device_bw()
        print("device", meas["device_gbs"], flush=True)
        meas["h2d"] = h2d()
        print("h2d", meas["h2d"], flush=True)
        meas["overlap"] = overlap()
        print("overlap", meas["overlap"], flush=True)
        meas["ffn"] = ffn()
        print("ffn", meas["ffn"], flush=True)
        meas["kv_gather"] = kv_gather()
        print("kv", meas["kv_gather"], flush=True)
        meas["engram"] = engram_gather()
        print("engram", meas["engram"], flush=True)
        (args.out / "bench.json").write_text(json.dumps(meas, indent=1))
    if args.gate:
        rows = estimate(meas, json.loads(args.gate.read_text()), args.out)
        for k, r in rows.items():
            print(k, {a: round(b, 2) for a, b in r.items()}, flush=True)


if __name__ == "__main__":
    main()
