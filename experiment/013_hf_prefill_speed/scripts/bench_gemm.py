"""The routed experts' GEMMs at the v4 shapes (360 experts, top-5, h 2048, i 768): the FP8 Triton
grouped GEMM with the operand order forced (SWAP 0 / 1, each autotuned) vs torch._grouped_mm on
BF16-expanded weights (008), per token count. TFLOP/s of GEMM1 (N 1536, K 2048) + GEMM2 (N 2048,
K 768).

    python bench_gemm.py --out gemm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "oellm_autoexp/hf_export/oellm_ambitious"))
import oellm_kernels as K  # noqa: E402

E, TOPK, H, I = 360, 5, 2048, 768


def timed(f, n=10):
    f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        f()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def quant(shape):
    w = torch.randn(shape, device="cuda") * shape[-1] ** -0.5
    wf = w.view(*shape[:-1], -1, 128)
    sc = wf.abs().amax(-1) / 448
    q = (wf / sc[..., None]).to(torch.float8_e4m3fn).view(shape)
    deq = (q.float().view(*shape[:-1], -1, 128) * sc[..., None]).view(shape).bfloat16()
    return q.view(torch.uint8), sc, deq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", nargs="+", type=int, default=[512, 2048, 8192, 16384, 32768])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    gu_q, gu_s, gu_d = quant((E, 2 * I, H))
    dn_q, dn_s, dn_d = quant((E, H, I))
    all_cfg = list(K._grouped_gemm_kernel.configs)
    rows = []
    for t in a.tokens:
        x = torch.randn(t, H, device="cuda").bfloat16()
        idx = torch.randint(0, E, (t, TOPK), device="cuda", dtype=torch.int32)
        sched = K.ExpertSchedule(idx, E, K.block_m_for(idx.numel(), E))
        act = torch.randn(t * TOPK, I, device="cuda").bfloat16()
        flops = t * TOPK * (2 * I * H + H * I) * 2
        row = {"tokens": t, "block_m": sched.block_m}
        for sw in (0, 1):
            K._grouped_gemm_kernel.configs = [c for c in all_cfg if c.kwargs["SWAP"] == sw]
            K._grouped_gemm_kernel.cache.clear()
            ms = timed(lambda: (K.grouped_gemm(x, sched, gu_q, gu_s, True), K.grouped_gemm(act, sched, dn_q, dn_s, False)))
            row[f"fp8_swap{sw}_ms"] = ms
            row[f"fp8_swap{sw}_tflops"] = flops / ms / 1e9
            row[f"fp8_swap{sw}_best"] = str(K._grouped_gemm_kernel.best_config)
        K._grouped_gemm_kernel.configs = all_cfg
        K._grouped_gemm_kernel.cache.clear()
        flat = idx.flatten()
        offs = torch.zeros(E, dtype=torch.int64, device="cuda").scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.int64)).cumsum(0).to(torch.int32)
        xs = x[sched.tokens.long()]
        ms = timed(lambda: (torch._grouped_mm(xs, gu_d.transpose(-1, -2), offs=offs),
                            torch._grouped_mm(act, dn_d.transpose(-1, -2), offs=offs)))
        row["bf16_grouped_mm_ms"] = ms
        row["bf16_grouped_mm_tflops"] = flops / ms / 1e9
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
