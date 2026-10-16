"""The KDA-style low-rank gate projections (f_proj / g_proj: hidden -> r -> H*D) plus beta,
forward + backward incl. weight gradients, at the speed suite's per-GPU token count:

    python scripts/korbi/hybrid_speed/bench_lowrank.py

  current  one GEMM to [f_low | beta | g_low] (2r + H wide), then the two r -> H*D GEMMs
  padded   the same with the first GEMM's output padded to a multiple of 64
  merged   W_f1 W_f0 and W_g1 W_g0 formed per step (H*D x hidden each), one wide GEMM
Prints the per-variant time and the top kernels of each.
"""

import json

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

N, HID, R, H, D = 8 * 4096, 768, 128, 12, 64
KD = H * D
dev, dt = "cuda", torch.bfloat16


def params():
    mk = lambda *s: (torch.randn(*s, device=dev, dtype=dt) * 0.02).requires_grad_()
    return dict(f0=mk(R, HID), f1=mk(KD, R), b=mk(H, HID), g0=mk(R, HID), g1=mk(KD, R),
                g1_bias=torch.zeros(KD, device=dev, dtype=dt, requires_grad=True))


def current(x, p, pad=False):
    w = [p["f0"], p["b"], p["g0"]]
    sizes = [R, H, R]
    extra = -sum(sizes) % 64 if pad else 0
    if extra:
        w.append(p["f0"].new_zeros(extra, HID))
    y = F.linear(x, torch.cat(w))
    f_low, beta, g_low = torch.split(y, sizes + ([extra] if extra else []), dim=-1)[:3]
    return F.linear(f_low, p["f1"]), beta, F.linear(g_low, p["g1"], p["g1_bias"])


def merged(x, p):
    wf, wg = p["f1"] @ p["f0"], p["g1"] @ p["g0"]
    sizes = [KD, KD, H]
    extra = -sum(sizes) % 64
    w = [wf, wg, p["b"]] + ([wf.new_zeros(extra, HID)] if extra else [])
    y = F.linear(x, torch.cat(w))
    f, g, beta = torch.split(y, sizes + ([extra] if extra else []), dim=-1)[:3]
    return f, beta, g + p["g1_bias"]


def step(fn, x, p):
    f, beta, g = fn(x, p)
    (f.float().sum() + beta.float().sum() + g.float().sum()).backward()


def main():
    x = torch.randn(N, HID, device=dev, dtype=dt, requires_grad=True)
    p = params()
    variants = {"current": current, "padded": lambda x, p: current(x, p, pad=True), "merged": merged}
    for name, fn in variants.items():
        for _ in range(5):
            step(fn, x, p)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(20):
            step(fn, x, p)
        e.record()
        torch.cuda.synchronize()
        print("RESULT " + json.dumps(dict(name=name, ms=s.elapsed_time(e) / 20)), flush=True)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                step(fn, x, p)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8, max_name_column_width=90))


if __name__ == "__main__":
    main()
