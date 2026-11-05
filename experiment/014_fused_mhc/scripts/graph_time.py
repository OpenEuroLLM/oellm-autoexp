"""014: GPU time of the unfused mHC write + read (3 kernels: write, coefficient GEMM, read) vs the fused kernel,
each captured in a CUDA graph (no Python / launch overhead), and which coefficient rounding is closer to an
fp64 reference (cuBLAS's bf16 GEMM output vs the fused kernel's dot).

    CC=/usr/bin/gcc PYTHONPATH=. python experiment/014_fused_mhc/scripts/graph_time.py
"""

import torch

from oellm_autoexp.hf_export.oellm_ambitious import oellm_kernels as k

import check_kernel as ck  # noqa: E402


def graph_us(fn, reps=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / reps / 10


def main():
    dev = "cuda"
    h = 2048
    for tokens in (1, 4, 16, 64, 256, 1024, 8192):
        g = torch.Generator().manual_seed(0)
        s = torch.randn(tokens, 4, h, generator=g).to(dev, torch.bfloat16)
        out = torch.randn(tokens, h, generator=g).to(dev, torch.bfloat16)
        p0, p1 = ck.params(h, dev, g), ck.params(h, dev, g)
        nw = (torch.ones(h, device=dev, dtype=torch.bfloat16), 1e-5, False)
        _, pend = k.mhc_read(s, p0["phi"], p0["alpha_pre"], p0["alpha_post"], p0["alpha_res"], p0["b_pre"],
                             p0["b_post"], p0["b_res"], 1e-6, 20)
        phi_pad = k.mhc_phi_padded(p1["phi"], torch.bfloat16)
        args1 = (p1["alpha_pre"], p1["alpha_post"], p1["alpha_res"], p1["b_pre"], p1["b_post"], p1["b_res"], 1e-6, 20)
        un = graph_us(lambda: k.mhc_read(k.mhc_write(s, out, pend), p1["phi"], *args1, nw))
        fu = graph_us(lambda: k.mhc_write_read(s, out, pend, phi_pad, *args1, nw))
        # coefficient rounding vs fp64
        y = k.mhc_write(s, out, pend)
        c64 = (y.view(tokens, 4 * h).double() @ p1["phi"].to(torch.bfloat16).double())
        c_blas = (y.view(tokens, 4 * h) @ p1["phi"].to(torch.bfloat16)).double()
        exact = c64.to(torch.bfloat16).double()
        print(f"tokens {tokens:5d}: unfused {un:7.1f} us  fused {fu:7.1f} us  "
              f"cuBLAS coeff != round(fp64): {float((c_blas != exact).double().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
