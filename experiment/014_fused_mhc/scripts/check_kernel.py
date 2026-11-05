"""014: the fused mHC write + read kernel (oellm_kernels.mhc_write_read) against mhc_write followed by
mhc_read, on random streams of realistic scale; then the time of both on a few token counts.

    CC=/usr/bin/gcc PYTHONPATH=. python experiment/014_fused_mhc/scripts/check_kernel.py
"""

import time

import torch

from oellm_autoexp.hf_export.oellm_ambitious import oellm_kernels as k


def params(h, dev, g):
    phi = torch.randn(4 * h, 24, generator=g, device="cpu").mul_(0.02).to(dev)
    return dict(phi=phi, alpha_pre=torch.tensor([1.0], device=dev), alpha_post=torch.tensor([1.0], device=dev),
                alpha_res=torch.tensor([1.0], device=dev), b_pre=torch.randn(4, generator=g).to(dev),
                b_post=torch.randn(4, generator=g).to(dev), b_res=torch.randn(4, 4, generator=g).to(dev))


def run(tokens, h=2048, norm=True, dev="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(tokens, 4, h, generator=g).to(dev, torch.bfloat16)
    out = torch.randn(tokens, h, generator=g).to(dev, torch.bfloat16)
    p0, p1 = params(h, dev, g), params(h, dev, g)
    nw = (torch.randn(h, generator=g).mul_(0.1).add_(1.0).to(dev, torch.bfloat16), 1e-5, False) if norm else None
    _, pend = k.mhc_read(s, p0["phi"], p0["alpha_pre"], p0["alpha_post"], p0["alpha_res"], p0["b_pre"],
                         p0["b_post"], p0["b_res"], 1e-6, 20)
    y_ref = k.mhc_write(s, out, pend)
    x_ref, (post_ref, res_ref) = k.mhc_read(y_ref, p1["phi"], p1["alpha_pre"], p1["alpha_post"], p1["alpha_res"],
                                            p1["b_pre"], p1["b_post"], p1["b_res"], 1e-6, 20, nw)
    phi_pad = k.mhc_phi_padded(p1["phi"], torch.bfloat16)
    y, x, (post, res) = k.mhc_write_read(s, out, pend, phi_pad, p1["alpha_pre"], p1["alpha_post"],
                                         p1["alpha_res"], p1["b_pre"], p1["b_post"], p1["b_res"], 1e-6, 20, nw)
    torch.cuda.synchronize()
    rep = {
        "tokens": tokens, "norm": norm,
        "streams_equal": bool(torch.equal(y, y_ref)),
        "x_maxdiff": float((x.float() - x_ref.float()).abs().max()),
        "x_mismatch_frac": float((x != x_ref).float().mean()),
        "post_maxdiff": float((post - post_ref).abs().max()),
        "res_maxdiff": float((res - res_ref).abs().max()),
    }

    def bench(fn, n=50):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6

    rep["us_unfused"] = bench(lambda: k.mhc_read(k.mhc_write(s, out, pend), p1["phi"], p1["alpha_pre"],
                                                 p1["alpha_post"], p1["alpha_res"], p1["b_pre"], p1["b_post"],
                                                 p1["b_res"], 1e-6, 20, nw))
    rep["us_fused"] = bench(lambda: k.mhc_write_read(s, out, pend, phi_pad, p1["alpha_pre"], p1["alpha_post"],
                                                     p1["alpha_res"], p1["b_pre"], p1["b_post"], p1["b_res"],
                                                     1e-6, 20, nw))
    return rep


if __name__ == "__main__":
    for tokens, norm in ((1, True), (5, True), (16, False), (16, True), (37, True), (1024, True), (8192, True)):
        print(run(tokens, norm=norm), flush=True)
