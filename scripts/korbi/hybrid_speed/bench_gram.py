"""Triadic KDA's Triton glue kernels (cute_kernels, cute_triadic_gdn.ops.kda_triton) at the speed
suite's tri_kda shape, against flash-linear-attention's KDA intra-chunk kernels:

    python scripts/korbi/hybrid_speed/bench_gram.py [--b 8 --t 4096 --h 6 --d 128]

Prints one RESULT line per kernel (ms, median of CUDA-event timings) and checks every Gram forward
variant against the reference product.
"""

import argparse
import json

import torch
import triton.testing

from cute_triadic_gdn.ops import kda_triton as kt


def reference_gram(qs, ks, gd):
    """Pkk (strictly lower) and Pq (lower), (B, T, H, 64) fp32, by the pairwise exponentials."""
    B, T, H, D = qs.shape
    c = kt.C
    q = qs.float().view(B, T // c, c, H, D)
    k = ks.float().view(B, T // c, c, H, D)
    g = gd.view(B, T // c, c, H, D)
    e = torch.exp((g[:, :, :, None] - g[:, :, None, :]).clamp(max=0))  # [B, N, t, s, H, D]
    pq = (q[:, :, :, None] * k[:, :, None, :] * e).sum(-1)
    pk = (k[:, :, :, None] * k[:, :, None, :] * e).sum(-1)
    t = torch.arange(c, device=qs.device)
    pq = pq * (t[:, None] >= t[None, :])[None, None, :, :, None]
    pk = pk * (t[:, None] > t[None, :])[None, None, :, :, None]
    return pk.permute(0, 1, 2, 4, 3).reshape(B, T, H, c), pq.permute(0, 1, 2, 4, 3).reshape(B, T, H, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--t", type=int, default=4096)
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    shape = (args.b, args.t, args.h, args.d)
    q = torch.nn.functional.normalize(torch.randn(shape, device=dev), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(shape, device=dev), dim=-1).bfloat16()
    g = -torch.nn.functional.softplus(torch.randn(shape, device=dev) - 2.0)
    qs, ks, _, _, _, gd, _ = kt.prep(q, k, g)

    # correctness on a slice (the reference is memory hungry)
    sl = (slice(0, 1), slice(0, 512))
    ref_pk, ref_pq = reference_gram(qs[sl], ks[sl], gd[sl].contiguous())
    variants = {"gram_fwd": kt.gram_fwd}
    if hasattr(kt, "gram_fwd_triton"):
        variants["gram_fwd_triton"] = kt.gram_fwd_triton
    for name, fn in variants.items():
        pk, pq = fn(qs[sl].contiguous(), ks[sl].contiguous(), gd[sl].contiguous())
        err = max((pk - ref_pk).abs().max().item(), (pq - ref_pq).abs().max().item())
        print(f"CHECK {name} max abs err {err:.2e}")

    dpk = torch.randn(args.b, args.t, args.h, kt.C, device=dev)
    dpq = torch.randn_like(dpk)
    bench = {
        "prep": lambda: kt.prep(q, k, g),
        **{name: (lambda fn=fn: fn(qs, ks, gd)) for name, fn in variants.items()},
        "gram_bwd": lambda: kt.gram_bwd(qs, ks, gd, dpk, dpq),
        "gram_bwd_triton": lambda: kt.gram_bwd_triton(qs, ks, gd, dpk, dpq),
    }
    try:
        from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra

        beta = torch.ones(shape[:3], device=dev)
        bench["fla_kda_fwd_intra"] = lambda: chunk_kda_fwd_intra(
            q=qs, k=ks, v=ks, gk=gd * kt._LOG2E, beta=beta, scale=1.0, chunk_size=kt.C
        )
    except ImportError:
        pass
    for name, fn in bench.items():
        ms = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
        print("RESULT " + json.dumps({"name": name, "ms": round(ms, 4), "shape": list(shape)}))


if __name__ == "__main__":
    main()
