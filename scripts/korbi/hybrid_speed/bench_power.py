"""Power retention (deg 2) kernel options against GDN and flash attention, forward + backward,
at the speed suite's per-GPU shape (8 x 4096 tokens, 768 = H x d wide):

    python scripts/korbi/hybrid_speed/bench_power.py [--d 64 128]

Backends of the retention package: triton (what the power layer uses), vidrial (fused CUDA,
from the image), cuda; chunk sizes from 128 up to the whole sequence (None: the quadratic
attention form). One RESULT line per configuration.
"""

import argparse
import importlib
import json

import torch
import torch.nn.functional as F


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def inputs(b, t, h, d, gating=True):
    mk = lambda: torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q, k, v = mk(), mk(), mk()
    log_g = (F.logsigmoid(6.9 + torch.randn(b, t, h, device="cuda"))).requires_grad_(gating)
    return q, k, v, (log_g if gating else None)


def fwd_bwd(f, *args):
    def run():
        o = f(*args)
        o.backward(torch.ones_like(o))
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--t", type=int, default=4096)
    ap.add_argument("--width", type=int, default=768)
    args = ap.parse_args()
    b, t = args.b, args.t
    for d in args.d:
        h = args.width // d
        q, k, v, log_g = inputs(b, t, h, d)

        def report(name, ms, **extra):
            print("RESULT " + json.dumps(dict(name=name, d=d, h=h, ms=ms,
                                              tokens_per_s=b * t / ms * 1e3, **extra)), flush=True)

        # flash attention (causal) and GDN at the same shape
        sdpa = lambda q, k, v: F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        report("sdpa_causal", bench(fwd_bwd(sdpa, q, k, v)))
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
            qn, kn = F.normalize(q.detach(), dim=-1).requires_grad_(), F.normalize(k.detach(), dim=-1).requires_grad_()
            gdn = lambda q, k, v, g: chunk_gated_delta_rule(q, k, v, g, beta)[0]
            report("gdn", bench(fwd_bwd(gdn, qn, kn, v, log_g)))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL gdn d={d}: {type(e).__name__}: {str(e)[:200]}", flush=True)

        for backend in ("triton", "vidrial", "cuda"):
            try:
                pr = importlib.import_module(f"retention.{backend}").power_retention
            except Exception as e:  # noqa: BLE001
                print(f"FAIL import retention.{backend}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            for chunk in (128, 256, 512, 1024, 2048, None):
                f = lambda q, k, v, g, c=chunk: pr(q, k, v, g, deg=2, scale=d**-0.5, chunk_size=c)
                try:
                    ms = bench(fwd_bwd(f, q, k, v, log_g))
                    report(f"power_{backend}_c{chunk}", ms, backend=backend, chunk=chunk,
                           peak_gib=torch.cuda.max_memory_allocated() / 2**30)
                except Exception as e:  # noqa: BLE001
                    print(f"FAIL power_{backend}_c{chunk} d={d}: {type(e).__name__}: {str(e)[:200]}",
                          flush=True)
                for x in (q, k, v, log_g):
                    x.grad = None
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
