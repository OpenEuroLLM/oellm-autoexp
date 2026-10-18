"""Why is fp8 slower than bf16 for the 3B-active MoE hybrid (docs/hybrid_moe_3b_campaign.md §11.1)?

    python scripts/korbi/hybrid_speed/fp8_bench.py   # on one H100, inside the training image

Transformer Engine layers at the A3B-30B/80B shapes (hidden 2048, 8192 tokens per micro-batch =
mbs 2 x 4096): fwd + bwd time in bf16 and in fp8 (tensorwise = Float8CurrentScaling, the
32B-dense production recipe; and delayed scaling for comparison), with the weights cast every
call (is_first_microbatch=None) or cast once and cached (is_first_microbatch=False, the
gradient-accumulation steady state), plus a kernel breakdown of each fp8 case (GEMM vs cast /
amax / transpose kernels).
"""

import time

import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format

DT = torch.bfloat16
T, H = 8192, 2048


def autocast(recipe):
    if recipe is None:
        return torch.autocast("cuda", enabled=False)
    fn = getattr(te, "fp8_autocast", None) or te.autocast
    return fn(enabled=True, fp8_recipe=recipe)


def bench(step, it=30, warm=8):
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(it):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3


def linear_case(n_in, n_out, recipe, first_mb):
    layer = te.Linear(n_in, n_out, bias=False, params_dtype=DT).cuda()
    x = torch.randn(T, n_in, device="cuda", dtype=DT, requires_grad=True)
    g = torch.randn(T, n_out, device="cuda", dtype=DT)

    def step():
        with autocast(recipe):
            y = layer(x, is_first_microbatch=first_mb) if recipe is not None else layer(x)
        y.backward(g)

    return step


def grouped_case(num, n_in, n_out, tokens_per_expert, recipe, first_mb):
    layer = te.GroupedLinear(num, n_in, n_out, bias=False, params_dtype=DT).cuda()
    m = [tokens_per_expert] * num
    x = torch.randn(sum(m), n_in, device="cuda", dtype=DT, requires_grad=True)
    g = torch.randn(sum(m), n_out, device="cuda", dtype=DT)

    def step():
        with autocast(recipe):
            y = layer(x, m, is_first_microbatch=first_mb) if recipe is not None else layer(x, m)
        y.backward(g)

    return step


def breakdown(step, label):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            step()
        torch.cuda.synchronize()
    rows = [(e.key, e.self_device_time_total / 10) for e in p.key_averages()
            if getattr(e, "self_device_time_total", 0) > 0]
    rows.sort(key=lambda r: -r[1])
    total = sum(t for _, t in rows)
    gemm = sum(t for k, t in rows if any(s in k.lower() for s in ("gemm", "sm90", "cutlass", "nvjet", "xmma")))
    print(f"    [{label}] kernels per step: total {total / 1e3:.2f} ms, GEMM {gemm / 1e3:.2f} ms "
          f"({100 * gemm / max(total, 1):.0f} %), other {(total - gemm) / 1e3:.2f} ms")
    for k, t in rows[:6]:
        print(f"      {t / 1e3:7.3f} ms  {k[:90]}")


def main():
    print(f"torch {torch.__version__}, TE {transformer_engine.__version__}, {torch.cuda.get_device_name()}")
    recipes = {
        "bf16": None,
        "fp8 tensorwise": Float8CurrentScaling(fp8_format=Format.HYBRID),
        "fp8 delayed": DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16,
                                      amax_compute_algo="max"),
    }
    cases = {
        "attention q + GQA kv proj 2048->2560": lambda r, f: linear_case(H, 2560, r, f),
        "CKDA-style proj 2048->6144": lambda r, f: linear_case(H, 6144, r, f),
        "shared expert fc1 2048->1536": lambda r, f: linear_case(H, 1536, r, f),
        "shared expert fc2 768->2048": lambda r, f: linear_case(768, H, r, f),
        # fp8 GEMMs need 16-aligned dimensions: tokens per expert padded to 176 (~170 real)
        "MoE fc1: 84 experts x 2048->1536, 176 tok each": lambda r, f: grouped_case(84, H, 1536, 176, r, f),
        "MoE fc2: 84 experts x 768->2048, 176 tok each": lambda r, f: grouped_case(84, 768, H, 176, r, f),
        "MoE fc1, 4x the tokens (688 / expert)": lambda r, f: grouped_case(84, H, 1536, 688, r, f),
        "big dense 5120->13824 (32B-dense-like)": lambda r, f: linear_case(5120, 13824, r, f),
    }
    import sys

    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for cname, make in cases.items():
        if only and only not in cname:
            continue
        print(f"\n== {cname}")
        base = bench(make(None, None))
        print(f"  bf16                           {base:7.3f} ms")
        for rname, rec in recipes.items():
            if rec is None:
                continue
            for first_mb, tag in ((None, "weights cast every call"), (False, "weights cached")):
                try:
                    ms = bench(make(rec, first_mb))
                    print(f"  {rname:15s} {tag:24s} {ms:7.3f} ms  ({base / ms:4.2f}x vs bf16)")
                except Exception as e:  # noqa: BLE001
                    print(f"  {rname:15s} {tag:24s} FAILED {type(e).__name__}: {str(e)[:100]}")
        try:
            breakdown(make(recipes["fp8 tensorwise"], None), "fp8 tensorwise, cast every call")
            breakdown(make(None, None), "bf16")
        except Exception as e:  # noqa: BLE001
            print(f"  breakdown failed: {e}")


if __name__ == "__main__":
    main()
