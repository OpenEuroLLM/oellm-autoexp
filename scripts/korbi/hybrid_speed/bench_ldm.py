"""Linear Depth Memory per-site cost at the production shape (h 2048, G 16, K 8, D 128; 8k / 16k
tokens per micro-batch): a chain of E sites (each the fused step of depth_memory.py: address
networks + write + next read), forward + backward, per site; the torch workspace and the CuTe one
(cute_kernels cute_ldm), plus the bytes the state occupies per token.

    python scripts/korbi/hybrid_speed/bench_ldm.py   (one GPU; PYTHONPATH as the training runs)
"""

import statistics
from types import SimpleNamespace

import torch

from megatron.core.transformer.depth_memory import DepthMemoryRead, DepthMemoryWrite, ldm_fused_step
from megatron.core.transformer.depth_memory_state import make_workspace

H, G, K, E = 2048, 16, 8, 8


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def config(impl):
    return SimpleNamespace(
        hidden_size=H, depth_memory_groups=G, depth_memory_key_dim=K, depth_memory_hidden_dim=32,
        depth_memory_lambda_init=0.99, depth_memory_alpha=1.0, depth_memory_impl=impl,
        depth_memory_window=4, depth_memory_covariance_dtype="float32", layernorm_epsilon=1e-6,
        init_method_std=0.02,
    )


D = H // G
state = (G * K * D + G * K * K) * 4
tape = (G * K * K + G * K * 4 + G * D + G + G) * 4
print(f"state (M, P) fp32 per token: {state / 1024:.1f} KB  (mHC-4 streams bf16: {4 * H * 2 / 1024:.1f} KB);"
      f" torch tape per token and site: {tape / 1024:.1f} KB", flush=True)

for impl in ("torch", "cute"):
    try:
        cfg = config(impl)
        writes = [DepthMemoryWrite(cfg).cuda() for _ in range(E)]
        reads = [DepthMemoryRead(cfg).cuda() for _ in range(E)]
        ws = make_workspace(cfg, E)
    except Exception as exc:  # pragma: no cover
        print(f"{impl}: unavailable ({exc!r})", flush=True)
        continue
    for T in (8192, 16384):
        x0 = torch.randn(T, 1, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        outs = [torch.randn(T, 1, H, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(E)]

        def chain(backward):
            s = x0
            total = 0
            for e in range(E):
                v = s if e == 0 else outs[e]
                s = s if e == 0 else s + outs[e]
                r = ldm_fused_step(writes[e], reads[e], ws, e, s, v)
                total = total + r.float().sum() * 1e-3
            if backward:
                total.backward()

        with torch.no_grad():
            t_f = timed(lambda: chain(False))
        t_fb = timed(lambda: chain(True))
        print(f"{impl:5s} T={T}: per site fwd {t_f / E:.3f} ms, fwd+bwd {t_fb / E:.3f} ms; workspace "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak", flush=True)
        torch.cuda.reset_peak_memory_stats()
