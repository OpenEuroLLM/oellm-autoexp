"""Microbenchmark: Primus-Turbo grouped GEMM vs sequential GEMM for Qwen3.5-35B-A3B MoE experts.

Expert shapes (per-EP-rank, EP=8, 256 total experts → 32 local):
  gate+up (fused SwiGLU): [tokens_per_expert, 2048] x [32, 2048, 1024]  (hidden→2*ffn)
  down:                   [tokens_per_expert, 512]  x [32, 512, 2048]    (ffn→hidden)

Run inside the container on a compute node:
  srun -p amd-tw-verification -N1 --ntasks=1 --gpus=1 --time=10:00 \
    singularity exec --rocm /shared_silo/scratch/containers/primus_v26.1.sif \
    python scripts/bench_primus_grouped_gemm.py
"""

import time
import torch

WARMUP = 20
ITERS = 100
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# Qwen3.5-35B-A3B: 256 experts, EP=8 → 32 local experts
NUM_LOCAL_EXPERTS = 32
HIDDEN = 2048
FFN = 512  # moe_ffn_hidden_size

# Simulate realistic token distributions for global_batch_size=1024, seq_len=8192, topk=8
# Total routed tokens per EP rank ≈ (1024 * 8192 * 8) / 8 = 8M, but per microbatch (mbs=2):
# tokens_per_rank ≈ (2 * 8192 * 8) / 8 = 16384
TOTAL_TOKENS = 16384


def make_token_distribution(total_tokens, num_experts, mode="uniform"):
    if mode == "uniform":
        base = total_tokens // num_experts
        remainder = total_tokens % num_experts
        lens = [base + (1 if i < remainder else 0) for i in range(num_experts)]
    elif mode == "skewed":
        # Zipf-like: some experts get many more tokens
        import numpy as np
        rng = np.random.default_rng(42)
        raw = rng.zipf(1.5, num_experts).astype(float)
        raw = raw / raw.sum() * total_tokens
        lens = [max(1, int(x)) for x in raw]
        diff = total_tokens - sum(lens)
        lens[0] += diff
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return torch.tensor(lens, dtype=torch.long, device=DEVICE)


def bench_sequential(x, weights, group_lens, trans_b, label):
    """Sequential per-expert GEMM (what Megatron does with moe_grouped_gemm=false)."""
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        offset = 0
        outputs = []
        for i in range(weights.shape[0]):
            n = group_lens[i].item()
            if n > 0:
                xi = x[offset:offset + n]
                wi = weights[i]
                if trans_b:
                    outputs.append(xi @ wi.T)
                else:
                    outputs.append(xi @ wi)
                offset += n
        if outputs:
            _ = torch.cat(outputs, dim=0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        offset = 0
        outputs = []
        for i in range(weights.shape[0]):
            n = group_lens[i].item()
            if n > 0:
                xi = x[offset:offset + n]
                wi = weights[i]
                if trans_b:
                    outputs.append(xi @ wi.T)
                else:
                    outputs.append(xi @ wi)
                offset += n
        if outputs:
            out = torch.cat(outputs, dim=0)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / ITERS * 1000
    print(f"  {label:30s}: {elapsed:8.3f} ms/iter")
    return elapsed


def bench_primus_grouped(x, weights, group_lens, trans_b, label):
    """Primus-Turbo grouped GEMM."""
    import primus_turbo.pytorch as turbo

    torch.cuda.synchronize()
    for _ in range(WARMUP):
        _ = turbo.ops.grouped_gemm(x, weights, group_lens, trans_b=trans_b)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = turbo.ops.grouped_gemm(x, weights, group_lens, trans_b=trans_b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / ITERS * 1000
    print(f"  {label:30s}: {elapsed:8.3f} ms/iter")
    return elapsed


def bench_torch_grouped_mm(x, weights_2d, group_lens, label):
    """torch._grouped_mm (if available in this PyTorch build)."""
    if not hasattr(torch, "_grouped_mm"):
        print(f"  {label:30s}: torch._grouped_mm not available")
        return None

    # torch._grouped_mm expects offsets, not lengths
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        group_lens.to(torch.int32).cumsum(0)
    ])

    torch.cuda.synchronize()
    for _ in range(WARMUP):
        _ = torch._grouped_mm(x, weights_2d, offs=offsets)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = torch._grouped_mm(x, weights_2d, offs=offsets)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / ITERS * 1000
    print(f"  {label:30s}: {elapsed:8.3f} ms/iter")
    return elapsed


def run_benchmark(name, M, K, N, num_experts, dist_mode):
    print(f"\n{'='*70}")
    print(f"{name} | M={M} K={K} N={N} experts={num_experts} dist={dist_mode}")
    print(f"{'='*70}")

    group_lens = make_token_distribution(M, num_experts, mode=dist_mode)
    assert group_lens.sum().item() == M

    x = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
    # weights layout: [num_experts, N, K] for trans_b=True (standard expert weight layout)
    weights = torch.randn(num_experts, N, K, dtype=DTYPE, device=DEVICE)

    t_seq = bench_sequential(x, weights, group_lens, trans_b=True, label="sequential (per-expert)")
    t_primus = bench_primus_grouped(x, weights, group_lens, trans_b=True, label="primus_turbo grouped_gemm")

    # Also try torch._grouped_mm if available
    # It needs weights as [G, K, N] (no transpose)
    weights_notrans = weights.transpose(-1, -2).contiguous()
    # torch._grouped_mm: expects 2D a and list of 2D b, or special format
    # Actually it uses a different API — skip if complex

    speedup = t_seq / t_primus if t_primus > 0 else 0
    print(f"  {'speedup':30s}: {speedup:.2f}x")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {DTYPE}, warmup: {WARMUP}, iters: {ITERS}")

    try:
        import primus_turbo
        print(f"primus_turbo: {primus_turbo.__version__}")
    except Exception as e:
        print(f"primus_turbo import failed: {e}")
        return

    # Gate+Up projection: [tokens, hidden] x [experts, 2*ffn, hidden]^T → [tokens, 2*ffn]
    for dist in ["uniform", "skewed"]:
        run_benchmark(
            "gate+up (SwiGLU fused)",
            M=TOTAL_TOKENS, K=HIDDEN, N=2 * FFN,
            num_experts=NUM_LOCAL_EXPERTS, dist_mode=dist,
        )

    # Down projection: [tokens, ffn] x [experts, hidden, ffn]^T → [tokens, hidden]
    for dist in ["uniform", "skewed"]:
        run_benchmark(
            "down projection",
            M=TOTAL_TOKENS, K=FFN, N=HIDDEN,
            num_experts=NUM_LOCAL_EXPERTS, dist_mode=dist,
        )

    # Also test with fewer tokens (early training, some experts get very few)
    for total in [4096, 8192]:
        run_benchmark(
            f"gate+up (fewer tokens={total})",
            M=total, K=HIDDEN, N=2 * FFN,
            num_experts=NUM_LOCAL_EXPERTS, dist_mode="uniform",
        )


if __name__ == "__main__":
    main()
