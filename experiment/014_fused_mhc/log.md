# 014: an exact fused mHC write + read

## Objective

Fuse sublayer l-1's mHC write with sublayer l's mHC read (coefficient GEMM, Sinkhorn, input mix, input RMSNorm)
**without** DeepSeek-V4.1's Single-Pass shift (A_{l-1} for block l), which would change the trained function of
v4. Context: docs/campaign_3b_hybrid_midtraining.md §7 (2026-11-05).

The current HF fast path costs 3-4 launches and ~18 h of activation traffic per sublayer:
- the bf16 cuBLAS coefficient GEMM over the streams (4 h);
- `_mhc_read_kernel` (4 h + 1 h);
- `_mhc_write_kernel` (5 h + 4 h).

## Kernels (oellm_autoexp/hf_export/oellm_ambitious/oellm_kernels.py: `mhc_write_read`, not wired into the model)

1. `_mhc_wr_write_kernel`, over token tiles × hidden tiles (16 × 64):
   - the write, with mhc_write's rounding points;
   - per hidden tile, the partial coefficient GEMM (`tl.dot` against phi padded to 32 columns) and the partial
     Gram matrix of the new streams.
2. `_mhc_wr_coeff_kernel`, over token tiles:
   - the partials summed in a fixed order (deterministic, no atomics);
   - coefficients with mhc_read's rounding points; the norm is the trace of the Gram matrix;
   - the input RMSNorm's sum of squares as Σ p_i p_j G_ij;
   - STEP 1 = h_pre / h_post / 1/rms (the critical path); STEP 2 = the Sinkhorn h_res, which only the next write
     needs (`split=True` puts it on a side stream, like vLLM's `mhc_split`).
3. `_mhc_wr_mix_kernel`, over token tiles × hidden tiles: the input mix from the new streams, which are re-read
   from L2, plus the RMSNorm.

## Results (local RTX 3090 Ti, random streams of realistic scale, h = 2048; scripts/check_kernel.py, prof_kernels.py, graph_time.py)

**Parity vs mhc_write → mhc_read:**
- The streams are bitwise equal.
- x differs by ≤ 1-2 bf16 ulp in 1-4 % of elements. This comes from the Gram-based input norm.
- At 16-256 tokens, h_pre / h_post / h_res also differ by up to ~0.004–0.006. Here the **reference** is the less
  accurate one: cuBLAS picks a split-K GEMM with **bf16 partial sums** (`splitKreduce_kernel<…, __nv_bfloat16>`),
  so 40-42 % of its bf16 coefficients differ from correctly rounded fp64. At 1, 4 and 1024 tokens the coefficients
  are 0.000 off and at 8192 tokens 0.004.
- Also from the reference: the cuBLAS coefficient GEMM takes 59 µs at 4 tokens (18 µs at 1 and 16; CUDA-graph
  timing).

**Per-kernel GPU time (torch.profiler, µs):**

| tokens | write | coeff STEP 1 + STEP 2 | mix | fused critical path (write + STEP 1 + mix) | current: write + cast + GEMM (+ split-K) + read |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.4 | 6.1 avg of 2 | 1.4 | ~11 | 2.4 + 1.7 + ~4 + 9.8 ≈ 18 |
| 16 | 7.9 | 6.8 avg of 2 | 1.6 | ~11 | 2.4 + 1.7 + 4.1 + 2.2 + 9.8 ≈ 20 |
| 64 | 8.6 | 6.9 avg of 2 | 2.0 | ~12 | ≈ 21 |
| 256 | 33.9 | 7.9 avg of 2 | 6.4 | ~45 | 9.8 + 3.3 + 8.2 + 1.9 + 17.9 ≈ 41 |
| 1024 | 110.6 | 24.6 (not split) | 22.3 | 158 | 42.8 + 3.6 + 25.8 + 2.3 + 56.7 ≈ 131 |
| 8192 | 887.6 | 67.9 (not split) | 192.9 | 1148 | 337.6 + 3.6 + 177.8 + 360.4 ≈ 880 |

**Reading:**
- **Decode (1-64 tokens):** the fused critical path is ~11 µs against ~18-20 µs for the current HF path.
  - That would be ~0.7-0.9 ms per decode step over 98 sublayers on this GPU.
  - Most of the gain is moving the 20-iteration log-space Sinkhorn off the critical path. vLLM's `mhc_split` already
    does that for the unfused path, so the margin against the vLLM decode path is smaller. **To be measured there.**
- **Prefill (≥ 256 tokens):** slower. The in-tile coefficient GEMM reuses phi only 16× (cuBLAS tiles 64-128 rows),
  and BT 64 spills registers. Prefill should keep the unfused path.
- **Remaining per-sublayer latency floors:**
  - the log-space Sinkhorn: ~10 µs serial on a 4 × 4 matrix;
  - the write kernel: 7 µs against the old write's 2.4 µs.

## Next (not done)

- **Sinkhorn:** a per-thread scalar 4 × 4 implementation, or an exp-space variant (a numerics check against the
  log-space reference).
- **Integration:** wire the fused path into the vLLM decode graphs (`OellmAmbitiousLayer`: the write of l-1 + the
  read of l at the layer boundary) for T ≤ 64, and measure it there (H100 / 4090).
- **Larger fusions** (DS's "Mega-mHC" direction):
  - the write into the sublayer's output epilogue (o_proj, MoE combine);
  - the input mix and norm into the next projection's prologue.
