# Low-bit expert FFN kernel on the RTX 3090 Ti (2026-11-09)

Kernel: `submodules/cute_kernels/lowbit_experts` (Triton). One v4 SwiGLU expert (hidden 2048, expert FFN 768: gate|up
1536 x 2048, down 2048 x 768), weights dequantized in registers inside the GEMM (FP8 E4M3 with 1 x 128 fp32 block
scales = the export's layout, software-decoded on sm_86; int8 per output row; int4 symmetric, groups of 64 / 128, two
per byte), bf16 activations, fp32 accumulation. Two launches: gate + up as one GEMM, then down with silu(gate) * up
applied while its input tiles load. Grouped: all E experts of a layer in the same two launches.

Measured copy bandwidth 903 GB/s; bound = weight + scale bytes / bandwidth. Times are CUDA-graph replays (eager adds
~60 us of Python launch overhead per expert pair, irrelevant for a graph-based decode engine).
Files: `expert_kernel_single.json`, `expert_kernel_grouped.json`, `expert_int4_error.json`.

## One expert per launch (us, graph)

| implementation | M=1 | M=4 | M=16 | M=64 | bound |
|---|---:|---:|---:|---:|---:|
| torch bf16 (gate|up matmul, silu*up, down) | 18.1 | 21.6 | 22.1 | 28.1 | 10.4 |
| torch dequant fp8 / int8 / int4 then bf16 | 125 / 122 / 196 | 126 / 124 / 195 | 127 / 125 / 196 | 135 / 132 / 210 | 5.4 / 5.2 / 2.8 |
| HF-export grouped FP8 GEMM (oellm_kernels) | 24.2 | 24.1 | 24.7 | 43.9 | 5.4 |
| fused bf16 | 16.4 | 16.5 | 17.6 | 34.9 | 10.4 |
| fused fp8 | 17.0 | 17.1 | 19.3 | 35.4 | 5.4 |
| fused int8 | 15.1 | 14.8 | 18.4 | 33.9 | 5.2 |
| fused int4 g128 | 16.7 | 17.3 | 17.7 | 36.6 | 2.8 |
| fused int4 g64 | 17.3 | 18.0 | 18.5 | 38.3 | 2.9 |

A single expert is latency-bound (~7 + ~10 us for the two kernels at M=1, whatever the format): 7-12x faster than
torch dequant and 1.4x faster than the existing export kernel, but not bandwidth-bound.

## Grouped: all experts of a layer in one launch (us per expert, graph)

| format | 5 experts, M=1 | 5 experts, M=16 | 48 experts, M=1 | 48 experts, M=16 | bound |
|---|---:|---:|---:|---:|---:|
| bf16 | 10.9 | 11.2 | 10.0 | 10.3 | 10.45 |
| fp8 | 7.6 | 7.9 | 5.8 | 8.6 | 5.39 |
| int8 | 6.5 | 9.6 | 5.2 | 8.0 | 5.24 |
| int4 g128 | 5.9 | 6.9 | 3.8 | 5.2 | 2.78 |
| int4 g64 | — | — | 6.0 | 6.6 | 2.94 |

With the layer's experts in one launch the kernel reaches the bandwidth bound for bf16 / int8 and 93 % for fp8 at
48 experts; int4 g128 reaches 73 %, g64 only ~50 % (smaller tiles, more scale loads). A batch-1 MoE layer (top-5)
costs ~38 us (fp8) / ~29 us (int4 g128) of routed-expert compute when the experts are resident. Remaining headroom:
int4 (a Marlin-style layout / CuTe kernel with pre-shuffled int4 tiles and async copies would close most of the 27 %),
and M=16 at 48 experts (dequant cost per tile, not bytes).

Numerics: max |y - ref| / max |ref| = 3-7e-3 against the fp32 reference of the same quantized weights for every
format, token count and grouping (the bf16 path through the same kernel: 3-5e-3); grouped == single bitwise.

## int4 error on real v4 expert weights

Sample: experts 0, 1 and the largest-scale ("runaway") expert of MoE layers 1, 51, 97 of the v4_2ph step-8000 export;
FP8 weights decoded to fp32 as the reference; relative error of the expert output on 512 Gaussian inputs (the
experts read RMS-normalised states; real activations would change the outlier picture).

| scheme | weight error (gate|up / down) | expert-output error (mean, range) |
|---|---|---|
| int8 per channel | 0.8 % / 0.9 % | 1.6 % (1.3-3.0 %) |
| int4 sym RTN g128 | 12.3 % / 12.5 % | 21.1 % (19.6-22.0 %) |
| int4 sym RTN g64 | 11.1 % / 11.3 % | 18.9 % |
| int4 sym, clip search g128 | 10.2 % / 10.4 % | 17.4 % |
| int4 sym, clip search g64 | 9.5 % / 9.7 % | 16.2 % |
| int4 asym RTN g128 | 10.0 % / 10.2 % | 17.1 % |
| int4 asym RTN g64 | 9.0 % / 9.1 % | 15.3 % |

- int8 per channel is near-lossless (1.6 %), except the deep runaway expert (layer 97 expert 317, kurtosis 1130,
  max |w| 0.93 vs ~0.02-0.06): 3.0 % -- outliers hurt per-channel scales; group scales are unaffected.
- int4 costs 15-21 % of every expert's output (SwiGLU amplifies the ~10 % weight error); the best plain scheme
  (asymmetric, groups of 64) is 15 %. Across 48 MoE layers that is unlikely to be free: int4 needs at least
  calibration with real activations (GPTQ / AWQ-style) and probably QAT to be near-lossless -- the s3 int4-QAT test
  is warranted. Speed-wise g128 is the format to aim for (g64 is ~1.6x slower in the kernel).
