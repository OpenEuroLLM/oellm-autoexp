# HF prefill / scoring speed of the ambitious v4_2ph export

## Objective
Maximal prefill (scoring) speed of the HF port (`oellm_autoexp/hf_export/oellm_ambitious`) for
hybrid_ambitious_v4_2ph_mn5 exports (98 layers: 48 MoE with 360 FP8 experts, 42 CKDA, 6 GQA
attention, 2 Engram with 80 GB tables; mHC-4 on every layer), in the export code itself so that
every harness (lighteval accelerate, lm-eval HF, score_on_flag) gets it.

Baseline: 008's fast HF path (grouped `torch._grouped_mm` on BF16-expanded experts, fused
Sinkhorn, sorted host lookup) -- 32 x 256-token prompts in 6.5 s (~1.25k tok/s), only through
score_on_flag's loader. A plain `from_pretrained(device_map="auto")` (lighteval) got the reference
per-expert loop and accelerate's placement, which cannot hold an 80 GB Engram layer on a GPU.

## Changes (export code; `OELLM_HF_IMPL=reference` restores the old path)
- `oellm_kernels.py` (new, shipped with exports / refresh_export / vllm_parity):
  - FP8 experts grouped by expert in a Triton GEMM, dequantized in the kernel exactly as Megatron
    (E4M3 * fp32 scale, rounded once to bf16; native cvt on sm89+, an exact fp16 bit trick
    elsewhere); token rows gathered in the kernel; sync-free block schedule; autotuned tiles;
  - sigmoid top-k router, SwiGLU + clamp + pab, ordered combine (ascending expert, bf16 rounding
    after each add, + shared expert) -- the reference's rounding points;
  - fused mHC read (+ the layer's input RMSNorm) and write; RMSNorm;
  - Engram hashes on the GPU (exact int64) and a zero-copy gather (times the sign rows) from tables
    pinned in host memory (cudaHostRegister in place).
- Attention: one GEMM for q | gate | k | v, fused qk-norm, GQA inside SDPA, XSA without
  expanding v. CKDA without a cache: FLA's fused input projections, no final state.
- Model: plain calls (`use_cache` not given) keep no decoding state; eval-mode calls without
  labels run under no_grad (lighteval scores with grad enabled; `OELLM_HF_GRAD=1` keeps it);
  `from_pretrained(device_map="auto"|...)` loads to host memory and places whole layers on the GPUs
  (`place_on_gpus`, Engram tables pinned; `OELLM_HF_PLACEMENT=accelerate` = stock); no host
  synchronization in the forward, so micro-batches (default 2 x stages) flow through the per-GPU
  stages concurrently (`OELLM_HF_PIPELINE=0` off, `OELLM_HF_PIPELINE_CHUNKS`); logits returned on
  the input's device.

## Local checks (3090 Ti, reduced model: real widths, random weights; scripts/tiny_model.py)
- Errors vs a float64 copy of each module (check_truth.py): attention 0.0046 (ref 0.0046), MoE
  0.0050 (ref 0.0057), Engram bitwise equal to the reference. mHC read / write bitwise equal
  (read + norm: ~1e-5 of elements 1 ulp). No host synchronization in the fast forward
  (set_sync_debug_mode). Greedy generate fast == reference; cached decode == recompute.
- Full-model estimate from the reduced model (bench_layers.py): 1x2048 737 -> 347 ms, 8x512
  1096 -> 603 ms vs 008's settings (sm86 numbers; grouped_mm falls back to a per-expert loop on
  sm86, so the MoE part of the comparison is not representative).
- mHC kernels on separate stream rows: write 279 -> 889 GB/s, read 282 -> 479 GB/s (3090 Ti).

## MN5 run (results/mn5_run1, scripts/run_mn5.sh)
- 2026-11-04 14:48: acc_debug held by the 012 session's job; submitted to acc_ehpc as 48494126
  (1 node, 1 h, snapshot autoexp_hf013a). Stages: tests/hf_export_gpu, check_truth / check_fast /
  layer timings on GPU 0 (fast vs 008 settings), then the real export fp8_6000 via a hard-linked
  view (`<export>_hf013`, code files replaced atomically) -- fast vs hf008, logprob parity, the
  reference on 8 prompts.
- Run 1 = acc_debug 48495311 (snapshot autoexp_hf013b, as06r5b27; 48494126 on acc_ehpc stays pending,
  old code). results/mn5_run1/summary.md.
  - tests/hf_export_gpu: test_ambitious_v2_hf all pass; 7 failures in test_fla_hf / test_engram_export_dtypes
    (OellmHybrid models, whose code path this change leaves untouched: pre-existing in the ep image).
  - H100, native FP8 path: errors vs float64 attention 0.00462 (ref 0.00462), MoE 0.00492 (ref 0.00521),
    Engram bitwise; reduced-model NLL delta +0.0003; no host syncs.
  - Reduced model, full-model estimate on ONE GPU (no pipelining): fast 13.6k / 20.8k / 25.5k tok/s at
    1x2048 / 8x512 / 8k-token batches vs the 008 settings 7.0k / 9.1k / 10.6k (1.9-2.4x). MoE 395 -> 151 ms,
    mHC 229 -> 51 ms per 8k tokens; MoE is 47 % of the rest (grouped FP8 GEMM ~120-150 TFLOP/s).
  - Real export, hf008 (score_on_flag loader, BF16 experts, 52 GiB on GPUs 0-2): load 176 s; w1 32x(1x256)
    6.40 s = 1,281 tok/s (008 measured 6.50 s); w2 32x256 / w3 4x2048: 0.827 s = 9,910 tok/s (the estimate
    said 10.6k: the estimate holds); w4 OOM'd in the parity code. fast: the remote-code loader did not copy
    oellm_kernels.py (`from . import` form) -- fixed (38121aea), verified with a local remote-code load.
- Run 2 = acc_debug 48496369 (autoexp_hf013c). results/mn5_run2/summary.md.
  - Grouped expert GEMM (GEMM1 + GEMM2, v4 shapes): FP8 Triton vs torch._grouped_mm on BF16 experts:
    512 tok 18.9 vs 10.3 TFLOP/s, 2k 60.7 vs 38.6, 8k 168 vs 128, 16k 224 vs 201, 32k 262 vs 236.
    The swapped-operand variant brings nothing (autotune keeps both).
  - Real export (fp8_6000), fast loaded by from_pretrained(device_map="auto"): load 320 s, 20-21 GiB per GPU
    (hf008: 52 GiB on 3 GPUs). Without pipelining: w2 32x256 / w3 4x2048 0.367 s = 22.3k tok/s
    (2.25x hf008), w4 1x8192 21.6k (2.19x), w1 32 x (1x256) 1.9k tok/s (1.48x; ~135 ms per 256-token
    forward: host-bound, ~0.8-1 ms of launches per MoE / CKDA layer). Original reference: 95 tok/s.
  - Pipelining over the 4 GPUs LOST: w2 with 8 / 4 / 16 micro-batches 1.13 / 0.59 / 2.19 s vs 0.367 s
    unpipelined -- every micro-batch costs the full host launch time; now adaptive (one micro-batch per
    >= 8k tokens, at most one per stage).
  - Parity (next-token logprobs): fast vs hf008 NLL delta +0.00025 / +0.00025 / -0.00076 / +0.00039
    (w1-w4, ~8.2k tokens each), mean |dlogp| 0.025-0.029; vs the reference (8 prompts, 2040 tokens):
    fast +0.00136 (mean |dlogp| 0.0255), hf008 +0.00074 (0.0219). Run 3: the reference on all 32.
- Run 3 = acc_debug 48497941 (autoexp_hf013d), fast only. results/mn5_run3/summary.md.
  - Adaptive pipelining: 8k-token batches unchanged (22.4k tok/s, no micro-batches); w5 32 x 1024
    (32k tokens) 0.962 s = 34.1k tok/s with 4 micro-batches vs 26.2k unpipelined (+30 %); 2 / 8 forced
    micro-batches 33.5k / 25.1k.
  - Parity vs the reference on all 32 prompts (8160 tokens): fast NLL +0.00048 (mean |dlogp| 0.0257),
    hf008 +0.00023 (0.0218) -- 008's reference-vs-reference noise was 0.010-0.020 per token.

## Result
| path (v4_2ph fp8_6000, 1 node 4 x H100) | 32 x 256 | 4 x 2048 | 1 x 8192 | 32 x 1024 | 32 x (1 x 256) | GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| reference HF (from_pretrained default before) | | | | | 95-99 tok/s | |
| 008 fast HF (score_on_flag loader only) | 9.9k | 10.0k | 9.9k | -- | 1.28k | 52 GiB x 3 |
| 013 (default from_pretrained(device_map="auto")) | 22.4k | 22.4k | 21.6k | 34.1k | 1.85-1.9k | 21 GiB x 4 |

## Next
- Host launch overhead (~0.1 s per forward) bounds small forwards (1 x 256: ~135 ms) and the
  pipelining; CUDA graphs need repeated shapes (lighteval's padded lengths change per batch).
- Two replicas per node (the model now fits 2 GPUs): place_on_gpus would have to take the GPU subset
  from lighteval's per-process max_memory, and the 160 GB of Engram tables be shared between processes.
- Small-batch MoE GEMMs are weight-read bound at ~1.3 TB/s.
- Existing exports keep their old code until refreshed (`python -m oellm_autoexp.hf_export.refresh_export <dir>`;
  not done on the live exports).
- 2026-11-04 (user: make it the default via env in the eval configs, refresh the live exports):
  hf_fast_prefill_env.yaml (the 013 switches, explicit) in flag_eval_oellm_mn5 and hf_export_mn5 (so
  hf_export_v2's score / paired checks); the lighteval HF branch runs refresh_code.sh on its export
  first (OELLM_HF_REFRESH_CODE=1). lighteval rows stay on vLLM by default (generative). Refreshed in
  place (code files only, atomic renames, config.json untouched, md5 = checkout d3319f92):
  ambitious_v4_2ph_fp8_{1000,2000,3700,5000,5800,5900,6000} + views/identity/ambitious_v4_2ph_fp8_6000.
  Not refreshed: LDM / v3 / e216 exports (fast path validated on mHC only), the _hf008 record.
