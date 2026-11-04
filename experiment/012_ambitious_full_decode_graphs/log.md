# Ambitious decode: whole-step CUDA graphs + fused small-op chains (vLLM)

## Objective
Batch-1-per-GPU decode of amb v4_2ph (export fp8_6000) towards >= 100 tok/s per request, DP4/EP4/TP1 on one MN5 node.

## Changes (vLLM fork, opt-in)
- `OELLM_ENGRAM_TABLES=uva`: Engram tables staged once per node into /dev/shm (16 parallel pread/pwrite, ~25 s / 80 GB table),
  pinned with cudaHostRegister by every rank, rows gathered by a Triton kernel straight from host memory. No host sync,
  so the whole decode step is capturable. Gather verified exact (eager + graph replay).
- Engine: `enforce_eager=False, compilation_config={mode: 0, cudagraph_mode: FULL_DECODE_ONLY, capture sizes 1..64}`
  (no torch.compile). GDN (CKDA) / short-conv (Engram) metadata pad graph batches with NULL_BLOCK_ID 0.
- `OELLM_DECODE_KERNELS=triton` (`layers/oellm_decode_kernels.py`): Engram n-gram hash (exact int64), sigmoid top-k router,
  clamped SwiGLU + post-act norm (routed and shared experts), gated RMS output norm (CKDA), merged CKDA input projections
  (q,k,v,f0,b,g0 as one GEMM; weights become views), fp32 router weight cached. Local kernel checks: hash / router ids /
  rms_gate exact, SwiGLU bf16-identical except the smooth norm (2e-5 of elements, 1 ulp).

## Results (decode-window tok/s per request; 256 prompt tokens, 256 generated, greedy)
| requests per GPU | 010 islands (eager) | full graph | full graph + fused kernels |
|---:|---:|---:|---:|
| 1 on the node | 20.6 | 48.5 | **80.4** |
| 1 | 20.2 | 48.6 | **72.3** |
| 8 | 18.8 | 32.6 | 41.5 |
| 32 | 16.7 | 18.8 | 21.7 |
run1 (o134c.0037) islands vs full graph; run2 (acc_debug 48491xxx, fullgraph012e_dbg) full graph vs + kernels.

Quality, fused vs reference kernels (both full graph): prefill NLL on 4 x 32 prompts (8160 tokens each) delta -0.0003 to -0.0007;
decode logprobs on common prefixes mean |dlogp| 0.003-0.03 (008's reference-vs-reference noise: 0.010-0.020); batch-1 greedy
sequences identical over 64 tokens on 3 of 4 ranks.

## Profile (full graph, reference kernels, batch 1/GPU; results/decode.analysis.json)
18.7 ms/step, 15.4 ms kernel-busy, 1.3 ms NCCL, ~5,200 kernels/step. Per layer type: MoE 48 x ~190 us (55 kernels),
CKDA 42 x ~140 us (47), Engram 2 x ~595 us (477: the Python hash loops), _mhc_read ~10 us each (98).

## Round 2 (validated, run5 = fullgraph012i_dbg on acc_debug)
Added (all opt-in): CKDA one-token decode in place with the sign gauge in FLA's kernel (`fused_recurrent_kda_fwd`
`sign_decode`, FLA bec998d; bit-exact locally), mHC Sinkhorn on a side stream joined at the write + a Triton
coefficient GEMV (exact vs fp64; cuBLAS uses reduced-precision bf16 reductions from 16 tokens), weight-only FP8 experts
`OELLM_MOE_FP8=w8a16` (E4M3 + 1x128 scales on the GPU, in-kernel dequant to the loader's exact bf16 values; ~13 GB/GPU
less; not bit-exact, 0.01-0.07% of outputs differ by MMA order, error vs fp64 identical to bf16: 2.2-2.5e-3).
Per-part switches: `OELLM_DECODE_KERNELS_OFF=engram_hash,route,swiglu,ckda_proj,o_norm,ckda_inplace,mhc_split,mhc_coeffs`.

| per-request decode tok/s | ref full graph | fast (kernels + FP8 experts) |
|---|---:|---:|
| 1 on the node | 48.7 | **108.2** |
| 1 per GPU | 48.9 | **100.0** |
| 8 per GPU | 32.7 | 60.8 |
| 32 per GPU | 18.7 | 32.3 |

Quality: decode logprobs vs prefill re-scoring of the same tokens (8 req/GPU) mean |dlogp| 0.014-0.022 vs reference
0.016-0.022; prefill NLL delta -0.0002..+0.0005; greedy agreement with the reference 55-86% (8-32 req/GPU).

Bug found by the ablation (run4) and fixed (vLLM 2bb5808001): the in-place CKDA call bypassed FLA's `input_guard`;
the merged projection's strided `beta` was read as contiguous for batch > 1 (8 req/GPU: |dlogp| 0.113, 26% > 0.1).

Fast settings: `OELLM_ENGRAM_TABLES=uva OELLM_DECODE_GRAPHS=0 OELLM_DECODE_KERNELS=triton OELLM_MOE_FP8=w8a16
OELLM_FUSION_IMPL=triton OELLM_MHC_IMPL=triton OELLM_ENGRAM_DECODE_IMPL=batched OELLM_LOCAL_EXPERT_LOAD=1` + LLM(enforce_eager=False,
compilation_config={mode: 0, cudagraph_mode: FULL_DECODE_ONLY, cudagraph_capture_sizes: [1..64]}); needs ~160 GB
/dev/shm per node. Not yet in flag_eval_fast_mn5.

## Next
- Speculative decoding (MTP / DFlash, planned): CKDA in-place and batched Engram decode are one token per request;
  verification of k+1 tokens needs multi-token decode + per-token state (FLA num_accepted_tokens), graph sizes for uniform batches.
- Re-profile with the fused kernels; remaining: fused_moe (~36 us/layer at batch 1, untuned config, BF16-dequantized FP8
  experts), _mhc_read 10 us, CKDA elementwise chain (index/clone/sign/gate), NCCL all-gather / reduce-scatter imbalance.
- HF path: not yet changed.
