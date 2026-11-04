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

## Next
- Re-profile with the fused kernels; remaining: fused_moe (~36 us/layer at batch 1, untuned config, BF16-dequantized FP8
  experts), _mhc_read 10 us, CKDA elementwise chain (index/clone/sign/gate), NCCL all-gather / reduce-scatter imbalance.
- HF path: not yet changed.
