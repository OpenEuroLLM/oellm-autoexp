# residual_arch throughput — final measured (130M, 4x GH200, JUPITER)

| arm | start | final | >=200 | gbs/mbs |
|---|---|---|---|---|
| probe1p3B_baseline | - | 442.8 | yes | 128/4 |
| baseline | 157 | 285.5 | yes | 128/32 |
| muown | 126 | 270.1 | yes | 128/32 |
| ngpt_analytic | 88 | 234.8 | yes | 128/32 |
| ngpt_wnorm | 95 | 227.2 | yes | 128/32 |
| ngpt_approx | 107 | 218.3 | yes | 128/32 |
| attnres4_lerp | 64 | 193.1 | no (-3.6%) | 128/32 |
| attnres8 | 60 | 177.5 | no (-12.7%) | 128/32 |
| mhc4 | 42 | 96.2 | no (-52%) | 384/24 |
| mhc2_lerp | 34 | 77.0 | no (-62%) | 384/16 |

6 of 10 above 200 (from 0 of 10). Every arm 1.8x-2.7x its starting throughput.

## What produced the gains (in order of size)
1. torch.compile globally (+19-26%) - was disabled by an inherited
   TORCHDYNAMO_DISABLE=1; needs TRITON_LIBCUDA_PATH set to the *directory*
   holding libcuda.so.1.
2. Liger fused linear+CE - removes the (b,s,V) logits, ~0.41 GiB per
   micro-batch sample at V=50304/seq=4096.
3. Fusing the analytic LERP tail: alpha is per-channel [h] and cos_theta
   per-token [s,b,1], so denom and its rsqrt broadcast to full [s,b,h]
   tensors. ngpt_analytic 192.0 -> 233.4.
4. Compiling the AttnRes *loop* form rather than a stacked one: the stack
   materializes [k,s,b,d] and grows with block count. +18%.
5. Removing ~10 GiB of retained fp32 copies (mul/normalize/pow save operands,
   so .float() on an [s,b,h] tensor keeps a full fp32 copy alive to backward).
6. No dataloader attention mask (FlashAttention takes is_causal and discards
   it); dropping overlap_grad_reduce/param_gather (hybrid logs: 176.9 vs 317.2).

## Why the last four fall short
* AttnRes (177-193, needs 1.04-1.13x). Memory-bound at 0.87-0.97 occupancy with
  no micro-batch headroom left. Next lever: the DepthResidualState refactor to
  preallocated static-shape buffers, which also unlocks CUDA graphs - every arm
  above 200 uses graphs, none below do.
* mHC (77-96, needs 2.1-2.6x). At n=4 every layer moves 4x the residual traffic;
  96.2/285.5 = 0.34 matches that arithmetic. The paper reaches 6.7% overhead
  only with fused TileLang kernels plus selective recompute, and a torchtitan
  implementation is reported near 400 TFLOP/s - so this is an implementation
  gap, closable only by writing the fused mixing kernel.
