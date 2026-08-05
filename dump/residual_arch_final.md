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
  no micro-batch headroom left.
* mHC: see below - now 193.8-195.8, needing only 1.02-1.03x.

## mHC: 94.9 -> 195.8 via fused stream ops
Both large wins came from the same defect repeated twice: an n-long contraction
written as a GEMM/einsum. At n=4 that is a handful of multiply-adds per output
element, so baddbmm/einsum dispatched a kernel over the widest tensor in the
model to do almost no arithmetic - and the write additionally materialized a
full [s,b,n,C] intermediate. Rewritten as fusible elementwise-plus-reduction,
Inductor reads the stream state once per op.

| change | TFLOP/s | kept |
|---|---|---|
| original | 94.9 | - |
| - per-sublayer collapse | 90.6 | reverted (flat) |
| + coefficient cache | 96.6 | yes |
| + fused write | 143.3 | yes |
| + fused read | 190.2 | yes |
| + mbs 24 -> 32 | 195.8 | yes |
| + fused coeffs\|read | 193.0 | reverted |
| + sinkhorn 8 -> 4 | 194.3 | reverted |

The earlier claim in this file - that only hand-written TileLang kernels could
close the gap, because 96.2/285.5 = 0.34 "matches the 4x-traffic arithmetic" -
was numerology. It was ordinary badly-shaped PyTorch.

## CUDA graphs do not help mHC (measured, negative)
Hypothesis: mHC could not be captured because its stream state lived on
`layer._depth_state`, invisible to CudaGraphManager, and graphs were assumed to
be worth what the other arms gain. Fixed by carrying the streams as the layer's
own hidden_states (numerics verified bit-identical to the previous code, and a
captured replay verified bit-identical to eager). Then measured, one session:

| mbs | graphs | TFLOP/s @ iter 60 | mem |
|---|---|---|---|
| 32 | no | 193.8 | 0.993 |
| 32 | yes | OOM in capture | - |
| 24 | no | 189.4 | 0.771 |
| 24 | yes | 188.7 | 0.806 |

At matched micro-batch graphs are worth -0.4%, i.e. nothing. The premise was
wrong twice over: the +19-26% in the list above is *torch.compile*, not graphs,
and "every arm above 200 uses graphs" was a correlation read as a cause. mHC is
bound by bandwidth over the 4x-wider stream state, not by launch overhead, which
is the only thing graphs remove. They also cost memory during capture, and at
0.993 occupancy that forces mbs 32 -> 24, which costs a further 2.3%.

Best mhc4 remains **mbs=32, no graphs: 193.8-195.8 TFLOP/s** (2.04-2.06x).
The refactor is still worth keeping: it deletes hidden cross-layer module state,
and it makes mHC's pipeline payload just the hidden state, so PP>1 now needs
only the shape negotiation widened.
