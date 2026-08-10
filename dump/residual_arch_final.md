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

## mHC: 94.9 -> 195.8 by way of fused stream ops
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

At matched micro-batch graphs are worth -0.4%, that is nothing. The premise was
wrong twice over: the +19-26% in the list above is *torch.compile*, not graphs,
and "every arm above 200 uses graphs" was a correlation read as a cause. mHC is
bound by bandwidth over the 4x-wider stream state, not by launch overhead, which
is the only thing graphs remove. They also cost memory during capture, and at
0.993 occupancy that forces mbs 32 -> 24, which costs a further 2.3%.

Best mhc4 is now **gbs=448 / mbs=28, CUDA graphs, Triton backward: 199.8 TFLOP/s** (2.11x).
The refactor is still worth keeping: it deletes hidden cross-layer module state,
and it makes mHC's pipeline payload just the hidden state, so PP>1 now needs
only the shape negotiation widened.

## Where mHC's time actually goes (torch.profiler, GH200)
Traces from `config/experiments/residual_arch/profile_mhc_jupiter.yaml`, mhc4 vs
the same-geometry baseline, aggregated by `dump/parse_mhc_trace.py`. Diffing the
two traces rather than classifying kernels by name matters: the compiled backward
of a fused op is named after different aten ops than its forward, and a
name-based bucketing missed every backward kernel.

mHC adds 1830 ms on a 4222 ms baseline step (+43%, consistent with the 1.46x
throughput ratio):

| group | ms | share of the mHC cost |
|---|---|---|
| backward of read/write | ~878 | 48% |
| eager elementwise (casts) | 279 | 15% |
| Sinkhorn fwd+bwd | ~240 | 13% |
| coefficient GEMMs | ~180 | 10% |
| forward write | 142 | 8% |
| forward read | 122 | 7% |
| RMS-norm pass | 115 | 6% |

Nearly half is *backward*, which no earlier optimization round had looked at.

### Per-term roofline, measured on GH200 (`dump/mhc_bwd_gh200.py`)
Against a 3617 GB/s d2d ceiling, at training shapes:

| term | ms | % peak | best alternative |
|---|---|---|---|
| forward write | 0.448 | 84% | - |
| grad_streams (contract over n) | 0.400 | 84% | matmul 14x worse |
| grad_h_post (over C -> n) | 0.329 | 63% | matmul worse |
| **grad_h_res (over C -> n x n)** | **1.176** | **28%** | **matmul 1.38x** |

Everything is at roofline except `grad_h_res`, which is also the largest single
kernel mHC adds (1.43 ms/sublayer, 3.6x the next). It contracts a 576-long axis
into an n x n output -- a GEMM shape -- while every other term contracts the
4-long stream axis, where a GEMM is catastrophic. Exactly the forward lesson in
reverse.

**Local hardware cannot be used to choose here.** On a 3090 Ti every term
measures at roofline and matmul is a 7% regression; a quarter of the bandwidth
hides anything occupancy-bound rather than bandwidth-bound.

### Hypotheses this killed
* *Fuse the RMS-norm into the previous write* (the plan at the end of the CUDA
  graph round): Inductor turns the write from a pointwise kernel into a reduction
  kernel, costing +0.756 ms to save 0.633 ms. Net negative.
* *Fuse the backward terms that share a reduction axis*: Inductor emits the same
  5 kernels and the same timings; it will not fuse across different iteration
  spaces. 1.01x.
* *`bmm` with tokens grouped 64x* is genuinely faster (1.72x) but computes 64x
  the FLOPs and needs a 268 MB transient -- unusable at 0.993 memory occupancy.

### Fused Triton write backward: merged, +0.67%
`megatron/core/transformer/mhc_kernels.py` computes all four write-backward
terms in one pass over the upstream gradient, instead of the four kernels
Inductor derives from the forward, each of which re-reads it. Isolated on GH200
that is 1.842 ms against 2.150 (1.17x). End-to-end, same session, same build,
`MHC_TRITON_BACKWARD` the only difference:

| arm | TFLOP/s @ iter 60 | ms/iter |
|---|---|---|
| mhc4, Inductor backward | 194.8 | 2022.7 |
| mhc4, Triton backward | **196.1** | 2009.1 |
| mhc4, Triton + CUDA graphs | OOM at 17/36 graphs | - |

**+0.67%, against ~1.7% predicted** from the isolated measurement (0.308 ms
saved x 108 sublayer invocations per step = 33 ms; only 13.6 ms materialized).
(Superseded by the widened tile search below, which took the isolated figure to
1.29x.)
About 40% of the isolated gain translates. Two candidate explanations were
checked and refuted: the gradient is contiguous in every call, so the
`.contiguous()` is a no-op, and memory occupancy is identical (0.9934) either
way, so the fused kernel frees nothing. The remainder is unexplained -- the
honest reading is that a microbenchmark on freshly allocated, L2-warm buffers
overstates what a kernel does inside a step running at 99.3% memory occupancy.

The CUDA-graph arm is worth noting separately: it OOM'd at the same 1.12 GiB
allocation as before, but it got to 17 of 36 graphs **without**
`cudaErrorStreamCaptureInvalidated`. So a Triton backward *is* capturable where
the torch.compile'd one was not -- warming a one-shot autotune search during a
forward is enough, where pre-warming Dynamo was not, because Dynamo could still
choose to recompile inside the capture. Graphs remain unusable here for the
unrelated reason that they do not fit at mbs=32.

### Widened tile search, and CUDA graphs finally capturing
Searching the tile cross product instead of eight hand-picked configs took the
fused kernel from 1.842 ms to 1.704 (1.17x -> 1.29x against Inductor's 2.194).

Capture was never invalid once the backward was Triton rather than
torch.compile -- it only failed to *fit*, needing ~1.2 GiB at 0.9934 occupancy.
mbs could not be tuned into the gap because it must divide gbs/DP = 96, leaving
24 and 32 with nothing between; raising gbs to 448 makes gbs/DP = 112, which 28
divides. That is the whole fix.

| arm | TFLOP/s @ iter 60 | mem |
|---|---|---|
| mbs=32, no graphs | 198.8 | 0.9934 |
| **gbs=448 / mbs=28, full CUDA graphs** | **199.8** | 0.9235 |
| mbs=32, graphs on attn sublayers only | OOM | - |
| mbs=32, graphs, 1 capture warmup step | OOM | - |

Neither of the two ways of shrinking the capture itself worked; only making room
in steady state did. Graphs are now worth +0.5% *despite* running at a smaller
micro-batch, where the earlier measurement had them at -0.4% at matched mbs --
consistent with the backward having changed underneath them, though at this size
the difference is close to the run-to-run noise floor.

**mhc4: 94.9 -> 199.8 TFLOP/s (2.11x).**

| step | TFLOP/s |
|---|---|
| fused stream ops (forward) | 195.8 |
| + fused Triton write backward | 196.1 |
| + widened tile search | 198.8 |
| + CUDA graphs (gbs 448 / mbs 28) | 199.8 |

### The matmul lever, and why it is not merged
Routing `grad_h_res` through matmul by way of a custom autograd Function is worth
1.38x on that kernel, about +2.1% end-to-end -- roughly 195.8 -> 200. The
backward was verified exact against autograd in float64.

It is **not merged**: a custom Function's backward compiles on first call, and
that call lands inside Megatron's backward graph capture, which dies with
`cudaErrorStreamCaptureInvalidated`. Pre-warming the compiled backward and the
cuBLAS workspace during the forward both failed to fix it. Shipping it would
trade ~2% for a broken capture path on an arm that measures no benefit from
graphs anyway, so the branch keeps the slower, capturable formulation.

## mhc4 + nGPT machinery: measured, and it hurts (6B tokens, 130M)
The question this sweep was set up for.

**Correction (2026-08-10).** An earlier version of this table quoted mhc4 at
2.638 and called it a final loss. It is neither: 2.638 is mhc4's *training* lm
loss at iteration 10350, and mhc4 never finished -- both attempts (1229316,
1231292) hit TIMEOUT, the second after 12h at iteration 10350/11445. Comparing
it against other arms' 11445 *validation* losses was wrong on two counts.

**Second correction (2026-08-10), larger than the first.** mhc4 was also not
comparable to anything. The arm overrides global_batch_size to 384 while
train_iters is derived from aux.gbs (128), so at equal iteration counts it had
consumed 3x the samples of every other arm -- 18.0B tokens against 6.0B.

Five arms carry the defect, not two: mhc4, mhc2_lerp, mhc2_lerp_lr0.002,
ngpt_analytic_lr0.002 and ngpt_wnorm_lr0.002 all show 768,000 consumed samples
at iteration 2000 against 256,000 for everything else. The _lr0.002 reruns are
inconsistent among themselves -- ngpt_approx_lr0.002 is gbs 128 while
ngpt_analytic_lr0.002 and ngpt_wnorm_lr0.002 are 384. Only mhc4 of the five ever
reached 11445, so the other four never entered a results table, but their
intermediate validation numbers (ngpt_analytic_lr0.002 2.946 @it6000,
ngpt_wnorm_lr0.002 3.037 @it6000, mhc2_lerp 2.778 @it8000) are 3x-token figures
and must not be read against the 6B arms.

Every arm that appears in the final comparison other than mhc4 is gbs 128 / 6.0B,
verified from consumed samples rather than from config: all the nGPT arms
included. Both the it10000 and it11445
mhc4 figures below are therefore 3x-data numbers, and the "0.145 lead over
baseline" they appeared to show was a token-budget artifact. Fixed in
commit fc0ede3: train_iters and aux.total_tokens now read
${.global_batch_size}, so the budget holds at 6.0B under any batch override.

mhc4 was resumed from its iteration-10000 checkpoint on the current build (195.9
TFLOP/s against the 95 it originally ran at) and finished the remaining 1445
iterations in 51 minutes. Its true converged number is **2.583 at 18.0B tokens**,
which stays on record as what mHC reaches with 3x the data and is excluded from
every comparison below.

### The controlled comparison
mhc4_gbs128 is mHC at the batch every other arm used: 11445 iterations, gbs 128,
lr 0.002, same geometry/data/schedule as baseline, differing in exactly one
thing. Consumed samples at iteration 11400 are identical to baseline's
(1,459,200), verified in both logs.

| arm | val @ it11445 |
|---|---|
| **mhc4_gbs128** | **2.666** |
| attnres4_lerp (lr 0.002) | 2.682 |
| attnres8_lerp (lr 0.002) | 2.689 |
| **baseline (lr 0.002)** | **2.721** |

**mHC beats the plain transformer by 0.055 on a fully controlled comparison**,
and is the best arm in the 6B-token sweep. That is a third of the 0.145 the
confounded comparison advertised, but it is real: the gap ran 0.089 / 0.070 /
0.060 / 0.059 / 0.057 / 0.055 across it2000-it11445, that is it compressed early
and then held, unlike the ngptpp lead which decayed all the way to 0.032.

Composing nGPT's magnitude machinery with mHC is *not* additive -- it is
negative. At matched tokens and matched lr this cannot yet be quantified,
because mhc4_lerp_scalers ran at lr 0.001 and mhc4_gbs128 at 0.002; at their own
learning rates the composition arms land at 2.714 and 2.919 against plain mHC's
2.666.

That is consistent with the two schemes constraining the same quantity twice:
mHC already makes the mixing operator non-expansive, so pinning the stream to
the unit sphere as well removes a degree of freedom rather than adding one.

Throughput footnote: these arms ran at 65 TFLOP/s because MHCMixer.write's LERP
branch was silently running eager (a lost `_compiled` definition swallowed by an
`except Exception`). Fixed afterwards -- 125.9 at mbs=16 -- but the loss numbers
above are unaffected, since global_batch_size never changed. The LERP path caps
at mbs=16; mbs=28/32 OOM even with recompute enabled on the compiled build.

## ngptpp: nGPT configured the way canGPT actually runs it (6B tokens, 130M)

Every nGPT arm lost to mhc4, so before concluding nGPT loses on merit, the four
config differences against `../titan-sci` (`titan_sci/models/canGPT_master`, SFT
run `cangpt_master_ams_stream_sft_latest_jupiter.toml`) were removed:

1. **Weight normalizer axis.** Both codebases resolve output projections with
   `norm_dim = 0 if out_norm_dim_0 else 1`. canGPT leaves it False (its SFT
   config has no `[normalizer]` section, so the dataclass defaults apply); our
   arms set True. Same name, same semantics, opposite axis.
2. **LERP mode.** canGPT_master does not implement `analytic` at all -- its set
   is approx|direct|both|direct_ste|both_ste|approx_fwd, and the SFT run uses
   `approx`. The exact-cos(theta) factor is our addition, so canGPT's recipe was
   never tuned against it.
3. **base_scale** 0.01 vs our `1/sqrt(dim)` = 1/24 at hidden 576. `alpha_scaler`
   is frozen, so this is ~4x the effective step size on alpha.
4. **No scaler LR boost.** The SFT run has no `[weight_norm_adam]` section and
   `mup_lr_scaling=false`; the eigen-LR reparameterization is the whole
   mechanism. Our 4.0 was invented here.

Plus weight decay 0.1 rather than 0, mirroring canGPT's hardcoded
`_HEAD_WEIGHT_DECAY`. With the normalizer on it only bites the head: normalized
matrices are shrunk and renormalized in the same step, and the 1-D scalers carry
`is_ngpt_scaler`, whose wd multiplier defaults to 0.

| arm | val @ it10000 | val @ it11445 |
|---|---|---|
| ngptpp (untied, ~159M) | 2.923 | 2.825 |
| ngptpp_tied (~130M) | 3.015 | 2.915 |
| ngpt_wnorm (old settings) | 3.047 | 2.972 |

**The config was worth 0.057.** `ngptpp_tied` is the like-for-like comparison
(`ngpt_wnorm` with the four fixes); the gain is real but small against a ~0.2
deficit. The lead also compressed monotonically -- 0.30 at it2000, 0.10 at
it4000, 0.045 at it6000, 0.034 at it8000, 0.032 at it10000 -- that is the fixes buy
a faster start, not a better endpoint. WSD decay did not re-open it.

`ngptpp` (untied) is *not* comparable to the rest of the sweep: untying adds
28.97M params (50304 x 576), 22% more model.

### Where the deficit actually lives
Sorting the completed arms at it11445: baseline 2.721, ngpt_analytic 2.729,
ngpt_approx 2.739. **The nGPT LERP residual alone is a wash against a plain
pre-norm transformer.** Every arm carrying the weight normalizer lands 0.2-0.25
behind: ngpt_wnorm 2.972, mhc4_lerp_scalers_wnorm 2.919, attnres4_lerp_wnorm
3.004, ngptpp_tied 2.915. Correcting axis/mode/base_scale/LR recovers 0.057 of
that; the rest is not a configuration problem.

The likely cause is the magnitude-reinjection machinery we never ported:
canGPT's SFT model also runs `softmax_scale`, `ffn_act_scale`,
`ffn_up_scale`/`ffn_down_scale`, SSMax, scalar `tanh_sq` attention gating and a
learnable pre-head scale vector. None exist in this Megatron build. Stripping
weight magnitude without canGPT's full scaler set to put it back is exactly the
failure mode observed. That, not more config matching, is the next experiment.

Note `head_unconstrained` is inert throughout this sweep: it runs
`untie_embeddings_and_output_weights: false`, that is *tied*, so there is no
`output_layer.weight` parameter, `classify_param` returns "embedding", and the
shared matrix is normalized regardless of the flag.

## The sweep's second uncontrolled variable: learning rate

Pulling lr out of each arm's log shows the 6B-token arms split across two
learning rates, and **there is no baseline at lr 0.001**:

| lr 0.002 | val | | lr 0.001 | val |
|---|---|---|---|---|
| attnres4_lerp_lr0.002 | 2.682 | | mhc4_lerp_scalers | 2.714 |
| attnres8_lerp | 2.689 | | attnres4_lerp | 2.715 |
| ngpt_approx_lr0.002 | 2.715 | | ngpt_analytic | 2.729 |
| **baseline** | **2.721** | | ngpt_approx | 2.739 |
| attnres4 | 2.737 | | ngptpp (untied, ~159M) | 2.825 |
| attnres8 | 2.741 | | ngptpp_tied | 2.915 |
| muown | 2.871 | | mhc4_lerp_scalers_wnorm | 2.919 |
| mhc4_gbs128 | **2.666** | | ngpt_wnorm | 2.972 |
| | | | attnres4_lerp_wnorm | 3.004 |

Every nGPT and mHC-composition result sits in the right-hand column with no
controlled reference. The missing run is baseline at lr 0.001 (~2h); without it
those nine numbers can only be ranked against each other.

mhc4_gbs128 landed at lr 0.002 by inheriting the base, as the original mhc4 did,
so the one clean architecture comparison in the sweep exists by luck rather than
design. Arms never reaching 11445 and therefore excluded: ngpt_scalers,
ngpt_scalers_wnorm, muown_nockpt, ngpt_analytic_lr0.002, ngpt_wnorm_lr0.002,
mhc2_lerp_lr0.002.

## The standardized sweep: one comparable set (2026-08-10)

Every arm below ran 11445 iterations at gbs=128, lr=0.002, 6.0B tokens, on the
same 130M geometry, data and schedule. Earlier tables in this file mixed two
learning rates and two token budgets; this one does not. tok/s/GPU is computed
from wall-clock (gbs x seq / ms_per_iter / 4), not from Megatron's reported
figure, which is derived from a config formula.

| arm | mbs | params | tok/s/GPU | final |
|---|---|---|---|---|
| mhc4_gbs128 | 32 | 126.5M | 193,750 | **2.665** |
| attnres4_lerp | 16 | 124.5M | 134,185 | 2.682 |
| attnres8_lerp | 16 | 124.5M | 152,445 | 2.689 |
| mhc4_lerp_scalers | 8 | 126.5M | 123,049 | 2.699 |
| ngpt_scalers | 16 | 124.5M | 197,249 | 2.700 |
| mhc2_lerp | 16 | 124.8M | 191,850 | 2.703 |
| ngpt_analytic | 32 | 124.5M | 231,781 | 2.708 |
| ngpt_approx | 32 | 124.5M | 218,453 | 2.715 |
| **baseline** | 32 | 124.5M | **276,290** | **2.721** |
| attnres4 | 16 | 124.5M | 226,455 | 2.737 |
| attnres8 | 32 | 124.5M | 149,200 | 2.741 |
| ngptpp (untied) | 16 | 153.5M | 187,998 | 2.775 |
| ngptpp_tied | 16 | 124.5M | 184,219 | 2.859 |
| mhc4_lerp_scalers_wnorm | 8 | 126.5M | 120,570 | 2.859 |
| muown | 32 | 124.5M | 269,862 | 2.871 |
| ngpt_wnorm | 32 | 124.5M | 226,064 | 2.874 |
| attnres4_lerp_wnorm | 16 | 124.5M | 195,864 | 2.887 |
| ngpt_scalers_wnorm | 16 | 124.5M | - | **NaN at it7600** |

### What the standardization changed
Running the nGPT-geometry arms at lr 0.001 while baseline ran 0.002 was costing
them more than any architectural difference measured in this sweep:

| arm | lr 0.001 | lr 0.002 | gain |
|---|---|---|---|
| attnres4_lerp_wnorm | 3.004 | 2.887 | 0.117 |
| ngpt_wnorm | 2.972 | 2.874 | 0.098 |
| ngptpp_tied | 2.915 | 2.859 | 0.056 |
| ngpt_analytic | 2.729 | 2.708 | 0.021 |

Two earlier conclusions do not survive it. "The nGPT LERP residual alone is a
wash against a plain pre-norm transformer" was based on 2.729 vs baseline 2.721;
at matched lr it is 2.708, that is 0.013 *ahead*. And the mhc4+LERP+scalers penalty
shrinks from 0.080 to 0.033 (2.699 vs 2.665) -- still negative, but a third of
what was reported.

### What survives every correction
* **mHC wins, by 0.056 over baseline**, on tokens, batch, lr and schedule all
  matched. It is the only scheme that beats baseline by more than 0.04.
* **The weight normalizer costs 0.14-0.15** and nothing recovers it: not the
  token fix, not the lr fix, not matching canGPT's configuration. Every arm
  carrying it lands 2.859-2.887, against 2.665-2.721 for everything else.
* **Block AttnRes only works with the LERP.** Alone it is worse than baseline
  (2.737 / 2.741); with the LERP it is second and third best (2.682 / 2.689).
* **mHC and nGPT do not compose.** Every nGPT component is either harmful with
  mHC (wnorm -0.19, LERP -0.033) or has no mechanism to help: the scalers exist
  to reinject magnitude that normalization removes, and mHC normalizes nothing.

### Cost
mHC's 0.056 costs 30% of throughput (193,750 vs 276,290 tok/s/GPU) and runs at
99.6% memory occupancy with no micro-batch headroom left. At matched wall-clock
baseline sees ~1.4x the tokens, which on these curves is worth roughly as much as
mHC's advantage -- so the scheme is close to break-even against simply training
the baseline longer. That comparison, and a second seed, are the experiments that
decide whether any of this is worth building on.

### ngpt_scalers_wnorm diverged
Healthy to iteration 7550 (loss 3.121, grad norm 0.367), grad norm NaN at 7600,
never recovered. The base sweep sets check_for_nan_in_loss_and_grad: false for
CUDA-graph capture, so it would have exited 0 with a NaN final had it not been
cancelled. ngpt_scaler_lr_mult 4.0 puts the scalers at an effective lr of 0.008,
and on a weight-normalized network the scalers carry all the magnitude. Note
canGPT's own SFT config gives the scalers no lr boost at all.

## The speed-Pareto result: no scheme moves the frontier (2026-08-10)

Every arm in this sweep runs 11445 iterations, that is matched *tokens*. That answers
"which scheme is better per token", not "which is better per second" -- and the
arms that beat baseline all do so while running slower. Giving baseline the same
wall-clock instead of the same tokens erases the difference:

| arm | elapsed | final | baseline at equal wall-clock | elapsed | final |
|---|---|---|---|---|---|
| mhc4_gbs128 | 2:13:00 | 2.6654 | 16,320 iters | 2:15:38 | 2.6662 |
| loop12x2_plain | 2:25:48 | 2.6782 | 18,046 iters | 2:25:27 | 2.6814 |
| loop12x2_ngpt | 2:55:18 | 2.6628 | 21,648 iters | (pending) | - |

Both settled comparisons are dead heats: 0.0008 and 0.0032, against the ~0.002
that three architecturally identical baselines agreed to at matched iterations.
The wall-clock matching is tight -- 2:25:48 vs 2:25:27 is 21 seconds apart -- and
in the mhc4 case the baseline got 2% *more* time than the arm it is pricing.

**mHC's +0.056 over baseline is entirely a compute effect.** It runs at 0.70x
throughput; spending those same seconds on 43% more tokens gets baseline to the
same loss. The same holds for looped iso-param depth at 0.63x.

That reframes the whole branch. The speed campaign (94.9 -> 199.8 TFLOP/s) was
worth doing precisely because it is what makes this comparison close -- at the
original 94.9 TFLOP/s mHC would have lost the wall-clock comparison badly. But
the architecture itself buys nothing on the speed-Pareto at this scale.

### What this does not rule out
* **Parameter efficiency.** loop6x2_ngpt reaches 2.727 against baseline's 2.721
  with 26% fewer parameters. Worthless for training throughput, potentially real
  for memory-bandwidth-bound decoding, which this sweep does not measure.
* **Scale.** Everything here is 130M / 6B tokens on one node. A scheme whose
  advantage grows with size would be invisible at this scale. Testing that needs
  a size ladder, not more variants at 130M.
* **The iso-FLOP looped arms are settled negatively though**: loop6x2_ngpt is
  strictly dominated by ngpt_analytic (same throughput to 0.2%, worse loss), and
  loop6x2_plain pays 0.037 of loss for 1.4% more speed. Looping holds FLOPs fixed
  and removes parameters, and parameters are close to free on the training clock.
