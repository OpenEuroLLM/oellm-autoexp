# Residual architectures and looped depth — results

130M dense transformer, 6.0B tokens, 4× GH200 (one node), JUPITER.

Two experiment groups feed this document. The **speed** sweeps live in this
directory (`residual_arch/*.yaml`, 60-iteration throughput probes); the **science**
sweeps live in `../residual_scaling_variants/` and produced every loss number
below.

## How to read these numbers

**Standard budget.** Every arm in the main table ran exactly:

| | |
|---|---|
| steps | 11,445 |
| global batch | 128 |
| learning rate | 0.002 (WSD, 2000 warmup, 20% decay tail) |
| sequence | 4096 → **6.0B tokens** |
| geometry | 18 layers, hidden 576, ffn 2304, 9 heads × 64, vocab 50304 (tied) |
| parallelism | TP=1, PP=1, DP=4 |

Verified from consumed-sample counts in the logs, not from config. Arms that ran
at other budgets were rerun; the superseded runs are not quoted anywhere here.

**Noise floor ≈ 0.015.** Three baselines at 16,320 / 18,046 / 21,648 steps came
out 2.6662 / 2.6814 / 2.6506. More tokens cannot raise the loss, so the middle run
is ~0.015 out of place. **Treat any difference below ~0.015 as unresolved.** These
are single runs; seed repeats are the cheapest outstanding work.

**Metric** is validation loss at the final iteration. Training loss and
intermediate checkpoints of longer runs are not comparable — see Pitfalls.

## Main table — 6.0B tokens, matched budget

`tok/s/GPU` is wall-clock derived: `gbs × seq / ms_per_iter / 4`.

| arm | params | tok/s/GPU | final |
|---|---|---|---|
| loop12x2_ngpt | 124.5M | 145,830 | **2.663** |
| mhc4 | 126.5M | 193,750 | **2.665** |
| loop12x2_plain | 124.5M | 173,559 | 2.678 |
| attnres4_lerp | 124.5M | 134,185 | 2.682 |
| attnres8_lerp | 124.5M | 152,445 | 2.689 |
| mhc4_lerp_scalers | 126.5M | 123,049 | 2.699 |
| ngpt_scalers | 124.5M | 197,249 | 2.700 |
| mhc2_lerp | 124.8M | 191,850 | 2.703 |
| ngpt_analytic | 124.5M | 231,781 | 2.708 |
| ngpt_approx | 124.5M | 218,453 | 2.715 |
| **baseline** | 124.5M | **276,290** | **2.721** |
| loop6x2_ngpt | 92.7M | 231,005 | 2.727 |
| attnres4 | 124.5M | 226,455 | 2.737 |
| attnres8 | 124.5M | 149,200 | 2.741 |
| loop6x2_plain | 92.7M | 279,471 | 2.758 |
| loop3x4_ngpt | 76.8M | 234,057 | 2.770 |
| ngptpp *(untied)* | 153.5M | 187,998 | 2.775 |
| loop3x4_plain | 76.8M | 273,637 | 2.791 |
| ngptpp_tied | 124.5M | 184,219 | 2.859 |
| mhc4_lerp_scalers_wnorm | 126.5M | 120,570 | 2.859 |
| muown | 124.5M | 269,862 | 2.871 |
| ngpt_wnorm | 124.5M | 226,064 | 2.874 |
| attnres4_lerp_wnorm | 124.5M | 195,864 | 2.887 |
| ngpt_scalers_wnorm | 124.5M | — | **NaN at it7600** |

`ngptpp` is the only arm not parameter-matched: it runs untied embeddings, so it
carries 28.97M more parameters and cannot be ranked against the rest.

## Residual architectures

**mHC (arXiv:2512.24880) is the strongest single scheme: +0.056 over baseline.**
n=4 streams mixed by a Sinkhorn-projected doubly stochastic matrix. Costs 1.6% in
parameters (1.99M, almost all the fused dynamic projection `phi`), 30% in
throughput, and runs at 99.6% memory occupancy with no micro-batch headroom.
n=2 composed with the LERP reaches 2.703.

**Block AttnRes only works with the LERP.** Alone it is *worse* than baseline at
both block counts (2.737 / 2.741); composed with the nGPT LERP it is second and
third best (2.682 / 2.689). The residual scheme is not carrying that result.

**nGPT's LERP residual is a small win: +0.013.** Analytic (exact cos θ) beats the
`approx` factor by 0.007. Adding the s_qk/s_uv scaler set is worth a further
0.008 (2.700).

**The weight normalizer costs 0.14–0.15 and nothing recovers it.** Every arm
carrying it lands 2.859–2.887. Not the token fix, not the learning-rate fix, not
matching canGPT's configuration (`ngptpp_tied`, 2.859, which uses canGPT's
`approx` LERP, `base_scale` 0.01, `out_norm_dim_0=false` and no scaler LR boost).
`ngpt_scalers_wnorm` diverged outright: healthy to iteration 7550 (loss 3.121,
grad norm 0.367), grad norm NaN at 7600. It ran `ngpt_scaler_lr_mult: 4.0`, that is
an effective 0.008 on the scalers, which on a weight-normalized network carry all
the magnitude. canGPT's own SFT config gives the scalers no LR boost at all.

**mHC and nGPT do not compose.** Every nGPT component is either harmful with mHC
(weight normalizer −0.19, LERP −0.033) or has no mechanism to help: the scalers
exist to reinject magnitude that normalization removes, and mHC normalizes
nothing — it constrains the *mixing operator* and leaves stream magnitude free.

**Muown** (Muon on row directions + sign step on magnitudes) is 0.15 behind
baseline at 2.871.

## Looped (weight-tied) depth

Prelude / block × iterations / coda over the 18-layer geometry, 3 layers at each
end. `num_layers` counts *unique* layers. Implemented as
`build_loop_schedule()` in `megatron/core/transformer/transformer_block.py`.

| arm | structure | unique | effective | params | final |
|---|---|---|---|---|---|
| loop12x2_ngpt | 3 + (12)×2 + 3 | 18 | 30 | 124.5M | **2.663** |
| loop12x2_plain | 3 + (12)×2 + 3 | 18 | 30 | 124.5M | 2.678 |
| loop6x2_ngpt | 3 + (6)×2 + 3 | 12 | 18 | 92.7M | 2.727 |
| loop6x2_plain | 3 + (6)×2 + 3 | 12 | 18 | 92.7M | 2.758 |
| loop3x4_ngpt | 3 + (3)×4 + 3 | 9 | 18 | 76.8M | 2.770 |
| loop3x4_plain | 3 + (3)×4 + 3 | 9 | 18 | 76.8M | 2.791 |

**Parameters dominate the ordering.** 124.5M → 92.7M → 76.8M maps onto
2.66–2.68 → 2.73–2.76 → 2.77–2.79. Loop topology barely matters beside how much
capacity was kept.

**At fixed parameters, looping is a large win.** `loop12x2_ngpt` reaches 2.663
against baseline's 2.721 at the same 124.5M — **+0.058**, the largest
architectural effect measured anywhere in this branch. It buys that with 1.67×
the FLOPs, which is the right trade when memory-limited rather than
compute-limited.

**At fixed FLOPs, looping is a loss.** `loop6x2_plain` runs at 1.01× baseline
speed (matched tokens *and* wall-clock) and finishes 0.037 behind;
`loop3x4_plain` is 0.070 behind. Looping holds FLOPs fixed and removes
parameters, and parameters are nearly free on the training clock.
`loop6x2_ngpt` is strictly dominated by `ngpt_analytic` — same throughput to
0.2%, worse loss.

### Where the recurrence sits (`loop_group_size`)

Identical parameters (92.7M), FLOPs and effective depth. Only the order in which
the block's six layers are applied changes:

```
group 6:  3 4 5 6 7 8 | 3 4 5 6 7 8      outer recurrence
group 3:  3 4 5 3 4 5 | 6 7 8 6 7 8
group 2:  3 4 3 4 | 5 6 5 6 | 7 8 7 8
group 1:  3 3 | 4 4 | 5 5 | 6 6 | ...    each layer repeated in place
```

| group | plain | nGPT | LERP gap |
|---|---|---|---|
| 1 | 2.7685 | 2.7597 | 0.009 |
| 2 | 2.7524 | 2.7455 | 0.007 |
| 3 | 2.7512 | 2.7375 | 0.014 |
| 6 | 2.7579 | **2.7269** | **0.031** |

**Plain residuals are indifferent to placement** — spread 0.018, no ordering,
against a 0.015 noise floor. **nGPT is monotone**: 2.7597 → 2.7455 → 2.7375 →
2.7269, with the LERP's advantage growing from 0.009 to 0.031.

**Practical: use outer recurrence.** It is free — every cell costs the same — and
worth ~0.02 with nGPT.

### The magnitude-invariance hypothesis is refuted

The motivating argument was that nGPT's LERP holds ‖x‖ constant with depth, so a
tied block sees inputs from the same distribution at every iteration and
repetition becomes well-posed. That predicts the LERP helps *most* where
repetition is tightest. Two independent axes say the opposite:

* **Group size**: LERP worth 0.009 at group 1 (a layer consuming its own output),
  rising monotonically to 0.031 at group 6 (five layers intervene).
* **Loop count**: gaps of 0.021 / 0.031 / 0.015 for 3×4 / 6×2 / 12×2 — no
  relation to how many times a block repeats.

The LERP does help looped models, reliably, by ~0.03 at the best topology. It is
not doing so by stabilising repeated application of shared weights.

## Speed-Pareto: matched wall-clock

Every arm above ran matched *tokens*. The arms that beat baseline all run slower,
so baseline was rerun with schedules sized to their wall-clock:

| arm | elapsed | final | baseline reference | elapsed | final |
|---|---|---|---|---|---|
| mhc4 | 2:13:00 | 2.6654 | 16,320 steps | 2:15:38 | 2.6662 |
| loop12x2_plain | 2:25:48 | 2.6782 | 18,046 steps | 2:25:27 | 2.6814 |
| loop12x2_ngpt | 2:55:18 | 2.6628 | 21,648 steps | 2:52:16 | 2.6506 |

All three are inside the 0.015 noise floor, so **on matched wall-clock no scheme
is distinguishable from baseline trained longer.**

**Read this as the bar kernel work must clear, not as a verdict.** Iteration time
is an engineering variable and effort here was very uneven: mHC received a
dedicated campaign (94.9 → 199.8 TFLOP/s, no model change), while the looped arms
ran with **CUDA graphs disabled entirely**. Graphs are worth ~20% at this size.
On that reading:

| arm | speedup needed to win on wall-clock |
|---|---|
| mhc4 | none, already at parity |
| loop12x2_plain | none, already at parity |
| loop12x2_ngpt | **1.22×** |

### Why the looped arms had no graphs, and the fix (2026-08-11)

This section previously gave two different explanations, both written from
reading rather than from a run. The measured one:

**Runner selection was never the problem.** At PP=1 — which looping requires —
`reuse_cudagraphs` is true (`cuda_graphs.py:1519`), a runner is chosen by
scanning for the first `FWD_READY` one (`:1566`), and a forward flips it to
`BWD_READY` (`:1709-1711`). N calls to one module capture N runners and replay
them in capture order, exactly as gradient accumulation already relies on. So
the original guard's stated reason — "keys graphs per module and assumes one
call per module per step" — was not the mechanism.

**The blocker was one level down, in the input-buffer chaining.**
`_resolve_input_buffer` (`:895-914`) marks a captured input `can_skip_replay_copy`
when the graph's input buffer *is* the buffer its producer wrote into, and replay
then skips the copy and asserts pointer equality (`:611`). That holds only while
each module occupies one fixed position in the layer chain. A looped module sits
at several depths, so at replay its input legitimately arrives in a different
allocation and the assertion fires. Removing the guard without this made all
three looped arms die on `assert user_input.data_ptr() == cudagraph_input.data_ptr()`.

**Fix:** force `can_skip_replay_copy = False` when a loop schedule is active, so
replay copies into the graph buffer instead of asserting. One hidden-state copy
per graphed call, and inert for every non-looped model (`loop_block_layers` is 0).

**Measured, 120-iteration probes on the mLSTM hybrid grid:**

| arm | no graphs | graphs | speedup |
|---|---|---|---|
| d18 (dense) | 630 ms | 477.3 | 1.32× |
| d12 (dense) | 589 | 401.7 | 1.47× |
| d9 (dense) | 569 | 405.8 | 1.40× |
| loop6x2 | 631 | 480.1 | **1.31×** |
| loop3x4 | 633 | 488.3 | **1.30×** |
| loop12x2 | 756 | 761.0 | **1.00× — unexplained** |

Numerics check: 12 iterations, graphs on vs off, per iteration. Iterations 1-2
bit-identical; max |Δloss| over 11 iterations is 4.2e-3 (loop6x2), 8.0e-4
(loop12x2), 3.6e-4 (loop3x4), against **9.5e-4 for dense d18**, which the change
cannot affect. Ordinary reduction-order drift, not a numerics bug.

Activation recompute needed a real fix, not just a lifted guard: the checkpointed
forward chunked `range(num_layers)` over `self.layers`, which under a schedule
applies the block once instead of once per iteration — a different model that
trains happily. It now walks the schedule. Both paths are covered in
`tests/unit_tests/transformer/test_loop_schedule.py`.

So the 1.22× above is reachable for `loop6x2`/`loop3x4`-style topologies but
**not yet demonstrated for `loop12x2`**, the arm that actually needs it.

## Throughput

Measured on the 60-iteration probes in this directory. Optimisation work is
summarised in `../../../dump/residual_arch_final.md`; headline items were
torch.compile globally (+19–26%), Liger fused linear+CE, fusing the mHC stream
read/write (94.9 → 190.2), and a hand-written Triton write backward.

| arm | tok/s/GPU | vs baseline |
|---|---|---|
| loop6x2_plain | 280,068 | 1.01× |
| baseline | 276,290 | 1.00× |
| loop3x4_plain | 273,637 | 0.99× |
| muown | 269,862 | 0.98× |
| ngpt_analytic | 231,781 | 0.84× |
| loop6x2_ngpt | 231,005 | 0.84× |
| mhc4 | 193,750 | 0.70× |
| loop12x2_plain | 173,559 | 0.63× |
| loop12x2_ngpt | 145,830 | 0.53× |
| mhc4_lerp_scalers | 123,049 | 0.45× |

Memory at mbs=32: baseline 0.737, ngpt_analytic 0.840, attnres8 0.775,
**mhc4 0.996** (no headroom — this is what caps mHC's micro-batch and forces
CUDA graphs off).

## Pitfalls found the hard way

Each of these produced a wrong published number before being caught. Worth
knowing before running anything else in this harness.

1. **`train_iters` derived from `aux.gbs`, not the arm's own batch.** Five arms
   overrode `global_batch_size` to 384 without touching `aux.gbs`, so they trained
   on 18.0B tokens against everyone else's 6.0B and mhc4 looked 0.145 ahead of
   baseline. Fixed in commit `fc0ede3`; both `train_iters` and `aux.total_tokens`
   now read `${.global_batch_size}`.
2. **Learning-rate split.** The nGPT-geometry arms ran at lr 0.001 while baseline
   ran 0.002, worth up to **0.117** — more than any architectural effect measured
   here. Two conclusions died on the rerun: "the nGPT LERP is a wash" (it is
   +0.013 ahead at matched lr) and the mhc4+LERP penalty (0.080 → 0.033).
3. **Training loss quoted as a final result.** mhc4's widely-quoted "2.638" was
   its *training* lm loss at iteration 10350 of a run that TIMEOUT'd and never
   finished. Only validation loss at `train_iters` is comparable.
4. **`check_for_nan_in_loss_and_grad: false`** (required for CUDA graph capture)
   means a diverged run exits 0 with a NaN result and lands in a results table as
   a number. Any collector must reject NaN and report the last finite iteration.
5. **Intermediate checkpoints of a longer run understate.** `lr_wsd_decay_iters`
   is 0.2 × `train_iters`, so an 18,046-step run is still at full LR at iteration
   11,445 and reads ~0.044 worse there than a properly-sized 11,445-step run.
   Iso-time references must be *sized*, not sampled from one long run.
6. **Silent feature no-ops.** New Megatron config fields need registering in
   `oellm_autoexp/backends/megatron/cli_metadata.py` (both `MegatronArgMetadata`
   and `MegatronActionSpec`) or the flags are silently dropped — the model builds
   and trains, just not the architecture you asked for. Six looped jobs ran as
   plain models before this was caught, detected only because throughput was
   impossibly high. `TransformerBlock` now logs `Looped depth active: … schedule=…`
   as positive confirmation.
7. **Early checkpoints mislead badly.** `loop3x4` showed a 0.802 nGPT gap at
   it2000 that collapsed to 0.035 by it4000. Gaps at the end of warmup predict
   nothing.
8. **A claim about the harness is not a measurement either.** Looping was blocked
   from CUDA graphs by a `NotImplementedError` justified from `CudaGraphManager`'s
   docstring. Reading the runner selection appeared to refute it, so the guard was
   removed — and all three looped arms then crashed on a `data_ptr` assertion one
   layer further down. Two explanations were published before a single run was
   done to check either. Both the guard and its removal deserved the scepticism
   the loss numbers get; the section above now records what a run actually showed.
9. **Check exit state before quoting throughput.** The first graphs-on numbers for
   `loop6x2`/`loop3x4` were read from logs of jobs that went on to fail — the
   arms trained a while, then died. `sacct` state, not the last log line.

## Open

1. **Seed repeats** (2 runs). Noise is ~0.015 and several conclusions rest on
   ~0.03 differences. Cheapest thing outstanding.
2. **Dense controls at 76.8M and 92.7M** (2 runs, faster than baseline). Needed to
   say whether looping beats a *smaller dense model* at equal parameters — the
   actual claim under memory constraints. The current grid does not test it.
   *In progress:* the hybrid grid in `../residual_scaling_variants/looped_hybrid_jupiter.yaml`
   runs these controls (`hybrid_d12`, `hybrid_d9`) on an mLSTM stack.
3. **Rerun the looped arms with CUDA graphs on.** Now possible — the guard is gone
   and the input-buffer chaining is fixed (see above). Measured at 1.30-1.31× on
   `loop6x2`/`loop3x4`. **But `loop12x2` gains nothing** (761.0 vs 756 ms) and it is
   the arm whose parameter win needs 1.22× to become a wall-clock win, so that
   case needs diagnosing before the rerun is worth the GPU-hours.
4. **~335 ms of depth-independent step cost**, even with graphs on. On the hybrid
   grid, 9 → 18 layers costs 72 ms of a 477 ms step, and `d12` (401.7 ms) is
   *faster* than `d9` (405.8). Fitting the dense points puts ~70% of the step in
   something that does not scale with depth. Graphs removed ~150 ms of it; the
   rest is unidentified and is the largest remaining throughput lever. Profile
   rather than guess — two guesses about this subsystem have already been wrong.
5. **PP>1 is implemented but never equivalence-checked** for mHC or Block AttnRes
   (PP=2 halves DP, so data order differs and a naive comparison fails). Treat as
   unverified.
6. **Scale.** Everything here is 130M / 6B tokens on one node. A scheme whose
   advantage grows with size is invisible at this scale; a size ladder, not more
   variants at 130M, is the way to find out.
