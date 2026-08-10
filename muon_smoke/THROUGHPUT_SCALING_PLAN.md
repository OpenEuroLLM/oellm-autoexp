# Muon throughput → GPU-hour estimate (for grant proposal)

Goal: turn a small number of real, measured throughput numbers on Capella
into a defensible GPU-hour budget for a Muon scaling-laws study, without
needing to actually run at full scale first.

## Method

Three axes, kept small and empirical:
1. **GPU count** (single node only — 1, 2, 4 H100s; all proven working today,
   no new infra needed).
2. **Micro batch size** (2, 8, 32 say — find where throughput plateaus before
   OOM; theoretical footprint at micro_batch=2 was only ~3.2GB/80GB, so there's
   real headroom unexplored).
3. **Model size** (0.1B now; 0.4B/0.9B as budget allows — larger models
   typically show *better* MFU since compute-per-parameter grows faster than
   fixed overhead, worth having at least one bigger data point).

From (1)+(2) at fixed model size: real per-GPU throughput (tokens/sec/GPU),
measured, not estimated.

From single-node DP scaling (1→4 GPU): a real, measured **DP scaling
efficiency** number — how close to linear speedup you actually get going from
1 to 4 GPUs on this cluster with this model/optimizer. This is the number
that lets you defensibly extrapolate to N nodes *without* having built real
multi-node infrastructure (which is separate, harder, unverified work — see
caveat at the bottom).

**GPU-hour formula:**

```
GPU-hours = Tokens_target / (Throughput_per_GPU × N_GPUs × ScalingEfficiency) / 3600
```

## What's already measured (real, today)

| Config | Steady-state throughput | Notes |
|---|---|---|
| 0.1B, 1×H100, micro_batch=2, global_batch=32 | ~50,000–65,000 tok/s/GPU | job 3846248, `RESULT_qwen3.md` |
| 0.1B, 4×H100 (dp=4), same batch config | ~45,000–62,000 tok/s/GPU *per device* | job [4-GPU run] — near-linear scaling observed, but noisy/short (50 iters), treat as preliminary |

Both runs used `micro_batch_size: 2` — nowhere near this model's memory
ceiling, so these are **not yet the model's best achievable throughput**,
just what the smoke tests happened to use.

## Worked example (illustrative, using today's preliminary numbers)

Target: smallest token budget already baked into the `multilingual_scaling`
model configs' `aux.center_tokens_set` — 6B tokens, at 64 GPUs (16 nodes),
assuming a conservative 90% scaling efficiency beyond one node:

```
GPU-hours = 6e9 / (58,000 × 64 × 0.9) / 3600 ≈ 32 GPU-hours
```

For the full 9-point token-budget set already defined in the model configs
(6B, 12B, 20B, 30B, 50B, 80B, 120B, 200B, 300B), the same formula scales
roughly linearly with tokens — sum them for a total per-model-size GPU-hour
ask, then multiply across however many model sizes the scaling-law study
needs.

**This example uses noisy, unoptimized (micro_batch=2) numbers — treat it as
a placeholder for the shape of the calculation, not final numbers for the
proposal.**

## Proposed next runs (to get real numbers to plug in)

Small, scoped, single-node only — no new infrastructure:
1. Micro-batch sweep at 0.1B, 1×H100: micro_batch = 2, 8, 32 (or until OOM).
   Find the throughput plateau.
2. Same sweep at 4×H100, to get a *real* DP scaling-efficiency number instead
   of the current 1-datapoint estimate.
3. Repeat (1) at one larger model size (0.4B or 0.9B) to get a second point
   on the "MFU improves with model size" curve.
4. Discard first ~5 iterations of every run (compile/warmup skews the mean
   badly — iteration 1 took 19.5s vs ~0.3s steady-state in the 4-GPU test),
   run ≥100 iterations, checkpoint saving off (separate cost, skews timing).

That's on the order of 6-8 short runs, each a few minutes on Capella.

## Caveat: real multi-node numbers

Everything above extrapolates from single-node (≤4 GPU) data using an
assumed scaling efficiency. If the proposal needs *measured* multi-node
throughput rather than an extrapolation, that requires building real
multi-node SLURM support for Capella first — genuinely new, untested
infrastructure (dynamic `MASTER_ADDR` resolution across nodes, cross-node
NCCL, no existing Capella cluster target in this repo to build on). Scoped as
a separate, bigger task if the extrapolation isn't credible enough for the
reviewers.
