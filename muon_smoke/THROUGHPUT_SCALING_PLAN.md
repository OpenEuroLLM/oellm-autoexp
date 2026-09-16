# Muon throughput → GPU-hour estimate (for grant proposal)

Goal: turn real, measured throughput numbers on Capella into a defensible
GPU-hour budget for a Muon scaling-laws study.

## Method

- **Per-GPU throughput**: measured directly, real Nemotron-tokenized data,
  real Qwen3 architecture, Muon optimizer, at 4 model sizes
  (`06_qwen3_throughput_1gpu.sbatch`).
- **DP scaling efficiency**: measured directly at 1 vs. 4 GPUs, same model
  and batch size (`07_qwen3_throughput_4gpu.sbatch`, job 3876406) — not
  assumed.
- **GPU-hour formula**: `GPU-hours = Tokens_target / (Throughput_per_GPU ×
  ScalingEfficiency) / 3600`. This is the *total compute cost* — it doesn't
  depend on how many GPUs you actually run on, only on throughput and
  efficiency. Wall-clock time at N GPUs = `GPU-hours / N`.
- **Token budgets**: the 9-point grid already defined in the
  `multilingual_scaling` model configs' `aux.center_tokens_set` — 6B, 12B,
  20B, 30B, 50B, 80B, 120B, 200B, 300B tokens.

## Raw throughput (measured, steady-state mean over 70 iterations, warmup discarded)

| Model | Params | micro_batch_size | Tokens/s/GPU | TFLOP/s/GPU | MFU (H100 989 TFLOPS bf16 peak) |
|---|---|---|---|---|---|
| Qwen3-dense 0.1B | 97M | 4 | 98,495 | 77.2 | 7.8% |
| Qwen3-dense 0.1B | 97M | 16 | 98,378 | 77.1 | 7.8% |
| Qwen3-dense 0.2B | ~200M | 4 | 82,710 | 134.4 | 13.6% |
| Qwen3-dense 0.4B | ~400M | 4 | 56,593 | 206.9 | 20.9% |
| Qwen3-dense 0.9B | ~900M | 4 | 44,191 | 279.3 | 28.2% |

**DP scaling efficiency** (job 3876406, 0.1B/mbs=4, 4×H100 vs. 1×H100):
82,751 / 98,495 = **84.0%** — real overhead from Muon's
`LayerWiseDistributedOptimizer` all-gather sync every iteration. Not yet
measured beyond 4 GPUs single-node (real multi-node validation is separate,
unbuilt infrastructure — see Caveat).

## Compute table — final numbers for the proposal

GPU-hours per (model size, token budget) cell, 84.0% measured efficiency
applied throughout.

| Model | tok/s/GPU (1 GPU) | 6B | 12B | 20B | 30B | 50B | 80B | 120B | 200B | 300B | Row total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1B | 98,495 | 20 | 40 | 67 | 101 | 168 | 269 | 403 | 671 | 1,007 | **2,746** |
| 0.2B | 82,710 | 24 | 48 | 80 | 120 | 200 | 320 | 480 | 800 | 1,199 | **3,270** |
| 0.4B | 56,593 | 35 | 70 | 117 | 175 | 292 | 467 | 701 | 1,169 | 1,753 | **4,780** |
| 0.9B | 44,191 | 45 | 90 | 150 | 224 | 374 | 599 | 898 | 1,497 | 2,245 | **6,121** |

**Grand total: 16,918 GPU-hours** (all 4 model sizes × all 9 token budgets).

**Wall-clock at 32 GPUs**: 16,918 / 32 = **529 hours (≈22.0 days)** for the
full grid.

## Key findings

- **MFU climbs steadily with model size**: 7.8% (0.1B) → 28.2% (0.9B) of H100
  peak. Bigger models are meaningfully more GPU-efficient, not just more
  expensive per token — worth factoring into which sizes the study
  prioritizes.
- **Batch size plateaus early**: 0.1B showed identical throughput at
  `micro_batch_size=4` and `16` (77.2 vs 77.1 TFLOP/s/GPU) — no benefit past
  a small batch size at this model scale. Model size is the real lever here,
  not batch size.
- **DP scaling has a real, non-trivial cost**: 84% efficiency at just 4 GPUs
  single-node means ~16% of compute is lost to optimizer-sync overhead
  already at small scale — worth stating explicitly in the proposal rather
  than assuming near-linear scaling.

## Caveat: real multi-node numbers

The 84% efficiency figure is measured at 4 GPUs, single node. It has not
been measured beyond that — genuine unknown whether it holds, improves, or
degrades further at real multi-node scale (8, 16, 32 GPUs). Getting that
would require building real multi-node SLURM support for Capella first —
new, untested infrastructure (dynamic `MASTER_ADDR` resolution across nodes,
cross-node NCCL, no existing Capella cluster target in this repo to build
on). Not done here given the compute budget for this investigation; state
the 84%-at-4-GPU figure as measured, and multi-node behavior as an open
question / follow-up in the proposal.
