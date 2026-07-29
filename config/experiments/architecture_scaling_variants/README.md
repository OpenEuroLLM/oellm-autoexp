# Architecture-Variant Scaling Rerun — Sliding-Window Attention

> **NOTE (2026-07-27):** the study has moved to the more recent MULTILINGUAL
> baseline (`dense_multilingual_scaling`) — see `multilingual/README.md` for
> the current <10k GPU-hr plan. The monolingual configs below remain valid
> against the `dense_scaling` (Nemotron/gpt-neox) baseline.

Implements the retraining plan in `dump/scaling_experiments/retraining_plan.md`
(run list: `dump/scaling_experiments/recommended_runs.csv`) for the first
architecture variant: **sliding-window attention (SWA) on >3/4 of the layers**,
targeting a large KV-cache reduction for inference-optimized training.

Goal: show "this architecture is also good and scales the same" — same data,
tokenizer, token order, optimizer and WSD schedule as the `dense_scaling`
baseline; only the attention pattern changes.

## The variant (`swa6w1024`)

- `window_size: 1024,0` — causal sliding window of 1024 at seq 4096.
- `window_attn_skip_freq: 6` — Megatron keeps **full** attention on every 6th
  layer (`layer % 6 == 0`, 1-based) and uses SWA on the rest (Gemma-3-style
  5:1 hybrid):

  | size | layers | SWA layers | full layers | SWA fraction |
  |------|-------:|-----------:|------------:|-------------:|
  | 50M  | 12     | 10         | 2 (6,12)    | 83% |
  | 130M | 18     | 15         | 3           | 83% |
  | 300M | 20     | 17         | 3 (6,12,18) | 85% |
  | 600M | 20     | 17         | 3           | 85% |
  | 1B   | 24     | 20         | 4           | 83% |

  KV cache at 4k context: ~2.7× smaller; asymptotically ~6× at long context.

Other hybrid variants: override `aux.swa_window` / `aux.swa_full_attn_freq`
(and `aux.variant` for naming) — everything else stays identical.
`window_attn_skip_freq` also accepts an explicit per-layer pattern string,
but prefer the integer form (the list expression contains shell
metacharacters and travels through the sbatch command line).

## What is matched to the baseline (do not touch)

- Data: Nemotron-CC "high" 15% downsample, pre-tokenized gpt-neox-20b, JUPITER
  copy (`/backend/megatron: data_nemonicco_split_jupiter` — same binaries as
  the LUMI original), `split: 99,1,0` ⇒ same deterministic 1% validation slice.
- `seed 1234`, `dataloader_type: single` ⇒ identical token order per (gbs,
  budget) as the baseline runs. (The older `korbi/megatron_niccolo_130M_scaling_jupiter.yaml`
  used `cyclic` + a separate valid set — intentionally NOT copied here.)
- Tokenizer `EleutherAI/gpt-neox-20b`, `padded_vocab_size 50304`, seq 4096,
  tied embeddings, MHA, qk-layernorm, qkv-bias, RMSNorm, RoPE 10000,
  init 0.02, bf16.
- Optimizer: adam β=(0.9, size-dependent β2 0.99/0.95), eps 1e-8, wd 0.1,
  clip 1.0. WSD: 2000-iter warmup, constant lr, linear decay over the final
  20% of the budget to min_lr 1e-5.
- Ground truth for every setting: `dump/scaling_experiments/reference_configs/*.json`.

## WSD ladder mechanics (per config file)

Each file is one ladder: a **stable** run at constant lr to 0.8 × D_max, plus
one **cooldown** job per budget D that waits (FileExistsCondition) for the
stable checkpoint at iter(0.8·D) (`save_extra_steps` puts a checkpoint exactly
there), branches from it (`ckpt_step` + `override_opt_param_scheduler`), and
decays linearly over the final 20% of iters. Cooldowns are cancelled if the
stable job dies. Note: a preempted/crashed cooldown restarts from the branch
checkpoint (its `load` points at the stable dir) — deterministic, just redoes
the decay.

## Files / submission order

Submit from JUPITER (`ssh jupiter "bash ~/work/Projects/oellm-autoexp/submit.sh
--config-name experiments/architecture_scaling_variants/<name> --submit-and-exit"`).

0. `swa_smoke_130M` — **run first**: verifies SWA args work in the container
   (TE/flash sliding-window support on GH200), data/tokenizer offline, speed.
1. Phase 1 (~720 GPU-hrs): `swa_{50M,130M,300M}_tuning` — 3×3 (gbs, lr) grids,
   budgets 6/12/20B. Checks the baseline optima transfer to SWA. If the optima
   shift, adjust the Phase 2 ladder combos before submitting them.
2. Phase 2 (~6.2k GPU-hrs): `swa_{50M,130M,300M,600M,1B}_{low,high}` — the
   main 5 sizes × 9 budgets grid.
3. Phase 3 (~3.6k GPU-hrs, only if budget allows / Phase 1 ambiguous):
   `swa_{600M,1B}_high_lr0.0005` — lr insurance at the costliest cells.

`slurm.sbatch.nodes` / `aux.mbs` per file are throughput choices for GH200
(gbs must stay divisible by mbs × 4 × nodes); changing them does NOT affect
the science — only `aux.gbs` and lr matter.

## Analysis

- Metric: final `lm loss validation` per (size, budget) cell vs
  `dump/scaling_experiments/post_decay_losses.csv`; then fit loss(N, D) on
  both grids and compare exponents. Differences < ~0.005 nats ≈ noise.
- W&B project `architecture_scaling_variants` (offline on JUPITER — sync
  after runs); reuse `dump/scaling_experiments/export_dense_summaries.py` /
  `extract_post_decay_losses.py` pointed at the new project.
- Caveat for interpretation: FLOPs/token is slightly lower than baseline on
  SWA layers (attention over ≤1024 instead of causal-avg 2048 keys) — for
  compute-optimal fits, recompute FLOPs/token; loss-vs-tokens comparisons are
  unaffected.
