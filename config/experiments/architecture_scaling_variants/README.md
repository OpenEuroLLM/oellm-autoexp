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

## Active multilingual handoff — 2026-07-31

This section supersedes the older monolingual execution notes above for the
current JUPITER work. Do not commit the current changes: the user explicitly
asked to defer commits.

### Production experiment

- Config: `multilingual/fullattn_mlstm7_gdn7_0.1B_50BT.yaml`; output group
  `architecture_scaling_variants_multilingual_7to1_liger_mbs32` (a fresh
  namespace because the older group has inaccessible projectnucleus symlinks).
- DAG: four independent 6B-token training stages (`final6BT`) followed by one
  dependent `datamix4-val` evaluation stage (`eval6BT`) per architecture —
  eight points and four dependency edges. A local dry run on 2026-07-31
  rendered all eight successfully.
- Variants: FullAttn + RoPE (theta 10k), 7:1 mLSTM + NoPE, 7:1 GDN + NoPE,
  and 7:1 SWA (window 1024) + RoPE (theta 100k). At the 16-layer 0.1B size,
  `*_freq: 8` means 14 hybrid/local layers and two full-attention layers.
- Science recipe: preserve **GBS=128** and **LR=5e-4**; 6B tokens; 2k warmup;
  linear WSD cooldown across the final 2,289 of 11,445 iterations; training
  split `100,0,0`; common external `datamix4-val` evaluation after annealing.
- Performance recipe: one JUPITER node (DP=4), MBS=32, TP=1, BF16, local CUDA
  graphs, current `jupiter_liger` container, Liger fused LM-head/CE with an
  explicit 8,192-token chunk. This leaves GBS unchanged while replacing the
  older four-node/MBS=4 draft with the validated high-throughput geometry.
- Known bad node: `jpbo-001-48` is excluded because the otherwise validated
  Apptainer image exposed zero CUDA devices there.
- First submission (2026-07-31): SWA train `1140566`, FullAttn train
  `1140567`, GDN train `1140568`, and mLSTM train `1140569`. **All four FAILED
  at iteration 1600** — the first periodic-evaluation step. Root cause: the
  training stage uses `split: 100,0,0` (no validation data), but Megatron sets
  `do_valid = valid_dataloaders is not None and (full_validation or eval_iters
  > 0)` (`training.py:4283`). With `eval_iters=100` inherited from `ml_base`,
  `do_valid` was forced `True` despite the empty validation dataloader, so the
  eval at iter 1600 dereferenced a `None` iterator and crashed every rank with
  `TypeError: 'NoneType' object is not an iterator`. Training itself was
  healthy up to that point (323 TFLOP/s/GPU, loss ~3.9).
- Fix: the `final6BT` training stage now sets `backend.megatron.eval_iters: 0`,
  which makes `do_valid` False and disables the in-loop evaluation entirely.
  The separate `eval6BT` stage keeps its own non-zero `eval_iters`
  (`204800 // gbs = 1600`) and runs the `datamix4-val` blend after annealing,
  so validation coverage is unchanged.
- Resubmission (2026-07-31, after fix): FullAttn train `1143970`, GDN train
  `1143971`, mLSTM train `1143972`, SWA train `1143973`. Their evaluation
  stages remain pending until the matching `iter_0011445` checkpoint exists.
- Confirmed healthy (2026-07-31, ~26 min in): all four passed iteration 1600
  (the previous crash point) with zero NoneType/Traceback/OOM/FATAL errors and
  zero nan/skipped iters. Steady-state at that point: FullAttn iter 2130 loss
  3.66 @ 306 TFLOP/s/GPU; mLSTM iter 2170 loss 3.67 @ 260; GDN iter 1840 loss
  3.76 @ 228; SWA iter 2440 loss 3.59 @ 353 — matching the earlier throughput
  benchmark ranking.

### Completed throughput evidence

All values are steady-state, per GPU, on one JUPITER GH200 node at MBS=32 /
GBS=128; full details are inline in
`multilingual/throughput_four_models_liger_0.1B.yaml`.

| Variant | TFLOP/s/GPU | Tok/s/GPU | Job | Interpretation |
|---|---:|---:|---:|---|
| FullAttn | 323.6 | 197.8k | 1138460 | Production candidate |
| mLSTM 7:1 | 260.4 | 191.2k | 1138461 | Nearly the same token rate; lower FLOP estimate is architecture-aware |
| GDN 7:1 | 231.9 | 168.8k | 1138463 | Genuine kernel/memory penalty; 99.7% memory use |
| SWA 7:1 | 355.3 | 217.1k | 1138462 | Production candidate |

Do **not** use the MBS=64 recomputation retry:
`multilingual/throughput_rnn_hybrids_liger_recompute_0.1B.yaml` completed but
regressed to 235.5 TFLOP/s/GPU (mLSTM, job 1138571) and 195.7 (GDN, 1138570).
The ranks and validation finished cleanly; only their hung outer Slurm wrappers
were cancelled to release allocations. Other full-attention A/B outcomes are
recorded in `multilingual/throughput_liger_optimization_sweep_0.1B.yaml`:
explicit Liger chunk 8192/MBS32 is best; MBS48, MBS64+recompute, TE op-fuser,
and DDP overlap did not improve it.

The reported TFLOPs use Megatron's architecture-aware FLOP estimator. Compare
Tok/s/GPU across FullAttn and recurrent hybrids when judging actual speed; a
fixed 300-TFLOP threshold is not architecture-neutral.

### Operational procedure

1. Sync only changed paths with `sync_to_jupiter.sh` or a targeted `rsync -ruR`
   to `/e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp-hybrid`.
2. On JUPITER, submit with `UV_CACHE_DIR=/e/project1/e-sta-openeurollm/poeppel1/.cache/uv`
   and `PYTHONPATH=.:submodules/Megatron-LM`:
   `uv run python scripts/run_autoexp.py --submit-and-exit --config-name experiments/architecture_scaling_variants/multilingual/fullattn_mlstm7_gdn7_0.1B_50BT`.
3. Monitor all four train jobs to completion and ensure the four eval jobs start
   only after their matching final checkpoint appears. Outer Slurm wrappers may
   linger after all ranks exit; confirm the final iteration and validation are
   in the log before cancelling a stuck wrapper solely to release resources.
4. Write new job IDs, final losses, and any retry cause into
   `experiment/003_multilingual_throughput_tuning/log.md` and this handoff
   section. Preserve user changes and do not commit unless explicitly asked.
