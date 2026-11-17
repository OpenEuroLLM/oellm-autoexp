# sweep100m: hyperparameter sweep at 100M active, WSD ladder per setting

Every setting gets a full WSD ladder: one stable run plus cooldown branches at 1x / 2x / 4x
Chinchilla (2.0B / 4.0B / 8.1B tokens), as in `../hyb100m/11_wsd_ladder.yaml`. The files are
generated from 11 by `gen_sweep100m.py` (edit the grid there and re-run it).

| File | Swept | Settings | Jobs |
|---|---|---|---|
| `adam.yaml` | batch size x lr x beta1 = beta2 | 4 x 5 x 3 = 60 | 240 |
| `muon.yaml` | batch size x lr x momentum | 4 x 5 x 3 = 60 | 240 |
| `muown_classical.yaml` | batch size x lr x momentum | 4 x 5 x 3 = 60 | 240 |
| `prebuild_index.yaml` | builds the one missing data index, once | | 1 |

## What is standard and what is ours

Reference: Swagatam's 100M "pure MoE baseline"
(`swagatam/alps/engram_scaling/100m_1b.yaml`, branch `feat/hybrid-v0.19-alps`).

- **Identical to the reference:** every model setting (4 x `LELELE*E`, width 640, 96 experts
  top-4, tied 262144-row embedding, 101,061,472 active parameters; no Engram layers), Megatron
  branch `oellm/v0.19-hybrid` (pinned commit), the team container, and the data loading:
  `create_attention_mask_in_dataloader: false`, `dataloader_type: cyclic`,
  `padded_vocab_size: 262144`; `mmap_bin_files: false` as the team's hybrid branch.
- **As the reference, unlike Korbinian's production run** (`korbi/production/hybrid_ambitious_*`,
  a different and much larger model): no document masking, no BOS per document, sequence
  length 4096, validation on a held-out split.
- **Ours (the study design):** the v8ml data mix (99 / 1 split), WSD schedule with cooldown
  branches, the optimizer grid below, micro-batch 16 with the Liger fused loss (speed only).

Why the loader settings matter (found 2026-11-16): with Megatron's default
(`create_attention_mask_in_dataloader: true`) each of the 522 datasets caches a 4096 x 4096
mask in every data-reading process and every micro-batch carries 16 of them. That filled the
node RAM in hyb100m/11 (146 GB allocated after 8 min vs 49 GB with it off; +57 GB vs +4 GB at
the first validation; `../smoke_test/14_memory_fix.yaml`, `15_memory_nomask.yaml`). The model
masks causally itself (`hybrid_layer_specs.py`); losses are identical with and without.

## Grid

| | Values |
|---|---|
| batch size (sequences of 4096 tokens) | 64, 128, 256, 512 |
| lr (all three optimizers) | 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3 |
| AdamW beta1 = beta2 | 0.9, 0.95, 0.975 |
| Muon momentum (`muon_momentum`) | 0.9, 0.95, 0.975 |
| muown_classical momentum (`--angular-muown-momentum`) | 0.9, 0.95, 0.975 |

One lr grid for all optimizers: Muon and muown_classical scale their matrix update by
0.2 x sqrt(max(rows, cols)) (Moonlight convention; set for Muon, the default for muown_classical),
which makes it the size of an AdamW update, so the same lrs apply. (hyb100m/11 ran Muon at 5e-3,
the LUMI value for batch 4096; that ladder is an extra point above this grid.)

beta / momentum: averaging horizon 1 / (1 - beta) = 10, 20, 40 steps, a factor of 2 like the lr
grid. Fixed: weight decay 0.1 (muown_classical 0.0), gradient clipping 1.0, seed 1234.

Things to know when reading the results:
- Muon and muown_classical use ONE lr for their matrices and for the Adam-handled parameters
  (embeddings, router, norms; Adam betas 0.95 / 0.95). Sweeping lr moves both.
- Muon runs with `--muon-batched-step`: same-shape matrices (e.g. the 96 experts of a layer) are
  orthogonalized in one batched GPU call instead of one by one. Each matrix is still treated
  separately (same maths up to floating-point summation order); it only removes the ~15 %
  per-step overhead Muon had in hyb100m/11. It is Megatron's default for muown_classical but
  not for Muon; checked against the unbatched step in `../smoke_test/16_muon_batched_check.yaml`.
- muown_classical: the momentum knob is for the directions; the row magnitudes are on Adam with
  betas (0.9, 0.95).
- Muon's weight decay (0.1) acts at its 5x higher lr; muown_classical has none.
- Each validation uses the next 2560 sequences of the validation split (the same ones for every
  run at the same token count); the cooldown jobs have their own validation index, so the final
  losses of 1x / 2x / 4x are measured on different slices.

## Same tokens at every batch size

All step counts are defined at batch size 256 and scaled by 256 / batch size: every batch size
trains on the same tokens, branches and evaluates at the same token counts (2560 sequences =
10.5M tokens per evaluation, every 252M tokens, plus a final one per job).

| Batch | Tokens / step | Stable (to step) | 1x branch -> end | 2x branch -> end | 4x branch -> end | Warmup | Eval every |
|---|---|---|---|---|---|---|---|
| 64 | 0.26M | 24672 | 6168 -> 7712 | 12336 -> 15420 | 24672 -> 30840 | 386 | 960 |
| 128 | 0.52M | 12336 | 3084 -> 3856 | 6168 -> 7710 | 12336 -> 15420 | 193 | 480 |
| 256 | 1.05M | 6168 | 1542 -> 1928 | 3084 -> 3855 | 6168 -> 7710 | 96 | 240 |
| 512 | 2.10M | 3084 | 771 -> 964 | 1542 -> 1928 | 3084 -> 3855 | 50 | 120 |

Warmup is N tokens (N = 101M active parameters, the N of the 20N / 40N / 80N budgets), i.e. 5 % of
the 1x budget, with a minimum of 50 steps: the rule scales with model size (Porian et al. 2024,
arXiv:2406.19146). `hyb100m/11` used 42M tokens (40 steps at batch 256).

(Batch 512, 2x: 1927.5 steps round to 1928, +0.01 % tokens; this is the one index
`prebuild_index.yaml` builds. Everything else reuses the indexes cached by hyb100m/11 + 11b.)

## Checkpoints

- permanent checkpoints only at the 1x / 2x branch points and the final step of the stable run
  (~16 GB each, ~48 GB per setting);
- crash recovery from ONE rolling checkpoint every ~40 min in `ckpt/non_persistent`; Megatron
  resumes from whichever is newer;
- a cooldown starts when its branch checkpoint's `.metadata` exists (written last).
- Restarts: automatic, at most 3 submissions per job; clean save + exit at 11.5 h.

## Cost (1 node = 4 GPUs per job)

Measured at batch 256 on hyb100m/11: AdamW ~4.7 s/step, Muon ~5.3, muown_classical ~4.6.

| | Per setting | Whole file (60 settings) |
|---|---|---|
| AdamW | ~50 GPU-h | ~3,000 |
| Muon | ~57 GPU-h (batch 64: ~75, estimate) | ~3,700 |
| muown_classical | ~51 GPU-h | ~3,100 |
| **All three** | | **~9,800** (2.7 % of the 360K grant) |

Batch 64 has 4x as many optimizer steps: Muon's stable run there will probably need one planned
resubmission at the 12 h limit (handled by the clean exit + restart).

## Running it

1. **Once:** `prebuild_index` (1 job, ~30 min, ~2 GPU-h); wait until it is COMPLETED.
   ```bash
   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/prebuild_index --submit-and-exit
   ```
2. Dry run and look at a script (`--global-batch-size`, `--train-iters`, `--lr`, betas / momentum):
   ```bash
   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/adam --dry-run --array-subset 28-31
   ```
3. Submit in tmux (the monitor releases the cooldowns and restarts crashes):
   ```bash
   python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/adam --array-subset <range>
   ```
   To stop: Ctrl-C the monitor FIRST, then `scancel`.

Job indices (same in all three files):
`index = ((gbs_i x 5 + lr_i) x 3 + beta_or_mom_i) x 4 + stage_i`, gbs_i 0..3 = 64 / 128 / 256 /
512, lr_i 0..4 (smallest first), beta_or_mom_i 0..2 = 0.9 / 0.95 / 0.975, stage_i 0..3 = stable /
cooldown1x / cooldown2x / cooldown4x.

| Batch size | Indices | Centre setting (middle lr, 0.95) |
|---|---|---|
| 64 | 0-59 | 28-31 |
| 128 | 60-119 | 88-91 |
| 256 | 120-179 | 148-151 |
| 512 | 180-239 | 208-211 |

Always submit all 4 stages of a setting together (whole blocks of 4 indices).

Outputs: `$OPT_SCALE_BASE/experiments/opt_scale/hyb100m_sweep/hyb100m_sweep_<opt>_lr<lr>_<beta|mom><value>_bs<batch>_<stage>/`
(with `node_mem_<jobid>.log`), logs in `$OPT_SCALE_BASE/logs/`. wandb project
`optimizer_scaling_laws`, tag `sweep`.

## Not exercised yet on this machine

- Resuming from the rolling (non-persistent) checkpoint: the first stable run that crashes or
  hits the time limit shows it.
- Batch sizes 64 / 128 / 512 (only 256 has run).
- The loader settings over a full-length run (tested for 500 steps including two validations).
