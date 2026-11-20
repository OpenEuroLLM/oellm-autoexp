# sweep100m: optimizer hyperparameter sweep on the 100M hybrid MoE (JUPITER)

Compares AdamW, Muon and muown_classical on the team's 100M-active hybrid Gated-DeltaNet MoE,
and measures how the best learning rate moves with batch size and token budget. The results
are meant as a prior for hyperparameter search at larger model sizes.

Status (2026-11-20): phase 1 (learning rate x batch size) is complete, all 216 jobs finished;
results are below. Phase 2 (beta / momentum) for AdamW and Muon ran on 2026-11-19; 21 of its
26 2x cooldowns and three 4x cooldowns were run again the same day (section "Phase 2 reruns")
and all phase 2 results are now valid. The grid that completes learning rate x beta / momentum
is prepared (last section). Short history and current results: `../LOG.md`.

To run a slice of it, follow `RUNBOOK.md`. This page describes the design.

## One setting = one ladder of 4 jobs

Every hyperparameter setting is trained once at constant learning rate and branched into
cooldowns, so one training run gives the final loss at three token budgets:

| Job | What it does |
|---|---|
| `stable` | warmup, then constant learning rate up to 6.5B tokens; saves a checkpoint at each branch point |
| `cooldown1x` / `cooldown2x` / `cooldown4x` | start from a branch checkpoint and decay the learning rate linearly to 0 over the last 20 % of the budget; final loss at 2.0B / 4.0B / 8.1B tokens |

The budgets are 20N / 40N / 80N tokens with N = 101,061,472 active parameters (1x / 2x / 4x
Chinchilla). The cooldown jobs are submitted by the monitor once their checkpoint exists.

## Files

| File | Content | In use |
|---|---|---|
| `phase1_adam.yaml`, `phase1_muon.yaml`, `phase1_muown_classical.yaml` | phase 1: 3 batch sizes x 6 learning rates, beta / momentum 0.95. 18 ladders = 72 jobs per file | done |
| `phase2_adam.yaml`, `phase2_muon.yaml` | phase 2: 3 batch sizes x 2 learning rates x 2 new beta / momentum values, 12 ladders = 84 jobs each | run 2026-11-19 |
| `phase2_fix_adam.yaml`, `phase2_fix_adam_untied.yaml`, `phase2_fix_muon.yaml` | the phase 2 cooldowns and scoring jobs that were run again on 2026-11-19 (16 + 4 + 30 jobs), from `gen_phase2_fix.py` | done |
| `phase2_muown_classical.yaml` | phase 2 for muown_classical; on hold until the Muon result is in | later |
| `phase2_adam_untied.yaml` | phase 2 reference: AdamW with the usual beta1 = 0.9, beta2 = 0.95 at batch 256, 2 ladders = 14 jobs | run 2026-11-19 |
| `plot_loss_grids.py` | heatmaps of the final loss (batch size x learning rate, one panel per budget) from the Slurm logs | |
| `run1x_<optimizer>.yaml` | the same grid as single runs trained from scratch to the 1x budget (no ladder); for checking that a branched cooldown equals a from-scratch run | not yet |
| `adam.yaml`, `muon.yaml`, `muown_classical.yaml` | full grid: 3 batch sizes x 5 learning rates x 3 beta / momentum values, 45 ladders = 180 jobs per file | not used; replaced by the phase 2 files |
| `prebuild_index.yaml` | built one missing data index; already run, not needed again for the shared cache | done |
| `gen_sweep100m.py` | generates every yaml above from `../hyb100m/11_wsd_ladder.yaml`; change the grid there and re-run it | |
| `RUNBOOK.md` | step by step: environment, dry run, submitting, health checks, stopping, collecting results | |

## Phase 1 grid

| | Values |
|---|---|
| batch size (sequences of 4096 tokens) | 128, 256, 512 |
| learning rate (all three optimizers) | 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3 |
| AdamW beta1 = beta2 | 0.95 |
| Muon momentum (`muon_momentum`) | 0.95 |
| muown_classical momentum (`--angular-muown-momentum`) | 0.95 |
| weight decay | 0.1 (AdamW, Muon), 0 (muown_classical) |
| gradient clipping, seed | 1.0, 1234 |

Why one learning-rate grid for all optimizers: Muon and muown_classical scale their matrix
update by 0.2 x sqrt(max(rows, cols)), which makes it the size of an AdamW update, so the same
learning rates apply.

Job index in a phase 1 file: `(batch_i x 6 + lr_i) x 4 + stage_i`, with batch_i 0..2 = 128 /
256 / 512, lr_i 0..5 (smallest first), stage_i 0..3 = stable / cooldown1x / cooldown2x /
cooldown4x. Always submit the 4 jobs of a ladder together.

| lr | Batch 128 | Batch 256 | Batch 512 |
|---|---|---|---|
| 2.5e-4 | 0-3 | 24-27 | 48-51 |
| 5e-4 | 4-7 | 28-31 | 52-55 |
| 1e-3 | 8-11 | 32-35 | 56-59 |
| 2e-3 | 12-15 | 36-39 | 60-63 |
| 4e-3 | 16-19 | 40-43 | 64-67 |
| 8e-3 | 20-23 | 44-47 | 68-71 |

## Phase 1 results (beta / momentum 0.95)

Final validation loss of each cooldown (runs of 2026-11-16 to 18, one seed). Bold = best
learning rate of the row. Figures: `plot_loss_grids.py` writes them to
`$OPT_SCALE_BASE/figures/sweep100m/`.

1x budget (2.0B tokens)

| Optimizer | Batch | 2.5e-4 | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 |
|---|---|---|---|---|---|---|---|
| AdamW | 128 | 2.477 | 2.385 | 2.349 | **2.338** | 2.358 | 2.381 |
| AdamW | 256 | 2.605 | 2.472 | 2.393 | 2.396 | **2.388** | 2.456 |
| AdamW | 512 | 2.879 | 2.675 | 2.553 | **2.523** | 2.610 | 2.851 |
| Muon | 128 | 2.416 | 2.321 | 2.280 | 2.266 | **2.265** | 2.281 |
| Muon | 256 | 2.527 | 2.386 | 2.321 | 2.291 | **2.284** | 2.288 |
| Muon | 512 | 2.738 | 2.525 | 2.412 | 2.364 | **2.344** | 2.352 |
| muown_classical | 128 | 2.427 | 2.331 | 2.279 | 2.259 | **2.257** | 2.262 |
| muown_classical | 256 | 2.532 | 2.389 | 2.316 | 2.286 | **2.278** | 2.285 |
| muown_classical | 512 | 2.743 | 2.523 | 2.406 | 2.352 | **2.336** | 2.377 |

2x budget (4.0B tokens)

| Optimizer | Batch | 2.5e-4 | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 |
|---|---|---|---|---|---|---|---|
| AdamW | 128 | 2.239 | 2.180 | 2.162 | **2.161** | 2.183 | 2.207 |
| AdamW | 256 | 2.305 | 2.222 | **2.177** | 2.181 | 2.178 | 2.219 |
| AdamW | 512 | 2.441 | 2.314 | 2.248 | **2.232** | 2.255 | 2.329 |
| Muon | 128 | 2.200 | 2.138 | 2.112 | **2.104** | 2.111 | 2.136 |
| Muon | 256 | 2.258 | 2.168 | 2.128 | **2.110** | 2.110 | 2.119 |
| Muon | 512 | 2.367 | 2.240 | 2.175 | 2.148 | **2.139** | 2.146 |
| muown_classical | 128 | 2.214 | 2.150 | 2.116 | **2.101** | 2.103 | 2.112 |
| muown_classical | 256 | 2.268 | 2.174 | 2.130 | 2.110 | **2.107** | 2.115 |
| muown_classical | 512 | 2.374 | 2.242 | 2.174 | 2.144 | **2.136** | 2.166 |

4x budget (8.1B tokens)

| Optimizer | Batch | 2.5e-4 | 5e-4 | 1e-3 | 2e-3 | 4e-3 | 8e-3 |
|---|---|---|---|---|---|---|---|
| AdamW | 128 | 2.120 | 2.082 | **2.074** | 2.082 | 2.105 | 2.133 |
| AdamW | 256 | 2.157 | 2.102 | **2.077** | 2.082 | 2.084 | 2.112 |
| AdamW | 512 | 2.232 | 2.150 | 2.110 | **2.102** | 2.113 | 2.142 |
| Muon | 128 | 2.096 | 2.054 | 2.036 | **2.035** | 2.048 | 2.077 |
| Muon | 256 | 2.127 | 2.068 | 2.042 | **2.030** | 2.035 | 2.047 |
| Muon | 512 | 2.189 | 2.108 | 2.067 | 2.050 | **2.047** | 2.053 |
| muown_classical | 128 | 2.113 | 2.069 | 2.045 | **2.035** | 2.038 | 2.048 |
| muown_classical | 256 | 2.139 | 2.077 | 2.047 | 2.034 | **2.032** | 2.042 |
| muown_classical | 512 | 2.196 | 2.112 | 2.069 | 2.052 | **2.047** | 2.070 |

What it shows (one seed, so differences of a few thousandths are not meaningful):

- No optimum sits at the edge of the grid and no run diverged, also not at 8e-3.
- AdamW's best learning rate moves down with the budget (2e-3 to 4e-3 at 1x, 1e-3 at 4x);
  Muon and muown_classical stay at 2e-3 to 4e-3.
- Muon and muown_classical are below AdamW (with beta1 = beta2 = 0.95) in every batch size and
  budget, by 0.07-0.18 at 1x and 0.04-0.055 at 4x.
- Muon and muown_classical are within 0.01 of each other. They are not a clean comparison:
  they differ in weight decay (0.1 vs 0) and in the orthogonalization (see below).
- Batch 128 is best at 1x and 2x; at 4x batch 128 and 256 are equal.

## Phase 2: beta / momentum (run 2026-11-19, reruns done)

**Question.** Phase 1 fixed beta / momentum at 0.95. Phase 2 asks three things:

1. How much does the final loss change with beta (AdamW) or momentum (Muon) at a well-tuned
   learning rate?
2. Does the best learning rate move with it? If yes, the two have to be searched jointly at
   larger model sizes; if no, they can be tuned one after the other.
3. Does either answer depend on batch size?

**Grid.** Every combination of the values in a row is one ladder: stable run + cooldowns at
1x / 2x / 4x as in phase 1, plus one scoring job per cooldown (7 jobs).

| Sweep | Batch sizes | Learning rates | New values | Ladders |
|---|---|---|---|---|
| AdamW, beta1 = beta2 | 128, 256, 512 | 1e-3, 2e-3 | 0.9, 0.975 | 12 |
| AdamW reference, beta1 = 0.9, beta2 = 0.95 | 256 | 1e-3, 2e-3 | (fixed) | 2 |
| Muon momentum | 128, 256, 512 | 2e-3, 4e-3 | 0.9, 0.975 | 12 |

26 ladders = 182 jobs, about 1,300 GPU-h. The value 0.95 at the same batch sizes and learning
rates already exists from phase 1, so each sweep has three values: 0.9, 0.95, 0.975 (the
averaging horizon doubles each time: 10, 20, 40 steps).

Weight decay 0.1, clipping 1.0, seed 1234, the data mix and the step counts are as in phase 1;
the data reader and the evaluation differ, see "What differs from phase 1" below. In the Muon runs only the momentum of the matrix update is swept; the parameters
Muon hands to Adam (embeddings, router, norms) keep beta1 = beta2 = 0.95.

**Why these choices.**

- *Two learning rates per optimizer.* They are the two that won phase 1: the best learning
  rate was 1e-3 or 2e-3 for AdamW in 8 of 9 cells (batch size x budget) and 2e-3 or 4e-3 for
  Muon in all 9. The loss is flat around the optimum, so the better of the two is a good
  estimate of the tuned loss. Whether the optimum moves shows up as a change, across the three
  beta / momentum values, of the loss gap between the two learning rates.
- *All three batch sizes.* The same momentum averages over four times as many tokens at batch
  512 as at 128, so the best value may depend on batch size. With one seed, the middle batch
  size is also the only check that a difference between 128 and 512 is a trend and not noise.
- *No repeats with other seeds.* Differences below about 0.005 should not be read as real
  (estimated from the scatter between neighbouring phase 1 runs, not measured).
- *The AdamW reference.* The sweep uses beta1 = beta2; the reference shows where the usual
  (0.9, 0.95) lands, so the AdamW baseline is not judged only with tied betas. One batch size
  is enough for that.
- *Muon before muown_classical.* The two responded to learning rate almost identically in
  phase 1 (losses within 0.008, same best learning rate in 7 of 9 cells), so Muon's result is
  used to decide how much of this muown_classical needs.

**What happens after, depending on the result.**

| Result | Next step |
|---|---|
| Momentum makes no clear difference for Muon (within about 0.005) | add 0.8 and 0.9875 at the best learning rate only, to find where it starts to matter; muown_classical gets a short check: 3 batch sizes x its best learning rate x momentum 0.9 and 0.975 (6 ladders) |
| Momentum clearly matters | run muown_classical on the same grid as Muon |
| The gap between the two learning rates changes with beta / momentum | add one learning rate on the side it points to (AdamW 5e-4 or 4e-3, Muon 1e-3 or 8e-3) |

After that: weight decay, with the same values for Muon and muown_classical so that their
comparison becomes clean.

**What differs from phase 1.** Only the beta / momentum value. The training jobs use the same
configuration as phase 1, including the cyclic data reader, so the new values compare directly
with the 0.95 runs of phase 1. This was decided on 2026-11-18 after the data-order check: the
cyclic reader makes cooldowns repeat part of the stable run's data, which costs about 0.003 in
final loss at 4x (`../data_order_check/README.md`); keeping the reader keeps that small bias
equal in all runs instead of introducing a difference between the phases.

One addition: each cooldown is followed by a scoring job (`score1x` / `score2x` / `score4x`,
no training, about 11 minutes) that evaluates the final checkpoint on a fixed validation set,
the first 25,600 held-out sequences. `../rescore` does the same for the phase 1 checkpoints, so
both phases are compared on those numbers:
`python3 ../rescore/collect_rescore.py --logs $OPT_SCALE_BASE/logs`. The evaluations printed
during training use other sequences each time and scatter by about 0.018; use them for curves.

**Files and job indexes.** Job index = setting x 7 + stage, with stage 0 stable,
1-3 cooldown1x / 2x / 4x, 4-6 score1x / 2x / 4x.

| File | Ladders | Jobs | Batch 128 | Batch 256 | Batch 512 |
|---|---|---|---|---|---|
| `phase2_adam.yaml` | 12 | 84 | 0-27 | 28-55 | 56-83 |
| `phase2_adam_untied.yaml` (the reference) | 2 | 14 | | 0-13 | |
| `phase2_muon.yaml` | 12 | 84 | 0-27 | 28-55 | 56-83 |
| `phase2_muown_classical.yaml` (held back, see the table above) | 12 | 84 | 0-27 | 28-55 | 56-83 |

Run names: `hyb100m_sweep_<optimizer>_lr<lr>_beta0.975_bs<batch>_<stage>`, `..._mom0.9_...`,
and `..._b1-0.9_b2-0.95_...` for the reference; wandb tag `phase2`. The scoring jobs are named
`hyb100m_rescore_..._score4x` and write to `experiments/opt_scale/hyb100m_rescore/`.

Submit each file inside its own tmux session; the process has to keep running to start the
cooldowns and the scoring jobs:

```bash
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase2_muon
```

Checked before submission (2026-11-19): the three files dry-run, and each of the 182 rendered
job scripts was compared with the phase 1 script of the same optimizer, learning rate, batch
size and stage: they differ only in the beta / momentum value, the names and the wandb tag.
The scoring jobs do not print the line the monitor waits for, so the monitor may keep running
after the last job; stop it when `squeue` is empty.

When checking the first logs: where the value is 0.9, the job script has no `--muon-momentum` /
`--adam-beta1` flag, because 0.9 is Megatron's command-line default and the launcher leaves
defaults out. The argument list at the top of the log shows the value in use.

## Phase 2 reruns

**What happened (2026-11-19).** A cooldown is started with `--load <stable run>/ckpt
--ckpt-step <branch step>`. The phase 1 and phase 2 cooldowns also carried
`--non-persistent-ckpt-type global`, inherited from the stable run, and with that flag Megatron
loads the stable run's rolling checkpoint instead of `ckpt_step` whenever the rolling one is
at the same or a later step (`megatron/training/checkpointing.py`, `_load_base_checkpoint`).
The job does not fail and prints a normal final loss; only the line "successfully loaded
checkpoint ... at iteration N" shows it.

A cooldown that starts before the stable run writes its next rolling checkpoint (about every
35 minutes) is not affected. In phase 1 every cooldown started in time: all 162 logs show the
branch step. In phase 2 the 2x cooldowns waited about an hour in the queue:

| Phase 2 jobs | Count | State |
|---|---|---|
| 1x cooldowns | 26 | valid, loaded the branch step |
| 2x cooldowns | 5 | valid (AdamW batch 256 beta 0.9, both learning rates; batch 512 lr 1e-3 beta 0.9; batch 512 beta 0.975, both) |
| 2x cooldowns | 16 | branched one rolling-checkpoint interval late: the cooldown has a third of its length |
| 2x cooldowns | 5 | loaded a step past their last step and did not train |
| 4x cooldowns | 23 | valid; the stable run ends at their branch step, so no later checkpoint exists |
| 4x cooldowns | 3 | never started (Slurm failure at launch); Muon momentum 0.975: batch 256 both learning rates, batch 512 lr 2e-3 |
| 4x scores | 2 | never submitted; their cooldowns finished but hung at exit (AdamW lr 2e-3: beta 0.9 batch 256, beta 0.975 batch 512) |

**Fix.** Cooldowns and scoring jobs now set `non_persistent_ckpt_type: null` (in
`gen_sweep100m.py`, so in every generated file), which leaves Megatron only `ckpt_step` to
load. The jobs above ran with the flag; the regenerated phase 1 and phase 2 files differ from
what was submitted in exactly this setting. The fix follows from the code and is not yet
confirmed by a run: check the "successfully loaded checkpoint" line of the first rerun.

**Rerun files.** `phase2_fix_<key>.yaml` contain only the jobs above. The stable runs are not
repeated; the branch checkpoints are read from disk.

| File | Cooldowns | Scoring jobs | Job index |
|---|---|---|---|
| `phase2_fix_adam.yaml` | 7 x 2x | 7 x 2x, 2 x 4x | 0-6 cooldowns, 7-15 scores |
| `phase2_fix_adam_untied.yaml` | 2 x 2x | 2 x 2x | 0-1 cooldowns, 2-3 scores |
| `phase2_fix_muon.yaml` | 12 x 2x, 3 x 4x | 12 x 2x, 3 x 4x | 0-14 cooldowns, 15-29 scores |

The 2x reruns are named `..._cooldown2x_rerun` and `..._score2x_rerun` (wandb tag
`phase2_fix`); the first attempts keep their names and folders. The 4x jobs keep their original
names, since nothing was produced under them. About 130 GPU-h in total.

Checked on the dry run (2026-11-19): each of the 42 2x rerun scripts differs from the script
submitted on 2026-11-19 only in the missing `--non-persistent-ckpt-type global`, the names and
the wandb tag; the 4x scripts match a 4x job that ran, apart from the learning rate; every
checkpoint the reruns read exists.

**Results.** `../rescore/collect_rescore.py` compares, for every score, the step the cooldown
loaded with the step it asked for, leaves out the scores of cooldowns that branched elsewhere
and lists them. The validation losses in the logs and in wandb of the 21 first-attempt 2x
cooldowns (`..._cooldown2x` without `_rerun`, phase 2 only) must not be used.

## Model, data and what differs from other team configs

Reference: Swagatam's 100M "pure MoE baseline"
(`swagatam/alps/engram_scaling/100m_1b.yaml`, branch `feat/hybrid-v0.19-alps`).

- **Same as the reference:** every model setting (4 x `LELELE*E`, width 640, 96 experts
  top-4, tied 262,144-row embedding, 101,061,472 active parameters, no Engram layers), Megatron
  branch `oellm/v0.19-hybrid`, the team container, and the data loading
  (`create_attention_mask_in_dataloader: false`, `dataloader_type: cyclic`,
  `padded_vocab_size: 262144`; `mmap_bin_files: false` as the team's hybrid branch).
- **Same as the reference, unlike Korbinian's production run** (`korbi/production/hybrid_ambitious_*`,
  a much larger model): no document masking, no BOS per document, sequence length 4096,
  validation on a held-out split, no cap on the expert matrices.
- **Specific to this study:** the WSD schedule with cooldown branches, the grid above,
  micro-batch 16 with the Liger fused loss (speed only), and the data mix below.

Data: `../data/datamix_v8ml_flagfull_jsc.txt`, the official JUPITER v8ml mix with identical
relative weights, minus 9 entries that are not on disk (1.9 % of the weight). The flag sources
(71 % of the weight) are read from the original collection because the decontaminated copy is
incomplete on JUPITER. Split 99 / 1 / 0.

Things to know when reading the results:

- Muon and muown_classical use one learning rate for their matrices and for the parameters
  handled by Adam (embeddings, router, norms; betas 0.95 / 0.95). Sweeping it moves both.
- muown_classical splits each matrix row into a length and a direction. The momentum is for
  the directions; the lengths are on Adam with betas (0.9, 0.95). Its weight decay is 0 here
  (Swagatam's setting) while Muon has 0.1, so the two differ in more than the update rule.
- Orthogonalization, 5 iterations in both: Muon uses the Polar Express coefficients
  (`muon_coefficient_type: polar_express`), muown_classical the classic Newton-Schulz ones
  (`angular_muown_coefficient_type: simple`, as logged by the runs).
- Muon runs with `--muon-batched-step`: matrices of the same shape (for example the 96 experts
  of a layer) are orthogonalized in one batched call. Each matrix is still treated separately;
  it is only faster (checked in `../smoke_test/16_muon_batched_check.yaml`).
- AdamW uses beta1 = beta2, which is not the usual (0.9, 0.95); phase 2 adds that as a reference.
- Validation: the last 1 % of the documents of every dataset is held out. Each evaluation uses
  the next 2,560 sequences of that pool, the same ones for every run at the same token count.
  The stable job and each cooldown job have their own sequence order, so the final losses at
  1x, 2x and 4x are measured on different sequences.

## Same tokens at every batch size

Step counts are defined at batch size 256 and scaled by 256 / batch size, so every batch size
trains on the same tokens and branches and evaluates at the same token counts (one evaluation
= 2,560 sequences = 10.5M tokens, every 252M tokens, plus a final one per job).

| Batch | Tokens / step | Stable (to step) | 1x branch -> end | 2x branch -> end | 4x branch -> end | Warmup steps | Eval every |
|---|---|---|---|---|---|---|---|
| 128 | 0.52M | 12336 | 3084 -> 3856 | 6168 -> 7710 | 12336 -> 15420 | 193 | 480 |
| 256 | 1.05M | 6168 | 1542 -> 1928 | 3084 -> 3855 | 6168 -> 7710 | 96 | 240 |
| 512 | 2.10M | 3084 | 771 -> 964 | 1542 -> 1928 | 3084 -> 3855 | 50 | 120 |

Warmup is N tokens (N = the 101M active parameters), which is 5 % of the 1x budget, with a
minimum of 50 steps. The rule is stated in tokens so that it scales with model size
(following Porian et al. 2024, arXiv:2406.19146).

## Checkpoints and restarts

- Permanent checkpoints: the two branch points and the final step of the stable run, and the
  final step of each cooldown.
- Crash recovery of the stable run: a rolling checkpoint about every 40 minutes in
  `ckpt/non_persistent`; Megatron resumes from whichever checkpoint is newer. Cooldowns and
  scoring jobs switch this off, so that they load exactly their branch step (see "Phase 2 reruns").
- A cooldown starts when the `.metadata` file of its branch checkpoint exists (it is written last).
- A crashed job is resubmitted automatically, at most 3 submissions per job.
- Disk per ladder: about 95 GB (Muon), 125 GB (AdamW), 215 GB (muown_classical, whose
  checkpoints are larger).

## Cost and speed (measured 2026-11-16/17, 1 node = 4 GPUs per job)

| | Batch 128 | Batch 256 | Batch 512 | GPU-h per ladder |
|---|---|---|---|---|
| AdamW | 2.1 s/step | 4.1 | 8.0 | 44-46 |
| Muon | 2.3 | 4.3 | 8.4 | 46-48 |
| muown_classical | 2.3 | 4.4 | 8.5 | 46-52 |

A stable run takes 7.5-8.6 h, a whole ladder about 11-12 h from submission. One phase 1 file
(18 ladders) is about 850 GPU-h; all three used about 2,570.

## Where the output goes

`$OPT_SCALE_BASE/experiments/opt_scale/hyb100m_sweep/hyb100m_sweep_<optimizer>_lr<lr>_<beta|mom><value>_bs<batch>_<stage>/`
holds checkpoints, tensorboard, offline wandb files and `node_mem_<jobid>.log`. Slurm logs are
in `$OPT_SCALE_BASE/logs/`. wandb project `optimizer_scaling_laws`, tag `sweep`; the runs log
offline and are uploaded from a login node (`RUNBOOK.md`, section 8).

## Not verified yet

- Resuming a stable run from the rolling checkpoint. Two phase 1 jobs crashed (Muon 8e-3,
  batch 128 and 256, a bus error right after a checkpoint save) and resumed correctly, but from
  a permanent branch checkpoint.
- Two muown_classical 1x cooldowns hung at exit after writing their final loss and checkpoint
  and ran into the time limit; the results are complete, the cause is unknown.
- The full-grid and `run1x` files have only been dry-run.
- The phase 1 and phase 2 cooldowns repeat part of the stable run's training data (cyclic reader); measured
  effect at 4x about 0.003 in final loss, see `../data_order_check/README.md`.

## Background: why the loader settings matter

With Megatron's default (`create_attention_mask_in_dataloader: true`) each of the 522 datasets
caches a 4096 x 4096 mask in every data-reading process. That filled the node's memory and
crashed jobs in `../hyb100m/11_wsd_ladder.yaml` on 2026-11-16. The model masks causally by
itself (`hybrid_layer_specs.py`), so the setting is off here, as in Swagatam's and Korbinian's
configs. With it off, allocated host memory stayed below 85 GB over an 8.5 h run.

## Grid: full learning rate x beta / momentum (prepared 2026-11-20)

`grid_adam.yaml`, `grid_muon.yaml` and `grid_muown_classical.yaml` complete phases 1 and 2 to a full
grid at every batch size (128, 256, 512): four learning rates x {0.8, 0.9, 0.95, 0.975}.

| File | Learning rates | New settings per batch size | Ladders | GPU-h (about) |
|---|---|---|---|---|
| `grid_adam.yaml` | 5e-4, 1e-3, 2e-3, 4e-3 | 0.8 at all four; 0.9 and 0.975 at 5e-4 and 4e-3 | 24 | 1,200 |
| `grid_muon.yaml` | 1e-3, 2e-3, 4e-3, 8e-3 | 0.8 at all four; 0.9 and 0.975 at 1e-3 and 8e-3 | 24 | 1,200 |
| `grid_muown_classical.yaml` | 1e-3, 2e-3, 4e-3, 8e-3 | 0.8, 0.9 and 0.975 at all four | 36 | 1,800 |

Jobs are configured as in phase 2 (seven per ladder: stable run, three cooldowns, three scoring
jobs), with the checkpoint fix of "Phase 2 reruns" (no rolling checkpoints in cooldowns and
scoring jobs). The job index is in the header of each file. The files are generated by
`gen_sweep100m.py`; the list of settings that already exist is `GRID_DONE` there.

Plots of the full grid: `plot_beta_grids.py` (PDF) and `make_sweep_page.py` (HTML).
