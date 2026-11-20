# opt_scale: log of the 100M sweep

What was run, what came out, what went wrong and which results to use. Newest entry last.
Design and per-phase detail are in `sweep100m/README.md`; this file is the short history.

Last updated 2026-11-20.

## Where things stand

| Part | Settings | State |
|---|---|---|
| Phase 1: 6 learning rates x batch 128 / 256 / 512 at beta / momentum 0.95; AdamW, Muon, Muown | 54 ladders | done, valid |
| Phase 2: beta / momentum 0.9 and 0.975 at two learning rates; AdamW, Muon; untied AdamW reference at batch 256 | 26 ladders | done, valid after reruns |
| Grid: completes 4 learning rates x {0.8, 0.9, 0.95, 0.975} for all three optimizers | 84 ladders | submitted 2026-11-20 00:50 (`sweep100m/grid_*.yaml`), running |
| Phase 1 scoring on the fixed validation set (`rescore/`) | 162 jobs | not submitted |

A ladder is one stable run with cooldowns at the 1x, 2x and 4x budgets (2.0B, 4.0B, 8.1B tokens),
about 50 GPU-h. "Muown" is `muown_classical`. Muown has only been run at momentum 0.95 so far.

## Which results to use

- Read results with `sweep100m/make_sweep_page.py` (HTML), `sweep100m/plot_beta_grids.py` (PDF)
  or `rescore/collect_rescore.py`. All three accept a cooldown only if its log shows that it
  loaded exactly the checkpoint step it asked for, and prefer a `_rerun` over the first attempt.
- Do not read checkpoints from a `..._cooldown2x` folder that has a `..._cooldown2x_rerun` folder
  beside it: the plain one branched from the wrong step (21 folders, phase 2 only).
- Two loss numbers exist. The in-training validation at the end of a cooldown (2,560 sequences)
  exists for every run. The fixed-set score (25,600 sequences, same for every checkpoint) exists
  for phase 2 only. The fixed-set loss is 0.013 to 0.052 higher; rankings agree.
- Differences below about 0.005 should not be read as real: there are no seed repeats, and the
  one controlled comparison (data order, below) moved the loss by 0.003.

## Results so far

Final in-training validation loss, best learning rate per cell, at 1x / 2x / 4x.

| Optimizer | Batch | 0.9 | 0.95 | 0.975 |
|---|---|---|---|---|
| AdamW (beta1 = beta2) | 128 | 2.342 / 2.173 / 2.089 | 2.338 / 2.161 / 2.074 | 2.371 / 2.166 / 2.072 |
| | 256 | 2.365 / 2.171 / 2.083 | 2.388 / 2.177 / 2.077 | 2.446 / 2.197 / 2.084 |
| | 512 | 2.459 / 2.208 / 2.096 | 2.523 / 2.232 / 2.102 | 2.706 / 2.309 / 2.134 |
| Muon | 128 | 2.256 / 2.100 / 2.032 | 2.265 / 2.104 / 2.035 | 2.285 / 2.118 / 2.046 |
| | 256 | 2.273 / 2.100 / 2.026 | 2.284 / 2.110 / 2.030 | 2.315 / 2.128 / 2.043 |
| | 512 | 2.332 / 2.131 / 2.041 | 2.344 / 2.139 / 2.047 | 2.390 / 2.166 / 2.062 |
| Muown | 128 | | 2.257 / 2.101 / 2.035 | |
| | 256 | | 2.278 / 2.107 / 2.032 | |
| | 512 | | 2.336 / 2.136 / 2.047 | |

Untied AdamW (beta1 0.9, beta2 0.95), batch 256: 2.374 / 2.171 / 2.075.

The 0.95 column takes the best of six learning rates, the others the best of two.

What this shows:

- Muon: momentum 0.9 is best in all nine cells; the gain over 0.95 shrinks from about 0.01 at 1x
  to about 0.004 at 4x. 0.9 is the lowest value run, so the optimum is not bracketed.
- AdamW: the best tied beta depends on the number of steps. Few steps (batch 512) favour 0.9,
  many steps (batch 128 at 4x) favour 0.95 to 0.975.
- Untied AdamW is level with the best tied setting at 2x and 4x and 0.009 behind at 1x. Comparing
  (0.9, 0.9), (0.9, 0.95) and (0.95, 0.95) at 4x suggests the gain from a higher tied beta comes
  from beta2, not from beta1. One batch size, two learning rates: a hint, not a result.
- Muon and Muown are within 0.006 of each other at 0.95; both are about 0.05 below AdamW at 4x.
- Batch 512 costs 0.015 to 0.024 against the best batch size at 4x, and much more at 1x.
- At beta 0.9 the higher of the two learning rates wins for AdamW at batch 256 and 512, and at
  0.975 the lower one wins at batch 128: the learning rate is not bracketed there.

## Known traps

**A cooldown can load the wrong checkpoint (found 2026-11-19).** With
`--non-persistent-ckpt-type global`, Megatron ignores `--ckpt-step` whenever the folder it loads
from has a rolling checkpoint at the same or a later step, and loads that instead
(`megatron/training/checkpointing.py`, `_load_base_checkpoint`). The job exits normally with a
plausible loss. A cooldown that waits in the queue while its stable run moves on therefore
branches too late. Phase 2: 21 of 26 `cooldown2x` were affected (16 branched one rolling interval
late, 5 loaded a step past their end and trained nothing). Phase 1: all 162 cooldowns checked,
none affected. Fix: `non_persistent_ckpt_type: null` on cooldown and scoring stages
(`gen_sweep100m.py`); confirmed by 24 reruns that loaded from folders with newer rolling
checkpoints. Check for any new run: the log line "successfully loaded checkpoint ... at
iteration N" must show the step in `ckpt_step`. Detail: `sweep100m/README.md`, "Phase 2 reruns".

**Cooldowns repeat training data (checked 2026-11-18).** With `dataloader_type: cyclic`, a job
that resumes with a different `train_iters` gets a different shuffle, so a cooldown rereads part
of what the stable run saw (about 25 % / 50 % / 80 % of its data at 1x / 2x / 4x). Measured cost
at 4x: 0.0033, the same for AdamW and Muon. All runs at one budget carry the same bias, so
comparisons within a budget are unaffected; the loss-against-tokens curve is slightly optimistic
at 4x. All phases keep the cyclic reader so that they compare directly. A run that starts fresh
can use `dataloader_type: single`. Detail: `data_order_check/README.md`.

**Slurm failures.** On 2026-11-19 three jobs hung at launch ("Unable to confirm allocation")
and two hung at exit until the time limit, out of about 180. They leave an empty log or a
finished result without a scoring job. A cancelled job is not restarted by the monitor.

**The monitor.** Cooldowns and scoring jobs are submitted by the monitor process, which must
keep running (tmux). Ctrl-C stops the monitor, not the jobs; resume with
`python scripts/monitor_autoexp.py --session <id>`. Start `tmux` and the launcher as separate
commands, or the launcher runs outside the session.

**Validation loss is below training loss** by 0.03 to 0.08 in every run. It is not dropout and
not the data mix (train and validation mix agree within 0.08 percentage points). Cause unknown;
it cancels in comparisons.

**Validation coverage.** The fixed set is the first 25,600 held-out sequences. It covers all 54
collections, but the smallest languages are nearly or fully absent.

## History

| Date | What |
|---|---|
| 2026-11-16 | Loader memory fix and batched Muon step verified; phase 1 submitted. |
| 2026-11-17 | Phase 1 complete: 54 ladders, 216 jobs. |
| 2026-11-18 | Data-order check: repeated data costs 0.003 at 4x. Phase 1 stands. |
| 2026-11-19 | Phase 2 submitted (AdamW, Muon, untied reference). Wrong-checkpoint problem found in 21 cooldowns; fixed in the generator; 24 cooldowns and 26 scoring jobs rerun (`sweep100m/phase2_fix_*.yaml`); all valid by the evening. |
| 2026-11-20 | Grid files written and dry-run for all three optimizers (84 ladders, about 4,200 GPU-h). |
| 2026-11-20 | Grid submitted at 00:50 from login node `jpbl-s01-04`. tmux session / monitor session id / stable job ids: `grid_muon` / 1795132253 / 3019734-3019757; `grid_muown` / 1795132385 / 3019767-3019826; `grid_adam` / 1795132394 / 3019785-3019828 (the last two ranges interleave). |

## Planned, not started

- 8x and 12x budgets for the best settings of the grid (stable run continued past step 6168,
  cooldowns at 8x and 12x). Needs one test of resuming a stable run past its planned end, and a
  check that the data cover 16B and 24B tokens.
- AdamW beta1 x beta2 at batch 256, to separate momentum from the second-moment average.
- Weight decay, at the best learning rate and beta / momentum of the grid.
- Larger models (230M, 500M, 900M, 1.8B active): a few runs per size, chosen by a Gaussian
  process whose prior comes from the 100M grid. Needs a throughput test per size first.
- That throughput test is written and dry-run, not submitted (2026-11-20):
  `smoke_test/17_scale_throughput.yaml`, read out with `smoke_test/collect_scale_throughput.py`.
  Four sizes (896 x 5, 1152 x 6, 1408 x 7, 1920 x 8: width and depth grow together) on 1 / 2 /
  4 / 8 nodes at a fixed global batch, with the experts on every GPU or split over the 4 GPUs of
  a node, for all three optimizers. The model shapes are stand-ins until the sizes are fixed.
  Design and job index are in the header of the yaml.
- Budget context: the October quota has to be used by early December (about 60,000 GPU-h in
  10 days from 2026-11-20). The 100M work can take about 9,000 to 13,000 of that.
