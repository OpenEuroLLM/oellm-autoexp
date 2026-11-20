# Optimizer hyperparameters at 100M active: learning rate, beta / momentum, batch size (AdamW, Muon, Muown)

Status: 2026-11-20, 16:15. The 1x budget is nearly complete; 2x and 4x are still running for the
settings added today. This issue will be updated when they finish.

## Summary

- Muon and Muown reach a lower loss than AdamW at every budget and batch size, by about 0.05 at the
  longest budget (2.03 against 2.07 to 2.08).
- Muon and Muown are within 0.006 of each other wherever both have been run.
- Momentum matters little for Muon and Muown: about 0.01 between 0.8 and 0.95. The best value is
  0.9 in nearly every case. 0.8 is not better.
- Beta matters much more for AdamW (0.05 or more), and the best value depends on the number of
  steps: 0.9 for short runs, 0.95 to 0.975 for long ones.
- Batch 128 and 256 are close. Batch 512 is worse, by a lot at the short budget and by 0.015 to
  0.024 at the long one.

## Setup

| | |
|---|---|
| Model | hybrid Gated DeltaNet + attention + MoE, 640 wide, 4 blocks of `LELELE*E` (32 layers), 96 experts, top-4 plus one shared expert, 101M active / 1.0B total parameters |
| Data | v8ml mix, sequence length 4096, 262k vocabulary |
| Machine | JUPITER, one node (4 GH200 GPUs) per job |
| Schedule | warmup, constant learning rate, linear decay to 0 over the last 20 % of the run |
| Budgets | 1x / 2x / 4x = 20 / 40 / 80 tokens per active parameter = 2.0B / 4.0B / 8.1B tokens |
| How budgets are run | one constant-rate run per setting; the 1x, 2x and 4x results branch from its checkpoints and each runs its own decay phase ("ladder") |
| AdamW | beta1 = beta2 (tied), weight decay 0.1, gradient clipping 1.0 |
| Muon | Nesterov momentum, weight decay 0.1; AdamW for non-matrix parameters |
| Muown | `muown_classical`, weight decay 0 |
| Reported number | validation loss at the end of the run (2,560 held-out sequences) |
| Seeds | one per setting |

Grid: learning rate in factor-2 steps (4 to 6 values per optimizer), beta / momentum in
{0.8, 0.9, 0.95, 0.975}, batch size in {128, 256, 512} sequences. 84 + 80 ladders in total.

## Results

Best validation loss over the learning rates run, with the winning learning rate in brackets.
Columns are the 1x / 2x / 4x budgets. A dash means the runs have not finished.

### AdamW (beta1 = beta2)

| Batch | Beta | 1x | 2x | 4x |
|---|---|---|---|---|
| 128 | 0.8 | 2.401 (2e-3) | - | - |
| 128 | 0.9 | 2.342 (2e-3) | 2.173 (1e-3) | 2.089 (1e-3) |
| 128 | 0.95 | **2.338** (2e-3) | **2.161** (2e-3) | 2.074 (1e-3) |
| 128 | 0.975 | 2.371 (1e-3) | 2.166 (1e-3) | **2.072** (1e-3) |
| 256 | 0.8 | 2.389 (2e-3) | - | - |
| 256 | 0.9 | **2.365** (2e-3) | **2.171** (2e-3) | 2.083 (2e-3) |
| 256 | 0.95 | 2.388 (4e-3) | 2.177 (1e-3) | **2.077** (1e-3) |
| 256 | 0.975 | 2.446 (2e-3) | 2.197 (2e-3) | 2.084 (2e-3) |
| 512 | 0.8 | 2.508 (4e-3) | 2.237 (4e-3) | - |
| 512 | 0.9 | **2.459** (2e-3) | **2.208** (2e-3) | **2.096** (2e-3) |
| 512 | 0.95 | 2.523 (2e-3) | 2.232 (2e-3) | 2.102 (2e-3) |
| 512 | 0.975 | 2.706 (1e-3) | 2.309 (1e-3) | 2.134 (2e-3) |

Untied reference (beta1 0.9, beta2 0.95), batch 256: 2.374 / 2.171 / 2.075. It matches the best
tied setting at 2x and 4x and is 0.009 behind at 1x.

### Muon

| Batch | Momentum | 1x | 2x | 4x |
|---|---|---|---|---|
| 128 | 0.8 | 2.259 (4e-3) | - | - |
| 128 | 0.9 | **2.256** (4e-3) | **2.100** (2e-3) | **2.032** (2e-3) |
| 128 | 0.95 | 2.265 (4e-3) | 2.104 (2e-3) | 2.035 (2e-3) |
| 128 | 0.975 | 2.285 (2e-3) | 2.118 (2e-3) | 2.046 (2e-3) |
| 256 | 0.8 | 2.279 (8e-3) | - | - |
| 256 | 0.9 | **2.272** (8e-3) | **2.100** (4e-3) | **2.026** (4e-3) |
| 256 | 0.95 | 2.284 (4e-3) | 2.110 (2e-3) | 2.030 (2e-3) |
| 256 | 0.975 | 2.315 (4e-3) | 2.128 (2e-3) | 2.043 (2e-3) |
| 512 | 0.8 | 2.331 (8e-3) | 2.143 (4e-3) | - |
| 512 | 0.9 | **2.322** (8e-3) | **2.131** (4e-3) | **2.041** (4e-3) |
| 512 | 0.95 | 2.344 (4e-3) | 2.139 (4e-3) | 2.047 (4e-3) |
| 512 | 0.975 | 2.390 (4e-3) | 2.166 (4e-3) | 2.062 (4e-3) |

### Muown

Only momentum 0.95 has 2x and 4x results so far.

| Batch | Momentum | 1x | 2x | 4x |
|---|---|---|---|---|
| 128 | 0.8 | 2.259 (4e-3) | - | - |
| 128 | 0.9 | **2.252** (4e-3) | - | - |
| 128 | 0.95 | 2.257 (4e-3) | 2.101 (2e-3) | 2.035 (2e-3) |
| 128 | 0.975 | 2.274 (2e-3) | - | - |
| 256 | 0.8 | 2.283 (4e-3) | - | - |
| 256 | 0.9 | **2.276** (8e-3) | - | - |
| 256 | 0.95 | 2.278 (4e-3) | 2.107 (4e-3) | 2.032 (4e-3) |
| 256 | 0.975 | 2.302 (4e-3) | - | - |
| 512 | 0.8 | 2.343 (4e-3) | - | - |
| 512 | 0.9 | **2.324** (4e-3) | - | - |
| 512 | 0.95 | 2.336 (4e-3) | 2.136 (4e-3) | 2.047 (4e-3) |
| 512 | 0.975 | 2.380 (4e-3) | - | - |

## What the numbers show

1. **Muon family against AdamW.** Best 4x loss: Muon 2.026, Muown 2.032, AdamW 2.072. The gap is
   larger at short budgets (1x: 2.256 against 2.338).
2. **Momentum for Muon and Muown.** 0.9 is best in all nine Muon cells that are complete. 0.8 is
   0.003 to 0.009 worse at 1x, so the optimum is now bracketed from both sides. The advantage of
   0.9 over 0.95 shrinks with budget, from about 0.01 at 1x to about 0.004 at 4x.
3. **Beta for AdamW.** With few steps (batch 512, or the 1x budget) 0.9 is best. With many steps
   (batch 128 at 4x) 0.95 and 0.975 are best. 0.8 is worse everywhere it has finished, by 0.02 to
   0.06.
4. **Learning rate and momentum interact.** Lower momentum prefers a higher learning rate: at
   momentum 0.8 and 0.9, Muon's best learning rate at batch 256 and 512 is the highest one run.
5. **Best learning rate moves with batch size and budget.** Larger batches prefer a higher
   learning rate; longer budgets prefer a lower one. For Muon the best value goes from 4e-3 to
   8e-3 between batch 128 and 512 at 1x, and from 4e-3 to 2e-3 between 1x and 4x at batch 128.
6. **Batch size.** At 4x, batch 512 costs 0.015 (Muon) to 0.024 (AdamW) against the best batch
   size. At 1x it costs 0.07 (Muon) to 0.12 (AdamW).

## Limits of these results

- **No seed repeats.** One controlled comparison (same setting, different data order) moved the
  loss by 0.003. Differences below about 0.005 should not be read as real.
- **Learning rate not bracketed in some cells.** Where the winning learning rate is the highest
  or lowest one run, the true optimum may lie outside: Muon and Muown at momentum 0.8 and 0.9,
  batch 256 and 512 (best is 8e-3, the top of the grid), and several AdamW cells at 0.9 and 0.975
  that so far have only two learning rates at 2x and 4x.
- **The decay phases reread some training data.** Measured cost: 0.003 at 4x, the same for AdamW
  and Muon, so comparisons within a budget are unaffected.
- **Validation loss is below training loss** by 0.03 to 0.08 in every run. The cause is not
  known. It is the same for all runs.
- **One model size.** Nothing here says how the best settings move with model size.

## Next

- Finish the 2x and 4x budgets of the grid (running).
- Repeat the same lattice at about 250M and 500M active parameters to measure how the best
  learning rate and momentum move with model size.

## Where things are

| | |
|---|---|
| Configs and generator | `config/experiments/laingsam1/opt_scale/sweep100m/` |
| History, known problems | `config/experiments/laingsam1/opt_scale/LOG.md` |
| Figures (PDF heatmaps) | `sweep100m/plot_beta_grids.py` |
| Interactive page | `sweep100m/make_sweep_page.py` |
