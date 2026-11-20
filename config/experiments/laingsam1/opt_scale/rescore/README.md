# rescore: the phase 1 final losses on one fixed validation set

Status (2026-11-18): configs written and dry-run, all 162 source checkpoints found on disk.
Not submitted, no results yet.

## Why

In the sweep (`../sweep100m`), every evaluation reads the next 2,560 sequences of the held-out
1 % split. Runs at the same budget are scored on the same sequences, so comparisons at one
budget are sound. But the final losses at 1x, 2x and 4x are measured on different sequences, and
a single evaluation of that size scatters by about 0.018. That is too coarse for comparing
budgets or fitting how the loss falls with tokens.

Here every final checkpoint is scored again, without training, on the same 25,600 held-out
sequences (105M tokens, ten times a regular evaluation), read in fixed order. The resulting
losses are comparable across budgets, batch sizes and optimizers.

They are on a different (larger) set than the losses in `../sweep100m/README.md`, so the two
sets of numbers differ slightly in level and should not be mixed.

## What runs

| File | Jobs |
|---|---|
| `rescore_adam.yaml`, `rescore_muon.yaml`, `rescore_muown_classical.yaml` | 54 each: 3 batch sizes x 6 learning rates x 3 budgets |

Each job loads the final checkpoint of one phase 1 cooldown, evaluates, and exits: about 11
minutes on one node, about 0.7 GPU-h. All 162 jobs: about 115 GPU-h.

Job index: `(batch_i x 6 + lr_i) x 3 + budget_i`, with budget_i 0 / 1 / 2 = 1x / 2x / 4x;
batch 128 = 0-17, batch 256 = 18-35, batch 512 = 36-53.

```bash
cd $OPT_SCALE_BASE/oellm-autoexp && source $OPT_SCALE_BASE/venvs/autoexp/bin/activate
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/rescore/rescore_muon
```

One tmux session per file. The scoring jobs do not print the line the monitor waits for, so the
monitor may keep running after they are done: check with `squeue` / `sacct` and stop it then.

## Results

```bash
python3 config/experiments/laingsam1/opt_scale/rescore/collect_rescore.py \
    --logs $OPT_SCALE_BASE/logs --csv rescore_phase1.csv
```

prints one table per budget (as in the sweep README) and writes one row per checkpoint.

## Things to know

- The phase 1 folders are only read. Output goes to
  `$OPT_SCALE_BASE/experiments/opt_scale/hyb100m_rescore/`, run names `hyb100m_rescore_...`,
  wandb tag `rescore` instead of `sweep`.
- The same scoring was first used in `../data_order_check` (two checkpoints, 2026-11-18).
  Those two are scored again here; the numbers should agree with it (Muon lr 2e-3 batch 256 4x:
  2.0522, AdamW lr 1e-3 batch 256 4x: 2.1003).
- The sequences are the same at every batch size because each job evaluates
  25,600 / batch size steps. This follows from the configuration; it has not been verified by
  comparing the sequences read.
- Rescoring does not remove the repeated training data in the cooldowns
  (`../data_order_check/README.md`); it only makes the measurement the same everywhere.
- `gen_rescore.py` generates the yaml files from `../sweep100m/phase1_<optimizer>.yaml`.
