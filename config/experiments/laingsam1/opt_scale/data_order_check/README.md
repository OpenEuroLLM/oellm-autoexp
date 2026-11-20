# data_order_check: do the ladder cooldowns suffer from repeated training data?

Status (2026-11-18): done for AdamW and Muon. Repeated data costs about 0.003 in final loss at
the 4x budget; the phase 1 results stand. muown_classical was not run.

## Result

Final loss at step 7710 on the fixed validation set (25,600 sequences):

| Optimizer | Phase 1 checkpoint (repeated data) | Fresh-data cooldown | Difference |
|---|---|---|---|
| Muon, lr 2e-3, batch 256 | 2.0522 | 2.0490 | 0.0033 |
| AdamW, lr 1e-3, batch 256 | 2.1003 | 2.0970 | 0.0033 |

- Both pairs differ in the same direction by the same amount, so the effect is probably real,
  but it is below the 0.005 set as the threshold beforehand, and with one seed part of it could
  come from the different data order alone.
- The gap between the optimizers is 0.048 in both columns: comparisons at one budget are not
  affected.
- 4x is the worst case; 1x and 2x have fewer repeats.
- In the first 90 cooldown steps the phase 1 runs had a training loss about 0.02 lower than the
  fresh-data runs, which confirms that the repeats are there.

Phase 2 in `../sweep100m` keeps the cyclic reader, so that its runs compare directly with phase 1;
the bias is the same in both. A sweep that starts fresh can avoid the repeats with
`dataloader_type: single` for the stable run and the cooldowns (cooldowns then need a shorter
evaluation interval, see below; not run as a full ladder yet).

## The concern

Reported by Diana Onutu for the multilingual scaling runs (Mattermost, November 2026; analysis in
`OpenEuroLLM/dense_multilingual_models_scaling_results`, `training/challenges/data_order`).

With `dataloader_type: cyclic`, Megatron shuffles a sample list whose length is
`train_iters x batch size`. A job that resumes from a checkpoint with a different `train_iters`
gets a different shuffle, and skips a different set of samples than the first job actually read.
It therefore trains again on some sequences the model has already seen.

The ladders in `../sweep100m` do exactly this: every cooldown resumes from the stable run with
its own `train_iters`. Estimated share of a cooldown's training data that the stable run had
already read (each such sequence is then seen twice, not more):

| Cooldown | Share of repeated data (estimate) |
|---|---|
| 1x | about 25 % |
| 2x | about 50 % |
| 4x | about 80 % |

The estimate assumes the sample list of a shorter run is the beginning of the list of a longer
one. This holds in the code for datasets that are read less than once per run, and was confirmed
from the cached index files (2026-11-18): for all 522 datasets and all run lengths of the
ladder, the shorter sample list is the beginning of the longer one, and no dataset is read
more than once.

What this can and cannot affect in the phase 1 results:

- Not affected: comparisons at the same budget (optimizer, learning rate, batch size). All runs
  at one budget have the same share of repeated data and are scored on held-out sequences.
- Possibly affected: the absolute loss at each budget, most at 4x, and trends across budgets.

A first look at the existing logs (training loss minus validation loss, stable runs against
cooldowns, 54 ladders) was inconclusive: the gap narrows by 0.01-0.02 in the 2x and 4x
cooldowns, which is the direction repeats would cause, but within the scatter of single
evaluations (about 0.018 each, as each uses only 2,560 sequences).

## The check

For three phase 1 runs at batch 256 (the best learning rate at 4x of each optimizer):

| File | Optimizer | Learning rate |
|---|---|---|
| `check_adam.yaml` | AdamW | 1e-3 |
| `check_muon.yaml` | Muon | 2e-3 |
| `check_muown_classical.yaml` | muown_classical | 4e-3 |

Each file has three jobs:

| Index | Job | What it does | Time limit |
|---|---|---|---|
| 0 | `fresh4x` | the 4x cooldown again from the phase 1 stable checkpoint (step 6168 to 7710), but with `dataloader_type: single`. That reader takes the sample list in order from where the stable run stopped, which is the part the stable run never read. | 5 h (about 2 h expected) |
| 1 | `eval_repeated` | no training: scores the existing phase 1 4x checkpoint on a fixed validation set | 1 h |
| 2 | `eval_fresh` | the same scoring for the `fresh4x` checkpoint; starts by itself once that checkpoint exists | 1 h |

The scoring jobs read the first 25,600 validation sequences (105M tokens) in fixed order, so both
checkpoints of a pair are scored on identical sequences. That removes the 0.018 scatter of the
regular evaluations from the comparison.

`fresh4x` differs from the phase 1 cooldown only in the reader and in evaluating every 120 steps
instead of 240. The shorter interval is needed because `single` does not wrap around on the
validation data, and at 240 the validation set Megatron sizes for the job is one evaluation too
short. The periodic evaluations of `fresh4x` are not used for the comparison.

Cost: about 8 GPU-h per `fresh4x` and well under 1 GPU-h per scoring job, so about 25-30 GPU-h
for all three files.

## Running it

```bash
cd $OPT_SCALE_BASE/oellm-autoexp && source $OPT_SCALE_BASE/venvs/autoexp/bin/activate
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/data_order_check/check_muon
```

Run it in tmux, one session per file: the process stays alive to start `eval_fresh` when the
cooldown has finished. To try the two untested parts cheaply first, submit only the scoring job
of the existing checkpoint with `--array-subset 1`.

Both worked (validation under `single` with the interval of 120, and the scoring jobs, which ended
COMPLETED). The notes written before the first submission:

- validation under `single` inside `fresh4x` (the team has hit an end-of-data error with it);
- the scoring jobs (`skip_train`). They do not print the "after training is done" line, so the
  monitor may not report them as finished; check the Slurm state.

## Reading the result

Both scoring jobs print one line, `validation loss at iteration 7710 on validation set`:

```bash
grep -h "on validation set" $OPT_SCALE_BASE/logs/hyb100m_dataorder_*_eval_*.out
```

| Outcome | Meaning |
|---|---|
| `eval_fresh` lower than `eval_repeated` by more than about 0.005, in all three pairs | repeated data costs something: the 4x losses of phase 1 are slightly pessimistic and budget trends need a caveat |
| differences small or of mixed sign | no measurable effect; the phase 1 numbers stand |

The two checkpoints of a pair also differ in the order of their training data, which by itself
moves the loss a little (no measured value; a few thousandths is the working guess). That is why
the result needs to agree across the three pairs.

Independent of the outcome, new ladders should not change the data order between the stable run
and its cooldowns; `dataloader_type: single` does that.

## Where things are

- Output: `$OPT_SCALE_BASE/experiments/opt_scale/hyb100m_dataorder/`, run names
  `hyb100m_dataorder_<optimizer>_lr<lr>_bs256_<job>`; logs in `$OPT_SCALE_BASE/logs/`.
- wandb: tag `data_order_check` instead of `sweep`, so the runs stay out of the sweep views.
- The phase 1 folders are only read (`--load`); nothing is written there.
- `gen_data_order_check.py` generates the three yaml files from `../sweep100m/phase1_<optimizer>.yaml`.
