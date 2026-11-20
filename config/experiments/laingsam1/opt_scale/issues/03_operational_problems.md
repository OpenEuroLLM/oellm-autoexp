# Problems found while running the 100M sweep: wrong checkpoint loaded, bus errors after saving, repeated data

Status: 2026-11-20. Three separate problems. The first silently produces wrong results and is the
one others are most likely to hit.

## 1. `--ckpt-step` is ignored when a newer rolling checkpoint exists

**What happens.** A job is told to load a specific step with `--load <dir> --ckpt-step N`. If it
also runs with `--non-persistent-ckpt-type global`, and `<dir>` holds a rolling (non-persistent)
checkpoint at step N or later, Megatron loads that newer checkpoint instead. The job starts from
the wrong step, finishes normally and reports a plausible loss.

**Where.** `megatron/training/checkpointing.py`, `_load_base_checkpoint`.

**When it bites.** Any job that branches from a run that is still training and writing rolling
checkpoints into the same folder. In this sweep, the decay phases branch from the constant-rate
run. If a decay job waits in the queue while its parent trains on, it branches too late.

**Impact here.** 21 of 26 jobs of one batch were affected: 16 branched one rolling interval late,
and 5 loaded a step past their own end and trained nothing. An earlier batch of 162 jobs was
checked and none was affected.

**How to detect it.** The log line

```
successfully loaded checkpoint from <dir> [ t 1/1, p 1/1 ] at iteration N
```

must show the step given in `--ckpt-step`.

**Workaround.** Do not pass `--non-persistent-ckpt-type` to a job that uses `--ckpt-step`. In
autoexp: `non_persistent_ckpt_type: null` on those stages. Confirmed on 24 reruns that loaded
from folders containing newer rolling checkpoints; the 84 jobs run since then all loaded the
requested step.

**Suggested fix in Megatron.** When `--ckpt-step` is given, load that step, or fail if it does
not exist. Never substitute another checkpoint.

## 2. Bus error on one rank a few seconds after a checkpoint save

**What happens.** A training job dies with `Fatal Python error: Bus error` (signal 7) on one GPU
process, 10 to 15 seconds after a permanent checkpoint has been written. The other ranks are then
killed.

**How often.** On 2026-11-20, 6 of 51 runs that reached their second permanent save, all within
seven minutes of each other, on different nodes. The same failure was seen on 2026-11-16.

**Impact.** No data loss: in all six the checkpoint was complete, and jobs branching from it
loaded it correctly. The run has to be resubmitted and waits in the queue again.

**What is known.**
- Model: 100M-active hybrid MoE, one node, 4 GPUs, checkpoint about 12 GB, `torch_dist` format.
- Host memory was not exhausted in the run inspected (325 GB available just before the crash).
- An earlier form of this problem was reduced by `create_attention_mask_in_dataloader: false`
  and `mmap_bin_files: false`, which lowered host memory use a lot. It has not gone away.

**Not known.** The cause. The Python traceback of the failing rank lists data-loader queue
feeder threads; the signal is caught by UCX's handler.

## 3. A resumed job with a different `train_iters` rereads training data

**What happens.** With `dataloader_type: cyclic`, the order of the training data depends on
`train_iters`. A job that loads a checkpoint and continues with a different `train_iters` gets a
different shuffle, so it rereads part of what the parent run already saw.

**Impact here.** The decay phases reread about 25 % / 50 % / 80 % of their data at the 1x / 2x /
4x budgets. Measured cost at 4x: 0.003 in validation loss, the same for AdamW and Muon. Since
every run at one budget has the same bias, comparisons within a budget are not affected. Loss
against tokens looks slightly too good at the long budget.

**Workaround.** For a run that starts fresh and is not branched, use `dataloader_type: single`.

## Smaller notes

- **A cancelled job is not restarted** by the autoexp monitor; a failed one is.
- **Slurm launch and exit hangs:** on 2026-11-19, 3 of about 180 jobs hung at launch ("Unable to
  confirm allocation") and 2 hung at exit until the time limit.
- **Start `tmux` and the launcher as separate commands.** Typed on one line, the launcher runs
  outside the session and dies with the terminal.

## Where things are

| | |
|---|---|
| Full history | `config/experiments/laingsam1/opt_scale/LOG.md` |
| Data-order check | `config/experiments/laingsam1/opt_scale/data_order_check/README.md` |
| Generator with the workaround | `config/experiments/laingsam1/opt_scale/sweep100m/gen_sweep100m.py` |
