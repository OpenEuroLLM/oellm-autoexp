# Runbook: one optimizer's phase 1 sweep (100M model, JUPITER)

For a teammate who takes over one optimizer. Following this page start to finish takes about
30 minutes of your time, then the jobs run for roughly a day.

## What you are running

`phase1_<optimizer>.yaml` trains the 100M-active hybrid MoE with one optimizer at every
combination of 3 batch sizes (128, 256, 512 sequences) and 6 learning rates (2.5e-4 to 8e-3).
Each of the 18 settings is a *ladder* of 4 jobs:

| Job | What it does |
|---|---|
| `stable` | trains at constant learning rate for 6.5B tokens and saves checkpoints at the branch points |
| `cooldown1x`, `cooldown2x`, `cooldown4x` | each starts from a branch checkpoint, cools the learning rate down to 0 and gives the final loss at 2.0B, 4.0B and 8.1B tokens |

So one file is 72 jobs, each on 1 node (4 GPUs): about 850 GPU-h per optimizer (measured:
44-52 GPU-h per ladder). The result is the best learning rate per batch size and token budget.
Background and design: `README.md` in this folder.

## Before you start

- JUPITER account, member of `e-aif-2026sc04-211`, `e-sta-openeurollm` and `datasets`, and
  container access: `../README.md`, section 0.
- Repository cloned at the agreed commit of branch `optimizer_scaling_laws`, and the one-time
  setup and environment check done: `../README.md`, sections 1-3. The setup script also
  installs the Liger kernel these configs need.

## 1. Environment (add to `~/.bashrc`, then open a new shell)

```bash
export OPT_SCALE_BASE=/e/scratch/e-aif-2026sc04-211/users/<you>
export HF_HOME=$OPT_SCALE_BASE/hf_home
# use the data indexes that are already built (read-only); without this line every new
# setting would spend ~25 min building its own
export OPT_SCALE_CACHE=/e/scratch/e-aif-2026sc04-211/users/slaing/cache/megatron_index
```

Every shell you submit from needs:
```bash
cd $OPT_SCALE_BASE/oellm-autoexp
source $OPT_SCALE_BASE/venvs/autoexp/bin/activate
```

## 2. Dry run (renders the job scripts, submits nothing)

```bash
OPT=muown_classical        # or muon / adam
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase1_$OPT --dry-run
```
It should end with `Rendered 72 script(s) (not submitted)`. Open one
`$OPT_SCALE_BASE/experiments/opt_scale/hyb100m_sweep/hyb100m_sweep_*/job.sbatch` and check that the
account, your paths and `--data-cache-path` (the shared cache above) are right.

## 3. First three ladders (learning rate 1e-3 at each batch size)

The jobs are driven by a monitor process that must keep running: it submits the cooldowns when
their checkpoint exists and resubmits a job that crashed. Run it inside tmux, and note which
login node you are on (`hostname`); the session only exists there.

```bash
tmux new -s p1_first
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase1_$OPT --array-subset 8-11,32-35,56-59
# answer y, wait for the "Submitted job" lines, then detach with Ctrl-b d
```

Check them about 15 minutes after they start running (`squeue --me`):

| Check | Command | Healthy |
|---|---|---|
| Training started | `grep "iteration " $OPT_SCALE_BASE/logs/hyb100m_sweep_${OPT}_*bs256_stable-*.out \| head -3` | first step within ~8 min of the job start (much longer means the shared index cache is not being used) |
| Same model and data as everyone else | same lines, `lm loss` of iteration 1 at batch 256 | 12.511 |
| Speed at batch 256 | `... \| tail -3`, `elapsed time per iteration (ms)` | about 4100-4800 (batch 128 half, batch 512 double) |
| GPU memory | same lines, `mem usages` | about 0.91-0.94 |
| Host memory | `grep MemAvailable $OPT_SCALE_BASE/experiments/opt_scale/hyb100m_sweep/hyb100m_sweep_${OPT}_*bs256_stable/node_mem_*.log \| tail -3` | stays above ~300 GB (in kB: 300000000) |

If any of these is off, stop here (section 6) and ask before submitting more.

## 4. The remaining 15 ladders

In a second tmux session (a second, independent monitor):
```bash
tmux new -s p1_rest
python scripts/run_autoexp.py --config-name experiments/laingsam1/opt_scale/sweep100m/phase1_$OPT --array-subset 0-7,12-31,36-55,60-71
```
Never give two monitors the same job numbers: both would submit the job.

## 5. While it runs

- A stable run takes about 7.5-8.6 h. The 1x cooldown starts after roughly a quarter of that,
  the 2x after half, the 4x at the end; a whole ladder is done after about 12 h of running.
- A run at a learning rate that is too high may diverge (loss becomes NaN) and stop early.
  That is a result, not a problem.
- A job that crashes is resubmitted automatically, at most 3 submissions per job, and resumes
  from its last checkpoint.
- Progress: `squeue --me`; per job
  `grep "iteration " $OPT_SCALE_BASE/logs/<job name>-<jobid>.out | tail -1`.
- Files: logs in `$OPT_SCALE_BASE/logs/`; checkpoints, tensorboard, offline wandb and the
  memory log in `$OPT_SCALE_BASE/experiments/opt_scale/hyb100m_sweep/<job name>/`.
- GPU-h used so far:
  ```bash
  sacct -u $USER -S 2026-11-01 -X -n -o ElapsedRaw,AllocTRES%60 \
    | awk '{match($2,/gres\/gpu=([0-9]+)/,m); s+=$1*m[1]} END{printf "%.1f GPU-h\n", s/3600}'
  ```

## 6. Stopping

Stop the monitor first, then the jobs; otherwise the monitor treats the cancellation as a
crash and resubmits.
```bash
tmux attach -t p1_rest      # then Ctrl-C; same for p1_first
scancel <jobid> ...         # or: scancel -u $USER  (all your jobs)
```

## 7. If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| Job fails within 2 minutes with a `PermissionError` on the cache path | a needed index is missing from the shared cache and cannot be written there | unset `OPT_SCALE_CACHE` for that submission (it then builds in your own cache), and tell Sam which setting it was |
| Job sits for 20+ minutes before the first step | it is building a data index | as above |
| `mem usages` climbing towards 0.98, MemAvailable falling steadily | host memory pressure | stop and report; these configs were fixed for this, so it would be new |
| A job reaches its 3rd submission and fails again | a reproducible error | leave it, note the job name and the last lines of its `.err` log |
| `squeue` shows no cooldown jobs hours after a branch point | the monitor is not running | `tmux attach` on the login node where you started it; if it is gone, ask before restarting |

## 8. When it is finished

All 72 jobs are `COMPLETED` in `sacct`, apart from stable runs that diverged. The final losses:
```bash
grep -H "on validation set" $OPT_SCALE_BASE/logs/hyb100m_sweep_${OPT}_*cooldown*.out \
  | sed 's/.*logs\/\(hyb100m_sweep_[^-]*\)-.*lm loss value: \([0-9.E+-]*\).*/\1  \2/' | sort
```
One line per cooldown: job name (optimizer, learning rate, momentum, batch size, budget) and
its final validation loss. Send that table, the GPU-h figure and any job names from section 7.
To upload the runs to wandb from a login node (`wandb` is only installed in the container;
log in once with `... wandb login`):
```bash
SIF=/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif
apptainer exec $SIF /opt/venv/bin/wandb sync -e openeurollm-project -p optimizer_scaling_laws \
  $OPT_SCALE_BASE/experiments/opt_scale/hyb100m_sweep/<job name>/wandb/wandb/offline-run-*
```

## Things not to change

- The generated `.yaml` files: the grid lives in `gen_sweep100m.py`, and everyone must run the
  same files for the results to be comparable.
- Anything in the shared cache directory.
