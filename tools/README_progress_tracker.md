# progress_tracker.py

Sweep run status table for `multilingual_scaling` experiments.

Given a sweep config YAML, the script discovers every expected training run, queries Slurm for accounting data, parses Megatron stdout/stderr logs, and prints a colour-coded per-job table with training metrics, throughput, GPU hours, and status. It can also emit CSV and Markdown reports.

## Usage

```bash
# Basic: print table to terminal (--machine is required: LUMI, LEO or MN5)
python tools/progress_tracker.py config/experiments/multilingual_scaling/training/lumi/qwen3_dense_0.21B.yaml \
    --machine LUMI --results-dir /scratch/project_465002530/multilingual_scaling/0.1B_ne/training

# Write a standalone CSV and Markdown for this cluster + size only
python tools/progress_tracker.py <config.yaml> --machine LUMI --results-dir <dir> \
    --csv status.csv --md status.md

# Merge into the shared all-cluster report (see "Shared all-cluster report" below)
python tools/progress_tracker.py <config.yaml> --machine LUMI --results-dir <dir> --accumulate \
    --csv <results_repo>/training/progress/progress.csv \
    --md  <results_repo>/training/progress/progress.md

# Point to specific monitor-state session directories
python tools/progress_tracker.py <config.yaml> --machine MN5 --results-dir <dir> \
    --monitor-dirs monitor_state/1778164338 monitor_state/1778489512
```

To update the shared report, run one wrapper per cluster. Each wrapper covers all four model sizes:

| Cluster | Script | Notes |
|---------|--------|-------|
| LUMI | `tools/update_progress_lumi.sh` | Defaults for repo, cache and venv paths are built in |
| Leonardo | `tools/update_progress_leo.sh` | Set `RESULTS_REPO` to your clone of the results repo |
| MareNostrum 5 | `tools/update_progress_mn5.sh` | Set `RESULTS_REPO` to your clone of the results repo |

Each wrapper runs `git pull --ff-only` on the results repo, then runs the tracker for every size with `--accumulate`. It does not commit: review `git diff`, then commit and push.

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `config` | *(required)* | Path to the sweep YAML config |
| `--machine` | *(required)* | Cluster the tracker runs on: `LUMI`, `LEO` or `MN5`. Jobs known to local sacct are tagged with it |
| `--foreign-machine` | LEO→MN5, MN5→LEO, LUMI→LEO | Cluster for jobs whose logs are in the local run dirs but that local sacct does not know (or reports a colliding job for) |
| `--accumulate` | off | Merge into the shared CSV given by `--csv` instead of overwriting it, then render `--md` from the merged CSV |
| `--size` | from run names | Value of the `Size` column, e.g. `0.1B_ne` |
| `--results-dir` | auto from config | Directory containing run subdirectories |
| `--prefix-remap` | `/gpfs/…:/home/diana/mn5` | `old:new` path prefix substitution for the config path (only used without `--results-dir`) |
| `--max-elapsed-ms` | `6000` | Drop throughput iterations slower than this (stragglers / checkpoint sync) |
| `--skip-first-iters` | `50` | Warmup iterations excluded from the throughput average |
| `--max-iters` | `500` | Maximum post-warmup iterations to include in the throughput average |
| `--csv` | *(none)* | CSV output path; with `--accumulate`, the shared CSV |
| `--md` | *(none)* | Markdown output path, rendered from the CSV rows |
| `--monitor-dirs` | auto-discovered | Monitor-state session directories containing `*.job.json` files |
| `--compute-storage` | off | Measure `checkpoints/` sizes (`du`). With `--accumulate`, values from a previous run are kept when this flag is off |
| `--cache-dir` | *(none)* | Write the throughput cache and `gpu_hours.csv` under this directory instead of into the results tree |
| `--lightweight` | off | Quick status only: no sacct, no log parsing, no files written |

## Shared all-cluster report

`dense_multilingual_models_scaling_results/training/progress/` holds a single `progress.csv` and a single `progress.md` covering every model size and every cluster. No machine can see the others' sacct or run directories, so the report is built up across clusters:

```
LUMI:  git pull → update_progress_lumi.sh → git commit + push
LEO:   git pull → update_progress_leo.sh  → git commit + push
MN5:   git pull → update_progress_mn5.sh  → git commit + push
```

**Merge rules** (`progress_report.merge_rows`). A run on machine M for size S:

1. Removes M's `local` rows for S, plus rows that other clusters had `inferred` for M on the runs M tracked in this run. M is authoritative for its own jobs.
2. Adds M's new `local` rows. A job that another machine already lists as `local` under the same (Run, JobID) is skipped and a warning is printed, because a matching Slurm ID on two clusters means an ID collision.
3. Adds M's `inferred` rows, which are jobs from synced logs that ran on another cluster, only where no row with that (Run, JobID) exists yet. The real cluster's next run replaces them.
4. Leaves rows of other machines and other sizes unchanged.

Rows are kept in a deterministic order (size, tier, lr, gbsz, stage, tokens, machine, start time), so re-running with the same input changes only `UpdatedAt`. If two clusters push at once and the CSV conflicts, take either side, re-run the tracker on the cluster whose rows were lost, and regenerate the Markdown from the CSV alone:

```bash
python tools/progress_report.py progress.csv --md progress.md
```

`--accumulate` exits early if sacct returns nothing for this machine's jobs. Without sacct, synced foreign jobs would be tagged as local and would shadow the real rows.

**Markdown layout** (`progress_report.render_md`):

- the last tracker run per cluster;
- an overview per size;
- one section per size with:
  - Sources: config and results root per cluster;
  - the training-progress summary, with GPU-h split into `GPU-h(LUMI)`, `GPU-h(LEO)` and `GPU-h(MN5)`;
  - Run locations: OutputPath, DataMixPath and DataPath per run and cluster;
  - the status breakdown, counted per run.

Runs that span clusters are merged into one line:

| Field | How it is merged |
|-------|------------------|
| Clusters | All clusters the run used, e.g. `LEO+LUMI` |
| GPU-h, overheads | Summed per cluster and in total |
| Progress | The furthest point reached on any cluster |
| Status | DONE if any job finished, otherwise the latest job's status |
| Remaining-GPU-h | Taken from the cluster that ran the latest job |
| Ckpt-GB | Summed over clusters, since every cluster keeps its own checkpoints |

## How it works

### 1. Config parsing

`parse_config()` reads the sweep YAML (resolving any Hydra `defaults` sub-configs by way of `_resolve_defaults` from `validate_sweep_runs.py`) and extracts every expected `(run_name, stage, token_budget)` combination. The sweep has two kinds of runs:

- **Stable runs** — one per `(lr, gbsz)` hyperparameter combo, trained to a fixed token budget.
- **Decay runs** — for each combo, only the token budgets in the combo's `center | cross | diagonal` sets are valid decay targets. The script builds this filtered list and skips combinations that are not planned.

Run names are rendered from the job-name template using `render_job_name()` from `validate_sweep_runs.py`.

### 2. Job discovery

For each expected run directory, `find_job_ids()` scans for:
- `config-{id}.yaml` — written at submission time (job may not have started yet)
- `logs/stdout-{id}.log` and `logs/stderr-{id}.log` — written when the job runs

Including config-based IDs lets the tracker detect submitted-but-not-yet-started jobs.

### 3. Slurm accounting (sacct)

All job IDs across every run are batched into a single `sacct` call. The result provides authoritative `State`, `Elapsed`, GPU count (from `AllocTRES`), and start/end timestamps. If `sacct` is unavailable (for example running off-cluster), the tracker falls back to timestamps from the log itself or a previously written CSV.

### 4. Monitor-state events

If `monitor_state/` session directories exist, the tracker loads per-run `{run_name}_*.job.json` files and extracts `action_state` events (for example `time_limit`, `segmentation_fault`, `finished_training`). Each event is matched to a specific Slurm job by way of the sacct time window. This is the primary signal for status classification; stderr log parsing is used as a fallback for manually submitted runs that have no monitor state.

### 5. Log parsing

For every job that has logs:

- **`parse_stdout()`** extracts the Megatron `arguments` block (hyperparams, model size), iteration counters, training and validation losses, and first/last timestamps.
- **`parse_stderr()`** scans for error patterns: segfaults, CUDA OOM, fatal errors, time-limit signals, and the wandb sync completion marker.
- **`load_or_compute_throughput()`** (from `megatron_throughput_from_logs.py`) returns the average TFLOP/s/GPU and Tok/s/GPU, with caching in `<run_dir>/throughput/avg_throughput.csv` so repeated runs are fast.
- **`analyze_job()`** (from `low_throughput_analysis.py`) identifies and totals iterations that ran below 30 % of the reference throughput average (for example during checkpoint saves or NCCL stalls).

### 6. Status determination

`determine_status()` returns `(emoji, status_word, error_description, action_word)`.

Status words and their meaning:

| Status | Meaning |
|--------|---------|
| `NOT_LAUNCHED` | Never submitted on this cluster: no run directory, or a directory with no `job.sbatch`, no `config-*.yaml`, no logs and no checkpoints |
| `CKPT_ONLY` | Never submitted on this cluster, but `checkpoints/` holds `iter_*` checkpoints copied from another cluster. In the merged report, that cluster's job rows take precedence |
| `QUEUED` | Submitted (`job.sbatch` or `config-*.yaml` present) but no iterations logged yet |
| `TRAINING` | Latest job is still running |
| `DONE` | Training reached `train_iters`, wandb synced, or a `finished_training` monitor event fired |
| `FAILED` | Hard error: segfault, OOM, fatal error, exit-code 1, or time-limit on a non-final job |
| `CANCELLED` | sacct state is `CANCELLED*` |

Action words (shown alongside `FAILED`/`CANCELLED`) classify what happened next:

| Action | Meaning |
|--------|---------|
| `AUTO_RESTARTED` | The next job was submitted by the autoexp pipeline (has a `config-{id}.yaml`) |
| `MANUALLY_RESTARTED` | The next job was submitted directly (no config file) |
| `NEW_SESSION` | A manual series ended and a new autoexp session began |
| `NONE` | Latest job; no subsequent run |

### 7. TTFI (time-to-first-iteration) and overhead

For each job the tracker computes:

- **TTFI** — wall time from Slurm job start (sacct) to the first logged training iteration. This captures node allocation, container startup, and data-loader warm-up overhead.
- **Low-throughput overhead** — cumulative time in iterations below 30 % of average throughput (checkpoint saves, NCCL collectives, stalls).
- **Overhead %** — `(TTFI GPU-h + low-TP GPU-h) / total GPU-h × 100`.

### 8. Output

Terminal output is a wide table with one row per Slurm job and a colour-coded status column. A second summary block shows per-run totals. Status colours:

- Green — DONE
- Blue — TRAINING
- Red — FAILED / CANCELLED (terminal)
- Yellow — FAILED + AUTO_RESTARTED
- Magenta — FAILED/CANCELLED + MANUALLY_RESTARTED
- Cyan — FAILED/CANCELLED + NEW_SESSION
- Yellow — QUEUED
- Grey — NOT_LAUNCHED

With `--csv`, one row per job is written with all metrics. With `--md`, a compact Markdown summary table and status breakdown are written (suitable for a shared document).

## Output columns

The CSV schema is the same on every cluster. It is defined as `CSV_FIELDS` in `tools/progress_report.py`, and there is one row per Slurm job. A run with no job yet gets one placeholder row with an empty `JobID`.

| Column | Description |
|--------|-------------|
| `Size` | Model size label, e.g. `0.1B_ne` |
| `Run` | Logical run name, rendered from the sweep config. The run directory may carry a prefix; see `OutputPath` |
| `Machine` | Cluster the job ran on: `LUMI`, `LEO` or `MN5` |
| `Source` | `local` if tracked on `Machine` itself (from sacct); `inferred` if seen in logs synced to another cluster (GPU-h estimated from the logs) |
| `JobID` | Slurm job ID; unique only within one cluster |
| `Owner` | Slurm user, or the user inferred from paths in the log |
| `StartTime` | sacct start time, or the first log timestamp for inferred jobs. Local time of that cluster |
| `Elapsed` | sacct elapsed time |
| `N_ne(B)` | Transformer-block parameter count (billions) |
| `N(B)` | Total parameter count including embeddings (billions) |
| `D(B)` | Token budget (billions) |
| `C(10^18)` | Compute estimate: 6 × N_ne × D (FLOPs × 10⁻¹⁸) |
| `Tier` | center / cross / diagonal |
| `Stage` | stable or decay |
| `TotIter` | Total planned training iterations |
| `CurIter` | Last logged iteration |
| `LastCkpt` | Last checkpoint saved |
| `Prog%` | Training progress within this phase |
| `TrainLoss` | Most recent training loss |
| `ValLoss` | Most recent validation loss |
| `LR` | Learning rate |
| `GBS` | Global batch size, in sequences |
| `MBS` | Micro batch size |
| `Nodes` | Node count from job.sbatch |
| `Workers` | Data loader workers |
| `TFLOP/s/GPU` | Average throughput (warmup-excluded) |
| `Tok/s/GPU` | Average token throughput |
| `TTFI(min)` | Minutes from job start to first iteration |
| `TTFI-GPU-h` | TTFI overhead in GPU-hours |
| `LowTP-time(h)` | Hours lost in low-throughput iterations |
| `LowTP-GPU-h` | GPU-hours lost in low-throughput iterations |
| `Overhead-time(h)` | TTFI + LowTP time (hours) |
| `Overhead-GPU-h` | TTFI + LowTP (GPU-hours) |
| `GPU-h(LUMI)`, `GPU-h(LEO)`, `GPU-h(MN5)` | The job's GPU-h, in the column of its `Machine` |
| `GPU-h` | Billed GPU-hours of the job (sacct; LUMI counts 2 GCDs = 1 GPU) |
| `Overhead%` | Overhead GPU-h / total GPU-h |
| `Remaining-GPU-h` | Per run, filled on the run's latest job row only. A `~` prefix means estimated from stable-run throughput |
| `Ckpt-GB`, `Ckpt-GB-remaining` | Per run, filled on the latest job row and only with `--compute-storage` |
| `Emoji`, `Status` | DONE / TRAINING / FAILED / CANCELLED / QUEUED / NOT_LAUNCHED |
| `Action` | AUTO_RESTARTED / MANUALLY_RESTARTED / NEW_SESSION / NONE |
| `Error` | Short error description if FAILED |
| `OutputPath` | Run directory on `Machine`: Megatron `--save` without `/checkpoints`. Falls back to the submission config, then to the scanned run dir |
| `DataPath` | Tokenized-data root: the common directory of all `Load the _IndexReader from …idx` lines in the job's log |
| `DataMixPath` | The data mix file: Megatron `--data-args-path`, or `--per-split-data-args-path` |
| `Config` | Sweep config the tracker ran with |
| `UpdatedAt` | Time of the tracker run that wrote the row |

---

## Dependent scripts

### `tools/gpu_hours.py`

Standalone GPU accounting tool. Scans a results directory for Slurm log files and queries `sacct` to report per-job and per-experiment GPU-hours.

**What `progress_tracker.py` uses from it:**
- `collect_job_ids(results_dir)` — walks the directory tree and returns `{experiment: [job_ids]}` by matching `slurm-JOBID.log` or `logs/stderr-JOBID.log` files.
- `query_sacct(job_ids)` — runs `sacct` and returns `{job_id: {state, elapsed, gpus}}`.
- `parse_elapsed(s)` — converts a sacct elapsed string (`D-HH:MM:SS` or `HH:MM:SS`) to fractional hours.

The tracker calls these to build its own GPU-h summary table and writes a `gpu_hours.csv` alongside the results.

**Standalone usage:**
```bash
python tools/gpu_hours.py /path/to/results_dir [--output gpu_hours.csv]
```

---

### `tools/megatron_throughput_from_logs.py`

Parses Megatron-style iteration lines to compute average per-GPU throughput (TFLOP/s and Tok/s).

**What `progress_tracker.py` uses from it:**
- `load_or_compute_throughput(log_path, max_elapsed_ms, skip_first_iters, max_iters_used)` — returns `{n_iters, avg_tflop_per_gpu, avg_tok_per_gpu, avg_elapsed_ms, ...}`. Results are cached in `<run_dir>/throughput/avg_throughput.csv` (one row per job ID) so re-running the tracker is fast even for logs with thousands of iterations.

**Caching behaviour:** On the first call for a job, the script parses all iteration lines and writes a per-iteration CSV (`<run_dir>/throughput/<log_stem>.csv`). Subsequent calls load the pre-computed average directly. The cache key includes `max_elapsed_ms`, `skip_first_iters`, and `max_iters_used`; if those change the average is recomputed.

**Standalone usage:**
```bash
python tools/megatron_throughput_from_logs.py /path/to/training_dir [--gpus-per-node 4] [--csv out.csv]
```

---

### `tools/low_throughput_analysis.py`

Identifies training iterations that ran significantly slower than the run average (below 30 % of the average TFLOP/s). These correspond to checkpoint saves, NCCL collective stalls, or other system pauses.

**What `progress_tracker.py` uses from it:**
- `analyze_job(log_path, num_gpus, max_elapsed_ms, skip_first_iters, max_iters_used)` — returns `{time_lost_h, gpu_h_lost, n_low_iters, avg_tflop, num_gpus}` for a single Slurm job's stdout log. Uses the same throughput cache as `megatron_throughput_from_logs.py` for the reference average, then scans all iterations (no elapsed-time filter) to catch genuine stall events.

**Standalone usage (directory-level):**
```bash
python tools/low_throughput_analysis.py /path/to/experiment_dir [--csv out.csv]
```

---

### `scripts/validate_sweep_runs.py`

Validates that every expected sweep run was launched correctly by comparing log contents against the config.

**What `progress_tracker.py` uses from it:**
- `_resolve_defaults(cfg, config_path)` — simulates Hydra defaults-list merging: loads sub-configs referenced in `cfg['defaults']` and fills in missing keys. Required to support split configs where `backend.megatron`, `seq_length`, and `sweep.groups` live in separate YAML files.
- `render_job_name(tpl, nexp, lr, gbsz, seed, stage, stable_tokens)` — substitutes concrete sweep-variable values into the OmegaConf job-name template to produce the expected run directory name.
- `substitute_omegaconf_path_vars(template, ctx)` — replaces `${key}` segments in path templates using a context dict, used to resolve the results base directory.

**Standalone usage (validation):**
```bash
python scripts/validate_sweep_runs.py config/experiments/multilingual_scaling/0.1B_ne.yaml
```

---

## Directory layout expected by the tracker

```
<results_base>/
  <run_name>/
    job.sbatch                        # SBATCH directives (nodes, gpus-per-node, --ckpt-step)
    checkpoints/
      latest_checkpointed_iteration.txt
    logs/
      stdout-<jobid>.log              # Megatron stdout
      stderr-<jobid>.log              # Megatron stderr
    config-<jobid>.yaml               # Written at submission by autoexp pipeline
    throughput/                       # Created by megatron_throughput_from_logs.py
      avg_throughput.csv
      stdout-<jobid>.csv

<project_root>/
  monitor_state/
    <session_id>/
      <run_name>_<n>.job.json         # Monitor-state event files
```
