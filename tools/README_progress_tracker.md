# progress_tracker.py

Sweep run status table for `multilingual_scaling` experiments.

Given a sweep config YAML, the script discovers every expected training run, queries Slurm for accounting data, parses Megatron stdout/stderr logs, and prints a colour-coded per-job table with training metrics, throughput, GPU hours, and status. It can also emit CSV and Markdown reports.

## Usage

```bash
# Basic: print table to terminal
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml

# Override results path (when cluster path differs from local mount)
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml \
    --results-dir /home/diana/mn5/multilingual_scaling/0.1B_ne/training

# Also write CSV and Markdown
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml \
    --csv status.csv \
    --md status.md

# Point to specific monitor-state session directories
python tools/progress_tracker.py config/experiments/multilingual_scaling/0.1B_ne.yaml \
    --monitor-dirs monitor_state/1778164338 monitor_state/1778489512
```

The helper script `update_progress.sh` wraps the four model-size configs (0.1B, 0.2B, 0.4B, 0.9B) with their cluster paths and monitor-state directories.

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `config` | *(required)* | Path to the sweep YAML config |
| `--results-dir` | auto from config | Directory containing run subdirectories |
| `--prefix-remap` | `/gpfs/…:/home/diana/mn5` | `old:new` path prefix substitution for the config path |
| `--max-elapsed-ms` | `6000` | Drop throughput iterations slower than this (stragglers / checkpoint sync) |
| `--skip-first-iters` | `50` | Warmup iterations excluded from the throughput average |
| `--max-iters` | `500` | Maximum post-warmup iterations to include in the throughput average |
| `--csv` | *(none)* | Optional CSV output path |
| `--md` | *(none)* | Optional Markdown table output path |
| `--monitor-dirs` | auto-discovered | Monitor-state session directories containing `*.job.json` files |

## How it works

### 1. Config parsing

`parse_config()` reads the sweep YAML (resolving any Hydra `defaults` sub-configs via `_resolve_defaults` from `validate_sweep_runs.py`) and extracts every expected `(run_name, stage, token_budget)` combination. The sweep has two kinds of runs:

- **Stable runs** — one per `(lr, gbsz)` hyperparameter combo, trained to a fixed token budget.
- **Decay runs** — for each combo, only the token budgets in the combo's `center | cross | diagonal` sets are valid decay targets. The script builds this filtered list and skips combinations that are not planned.

Run names are rendered from the job-name template using `render_job_name()` from `validate_sweep_runs.py`.

### 2. Job discovery

For each expected run directory, `find_job_ids()` scans for:
- `config-{id}.yaml` — written at submission time (job may not have started yet)
- `logs/stdout-{id}.log` and `logs/stderr-{id}.log` — written when the job runs

Including config-based IDs lets the tracker detect submitted-but-not-yet-started jobs.

### 3. Slurm accounting (sacct)

All job IDs across every run are batched into a single `sacct` call. The result provides authoritative `State`, `Elapsed`, GPU count (from `AllocTRES`), and start/end timestamps. If `sacct` is unavailable (e.g. running off-cluster), the tracker falls back to timestamps from the log itself or a previously written CSV.

### 4. Monitor-state events

If `monitor_state/` session directories exist, the tracker loads per-run `{run_name}_*.job.json` files and extracts `action_state` events (e.g. `time_limit`, `segmentation_fault`, `finished_training`). Each event is matched to a specific Slurm job via the sacct time window. This is the primary signal for status classification; stderr log parsing is used as a fallback for manually submitted runs that have no monitor state.

### 5. Log parsing

For every job that has logs:

- **`parse_stdout()`** extracts the Megatron `arguments` block (hyperparams, model size), iteration counters, training and validation losses, and first/last timestamps.
- **`parse_stderr()`** scans for error patterns: segfaults, CUDA OOM, fatal errors, time-limit signals, and the wandb sync completion marker.
- **`load_or_compute_throughput()`** (from `megatron_throughput_from_logs.py`) returns the average TFLOP/s/GPU and Tok/s/GPU, with caching in `<run_dir>/throughput/avg_throughput.csv` so repeated runs are fast.
- **`analyze_job()`** (from `low_throughput_analysis.py`) identifies and totals iterations that ran below 30 % of the reference throughput average (e.g. during checkpoint saves or NCCL stalls).

### 6. Status determination

`determine_status()` returns `(emoji, status_word, error_description, action_word)`.

Status words and their meaning:

| Status | Meaning |
|--------|---------|
| `NOT_LAUNCHED` | Run directory does not exist |
| `QUEUED` | Directory exists but no iterations have logged |
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

| Column | Description |
|--------|-------------|
| `Run` | Run name (only shown on first job per run) |
| `JobID` | Slurm job ID |
| `N_ne(B)` | Transformer-block parameter count (billions) |
| `N(B)` | Total parameter count including embeddings (billions) |
| `D(B)` | Token budget (billions) |
| `C(10^18)` | Compute estimate: 6 × N_ne × D (FLOPs × 10⁻¹⁸) |
| `Tier` | center / cross / diagonal / stable |
| `Stage` | stable or decay |
| `TotIter` | Total planned training iterations |
| `CurIter` | Last logged iteration |
| `LastCkpt` | Last checkpoint saved |
| `Prog%` | Training progress within this phase |
| `TrnLoss` | Most recent training loss |
| `ValLoss` | Most recent validation loss |
| `LR` | Learning rate |
| `GBS` | Global batch size (tokens × sequences) |
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
| `GPU-h` | Total GPU-hours consumed (from sacct) |
| `Overhead%` | Overhead GPU-h / total GPU-h |
| `Status` | DONE / TRAINING / FAILED / CANCELLED / QUEUED / NOT_LAUNCHED |
| `Action` | AUTO_RESTARTED / MANUALLY_RESTARTED / NEW_SESSION / NONE |
| `Error` | Short error description if FAILED |

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
