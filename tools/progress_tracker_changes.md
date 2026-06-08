# progress_tracker — changes log

## What the tool does

`tools/progress_tracker.py` reads a sweep config YAML, finds all expected training
runs under a results directory, queries `sacct` for Slurm job accounting data, and
prints a per-Slurm-job status table covering:

- Training progress (current iter / total iters, last checkpoint)
- Loss (train + validation)
- Throughput (TFLOP/s/GPU, Tok/s/GPU)
- GPU-hours consumed per job
- Overhead analysis: time-to-first-iteration (TTFI) and low-throughput iterations
- Pass/fail status with error classification

Optionally writes a CSV (`--csv`) and a Markdown summary (`--md`).

Companion tools called internally:
- `megatron_throughput_from_logs.py` — parses and caches per-job throughput
- `low_throughput_analysis.py` — identifies iterations below 30 % of run average
- `gpu_hours.py` — collects job IDs and GPU-h via sacct
- `validate_sweep_runs.py` — resolves sweep config to expected run names

---

## Bug fixes (2026-05-22)

### Bug 1 — Overhead percentage showing > 100 % (e.g. 219817 %)

**Root cause A — cross-cluster Slurm job ID collision**

MN5 (MareNostrum) and Leonardo both use Slurm with independent numeric job ID
sequences.  The same integer (e.g. `40398055`) can refer to a GPU training job
on MN5 *and* a completely different CPU-only job on Leonardo.

When MN5 results are synced to Leonardo's filesystem, the log files are named
`stdout-{MN5_JOB_ID}.log`.  `find_job_ids` picks up these IDs, and Leonardo's
`sacct` returns *Leonardo's* job with that ID — a CPU job with `gres/gpu=0`,
wrong start time, wrong elapsed time.  The log timestamps (from the MN5 run)
then don't match the sacct start time at all, producing TTFI values measured in
**days** (e.g. 2251 GPU-h for a run with 0.0 reported GPU-h).

**Fix — cross-cluster collision detection** (`progress_tracker.py`):

After querying sacct, if the entry reports **zero GPUs** (`gres/gpu` absent from
TRES) but the run's `job.sbatch` declares GPUs (`total_gpus > 0`), the sacct
entry is from a different cluster's job.  Discard it:

```python
if sacct_entry.get("gpus", 0) == 0 and total_gpus > 0:
    sacct_entry = {}     # discard: foreign-cluster job ID collision
    sacct_state = sacct_elapsed = ""
    gpu_hours = external_job_gpu_h.get((run_name, job_id))  # keep CSV fallback
    gpus = total_gpus
```

Effect: `sacct_start = None` → TTFI cases 1 and 2 don't fire → `ttfi_min =
None`.  Iteration data, losses, and throughput from the log are still shown
(they come from the real MN5 training run and are valid).

---

**Root cause B — double-counting warmup in `low_throughput_analysis.py`**

TTFI (Time To First Iteration) is measured as:

```
TTFI = sacct job start → timestamp of the first logged iteration
```

Megatron logs the timestamp at the *end* of each iteration.  So TTFI implicitly
includes the wall time of iteration 1 itself (and, in practice, a few more early
iterations before the first `[timestamp] iteration N/M` line appears).

`analyze_job` previously summed the `elapsed_ms` of **all** iterations whose
throughput was below the 30 % threshold, with no warmup skip.  Warmup iterations
(1–50) are always slow and almost always below the threshold, so their elapsed
time was counted in `low_elapsed_ms`.  But that same elapsed time was already
included in TTFI → double-counted.

**Fix** (`low_throughput_analysis.py`, `analyze_job`):

Skip the first `skip_first_iters` iterations (default 50, the same cutoff used
when computing the reference average) when summing `low_elapsed_ms`.  The warmup
window is attributed entirely to TTFI; low-throughput detection starts at
iteration 51.

```python
# before
low_elapsed_ms = sum(et for _it, et, tflop in rows if tflop < threshold)

# after
rows_post_warmup = [(it, et, tflop) for it, et, tflop in rows if it > skip_first_iters]
low_elapsed_ms = sum(et for _it, et, tflop in rows_post_warmup if tflop < threshold)
```

---

**Root cause B — wrong denominator in the summary table**

The summary table (printed below the per-job table) computed:

```
oh_pct = oh_gpu_h_sum / h
```

where `h` came from `_gpu_collect_job_ids` (finds jobs via `stderr-JOBID.log`
files only).  But `oh_gpu_h_sum` is built from **all** tracked rows, including
jobs that had Slurm accounting records (sacct start/end) but never wrote a
stderr log — for example, a job cancelled in 1 second before the container even
started.  Those jobs' TTFI goes into the overhead numerator while their GPU-h is
absent from `h`, inflating the ratio arbitrarily.

**Fix** (`progress_tracker.py`, summary loop):

Replace `h` with the sum of per-row `gpu_hours` values for that run.  This is
consistent with the data source used to compute `oh_gpu_h_sum`.

```python
# before
oh_pct_val = oh_gpu_h_sum / h * 100.0 if oh_gpu_h_sum is not None and h > 0 else None

# after
run_gpu_h_sum = sum(r["gpu_hours"] for r in run_rows if r.get("gpu_hours") is not None)
oh_pct_val = oh_gpu_h_sum / run_gpu_h_sum * 100.0 if oh_gpu_h_sum is not None and run_gpu_h_sum > 0 else None
```

The same fix was applied to the grand-total line and to both locations in the
Markdown output path.

---

### Bug 2 — QUEUED jobs always show progress as N/A

**Root cause**

When a run is waiting for its next Slurm job to start (status = QUEUED / PENDING),
`stdout_data.get("last_iter")` is `None` because no log has been written yet.
`compute_progress(None, ...)` returns `None`.

The fill-forward loop did handle one special case — if `last_ckpt >= train_iters`
it set `progress = 100 %` — but did nothing for the common mid-run case where
`last_ckpt < train_iters`.

**Fix** (`progress_tracker.py`, fill-forward loop):

1. `ckpt_step` (the decay-phase start iteration from `job.sbatch`, or `None` for
   stable runs) is now stored in every row dict.

2. In the fill-forward loop, after `train_iters` has been propagated from earlier
   rows, estimate progress from the checkpoint when `last_iter` is still `None`:

```python
# before
if r.get("last_iter") is None and r.get("last_ckpt") is not None:
    ti = r.get("train_iters")
    if ti is not None and r["last_ckpt"] >= ti:
        r["last_iter"] = ti
        r["progress"] = 100.0

# after
if r.get("last_iter") is None and r.get("last_ckpt") is not None:
    ti = r.get("train_iters")
    if ti is not None:
        if r["last_ckpt"] >= ti:
            r["last_iter"] = ti
            r["progress"] = 100.0
        elif r.get("progress") is None:
            r["progress"] = compute_progress(r["last_ckpt"], r.get("ckpt_step"), ti)
```

`compute_progress(last_ckpt, ckpt_step, train_iters)` = `100 × (last_ckpt −
ckpt_step) / (train_iters − ckpt_step)`, which correctly handles both stable
runs (`ckpt_step = None → 0`) and decay runs (`ckpt_step` = branching checkpoint
iteration).

---

## New features (2026-05-22)

### Zero-iteration warning

When a job is classified as `FAILED` or `CANCELLED` but never logged a single
training iteration, the `Error` column is prefixed with `"zero iters"`.  This
distinguishes a job that was cancelled mid-training (e.g. 65 % done) from one
that crashed during container startup and never reached the training loop.

### `--machine` flag

```
python progress_tracker.py config/... --machine LEONARDO
python progress_tracker.py config/... --machine MN5
```

Tags every row and CSV entry with the cluster name.  Useful when results from
Leonardo and MN5 are synced into the same directory and the CSV is shared across
teams.  Has no effect on the terminal table.

### `Machine` and `StartTime` columns in CSV output

- **Machine** — value of `--machine` (empty string if not passed).
- **StartTime** — sacct job start time formatted as `YYYY-MM-DD HH:MM:SS`.
  Empty for QUEUED/NOT_LAUNCHED rows.
- **Elapsed** — raw sacct elapsed string (`HH:MM:SS` or `D-HH:MM:SS`), already
  stored in rows; now written to CSV.

---

## Usage examples

```bash
# Basic run (Leonardo local paths)
megatron_exec python tools/progress_tracker.py \
    config/sweep/multilingual_scaling/qwen3_dense_0.1B_ne.yaml \
    --results-dir /leonardo_work/OELLM_prod2026/slaing00/multilingual_scaling/0.1B_ne/training \
    --machine LEONARDO \
    --csv /tmp/0.1B_ne_status.csv \
    --md /leonardo_work/OELLM_prod2026/slaing00/multilingual_scaling/0.1B_ne/training/progress.md

# With explicit monitor-state dirs for better status classification
megatron_exec python tools/progress_tracker.py \
    config/sweep/multilingual_scaling/qwen3_dense_0.1B_ne.yaml \
    --results-dir /leonardo_work/.../0.1B_ne/training \
    --monitor-dirs monitor_state/1778946203 \
    --machine LEONARDO \
    --csv status.csv
```

---

## Known limitations / future work

- **MN5 vs Leonardo separation**: when results from both clusters live in the
  same directory, sacct only returns data for the local cluster's jobs.  MN5 jobs
  show `GPU-h = N/A` unless a previous CSV with MN5 data is passed back via
  `--csv` (the `external_job_gpu_h` fallback).  A proper fix would require either
  running the tracker on MN5 directly, or adding a `--external-csv` path that
  explicitly preserves cross-cluster GPU-h without being overwritten each run.

- **Caching sacct calls**: the sacct batch query is fast for < 1000 job IDs but
  can be slow for very large sweeps.  The throughput data is already CSV-cached
  per job; sacct results could similarly be cached to a temp file (keyed by job
  IDs + timestamp) to avoid re-querying within a monitoring session.

- **Periodic execution inside monitor**: the monitor loop currently does not call
  `progress_tracker`.  A lightweight integration would be to write the `--md`
  file periodically (e.g. every N monitor polls) so the Markdown summary stays
  up to date without a manual invocation.
