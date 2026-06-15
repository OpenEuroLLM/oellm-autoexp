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

## Bug fixes (2026-06-15)

### Bug 7 — TTFI-GPU-h missing when `job.sbatch` encodes GPU count as env var

**Root cause**

`parse_sbatch()` looks for `#SBATCH --gpus-per-node` and `#SBATCH --gres=gpu:N`
directives.  Some single-run configs (e.g. `baby_9b_dense`) store the canonical
sbatch file under `<run_dir>/script/job.sbatch`, while the root-level
`<run_dir>/job.sbatch` (used by the job-chain mechanism) encodes the GPU count
as an `export GPUS_PER_NODE=4` environment variable rather than an `#SBATCH`
directive.  The regex therefore finds `nodes` but returns `gpus_per_node = None`,
making `total_gpus = 0` and causing `ttfi_gpu_h` to remain `None`.

**Fix** (`progress_tracker.py`, `main` sbatch lookup):

After parsing the primary `job.sbatch`, retry with `script/job.sbatch` whenever
`gpus_per_node` is still `None`.  Fields already resolved from the primary file
(`nodes`, `--ckpt-step`) are kept and not overwritten:

```python
sbatch_nodes, sbatch_gpus_per_node, sbatch_ckpt_step = parse_sbatch(run_dir / "job.sbatch")
if sbatch_gpus_per_node is None:
    _n2, _g2, _c2 = parse_sbatch(run_dir / "script" / "job.sbatch")
    sbatch_nodes = sbatch_nodes or _n2
    sbatch_gpus_per_node = _g2
    sbatch_ckpt_step = sbatch_ckpt_step or _c2
```

---

## Bug fixes (2026-06-15)

### Bug 5 — Fill-forward propagating stale metrics into config-only CANCELLED rows

**Root cause**

`find_job_ids` returns the union of IDs from `config-{id}.yaml` files *and* log
files.  A job that was submitted (config written) but cancelled before it ever
wrote a log line therefore still enters the per-job processing loop in `main`.
`parse_stdout` and `parse_stderr` both return all-`None` / all-`False` for a
non-existent file.

The fill-forward pass that follows row collection runs unconditionally on every
row.  When an earlier job in the same run had real training data, the `known`
dict already holds `last_train_loss`, `avg_tflop_per_gpu`, `avg_tok_per_gpu`,
`train_iters`, and other fields.  Those were injected verbatim into the
config-only row, so the CANCELLED stub appeared to show loss and throughput
values from a completely different job execution.

**Fix** (`progress_tracker.py`):

1. After the log-path fallback chain, record whether a log file was actually
   found:

   ```python
   has_log = stdout_log.is_file()
   ```

2. Store `has_log` in every row dict (both the no-job-ids branch and the
   per-job branch).

3. Guard the injection step of the fill-forward so only rows with a real log
   receive inherited values:

   ```python
   # before
   if r.get(fld) is None and fld in known:
       r[fld] = known[fld]

   # after
   if r.get(fld) is None and fld in known and r.get("has_log"):
       r[fld] = known[fld]
   ```

   Rows without a log still *contribute* to `known` if they somehow carry data,
   but they will never receive values injected from a prior job.

---

### Bug 6 — Config-only CANCELLED rows cluttering the table

**Root cause**

A consequence of Bug 5's discovery: even after fixing the fill-forward, a
config-only CANCELLED row still appeared as a row full of `N/A` values.  A job
that was cancelled before it started provides no actionable information — the
run's restart history is already captured by the `action_word` column of the
preceding row.

**Fix** (`progress_tracker.py`, after fill-forward passes):

```python
rows = [r for r in rows if r.get("has_log") or r["status_word"] != "CANCELLED"]
```

Rows are dropped only when **both** conditions hold: no log file exists *and*
the status is CANCELLED.  QUEUED rows (submitted, not yet started) and
NOT_LAUNCHED rows are kept so pending work remains visible.

---

### Bug 3 — `KeyError: 'aux'` and `KeyError: 'sweep'` for configs without a sweep

**Root cause**

`parse_config` accessed `meg["aux"]` and `cfg["sweep"]["groups"]` with hard
dictionary lookups.  Experiment configs that compose sub-configs via the Hydra
defaults list (e.g. `baby_9b_dense.yaml`) do not carry `backend.megatron.aux`
or `sweep` in the YAML that the script loads directly — those keys live in
separate sub-config files that Hydra merges at runtime but the script's
`_resolve_defaults` helper did not bring in for this config layout.

`aux` holds sweep-level metadata (`tokens`, `cooldown_decay_fraction`, etc.) that
is only meaningful when a real sweep is defined.  For configs with `sweep: none`
(`groups: []`), none of the code paths that read `aux["tokens"]` are ever reached,
and `aux.get("cooldown_decay_fraction", 0.2)` already has a safe default.

**Fix** (`progress_tracker.py`, `parse_config`):

```python
# before
aux = meg["aux"]
groups = cfg["sweep"]["groups"]

# after
aux = meg.get("aux", {})
groups = cfg.get("sweep", {}).get("groups", [])
```

Both accesses now return safe empty defaults (`{}` / `[]`) when the key is
absent, so the tracker works for single-run (no-sweep) experiment configs.

---

### Bug 4 — empty table for non-sweep (single-run) configs

**Root cause**

Even after Bug 3's key-error fix, running the tracker on a single-run config
(e.g. `baby_9b_dense.yaml`) produced an empty table with no rows.  Four
separate issues combined to cause this:

**4a — empty `run_specs` when there are no sweep combos**

`parse_config` builds a `combos` list from `sweep.groups`.  When the config has
`sweep: none` (no groups), `combos` is empty, `run_specs` is never populated,
and the row-collection loop has nothing to iterate over.

**Fix** (`progress_tracker.py`, `parse_config` + `main`):

`parse_config` now sets `is_single_run = True` and records `job_name` (from
`cfg["job"]["name"]`) when no combos are found.  In `main`, a fallback block
after the combo loop detects this and synthesises a single entry:

```python
if not run_specs and cfg.get("is_single_run") and cfg.get("job_name"):
    _single_name = cfg["job_name"]
    run_specs = [(_single_name, "stable", 0, "")]
    run_to_gbsz[_single_name] = 0
```

**4b — `_run_id_sets` did not find logs stored inside `logs/`**

Single-run configs write both `config-{id}.yaml` and `slurm-{id}.log` inside
`<run_dir>/logs/`.  `_run_id_sets` looked for `config-*.yaml` only at the
run-dir root, and its `slurm-*.log` fallback only fired when `logs/` did not
exist — so both were missed when `logs/` was present but contained combined
logs rather than split `stdout/stderr` files.

**Fix** (`progress_tracker.py`, `_run_id_sets`):

```python
if logs_dir.is_dir():
    for f in logs_dir.iterdir():
        # split stdout/stderr logs (sweep convention)
        m = re.match(r"(?:stdout|stderr)-(\d+)\.log$", f.name)
        if m:
            log_ids.add(m.group(1))
        # config files stored in logs/ (single-run convention)
        m = re.match(r"config-(\d+)\.yaml$", f.name)
        if m:
            config_ids.add(m.group(1))
    # combined slurm-*.log inside logs/ when no split logs found
    if not log_ids:
        for f in logs_dir.iterdir():
            m = re.match(r"slurm-(\d+)\.log$", f.name)
            if m:
                log_ids.add(m.group(1))
```

**4c — stdout log fallback did not check `logs/slurm-{id}.log`**

The log-reading block in `main` fell back to `<run_dir>/slurm-{id}.log` when
`logs/stdout-{id}.log` was absent.  For single-run configs the combined log is
at `<run_dir>/logs/slurm-{id}.log`, which was never tried.

**Fix** (`progress_tracker.py`, `main` log-file lookup):

```python
if not stdout_log.is_file():
    slurm_log = logs_dir / f"slurm-{job_id}.log"   # check logs/ first
    if not slurm_log.is_file():
        slurm_log = run_dir / f"slurm-{job_id}.log" # then run-dir root
    if slurm_log.is_file():
        stdout_log = slurm_log
        stderr_log = slurm_log
```

**4d — `gpu_hours.csv` written to non-writable shared parent**

For sweep configs `resolved_base` is the user-owned experiment directory.  For
single-run configs it is the shared parent (`/production_training`), which
other users own.  Writing `gpu_hours.csv` there raised `PermissionError`.

**Fix** (`progress_tracker.py`, `main` GPU-h CSV path):

```python
if cfg.get("is_single_run"):
    _csv_anchor = Path(args.csv).parent if args.csv else Path(".")
    gpu_csv_path = _csv_anchor / "gpu_hours.csv"
else:
    gpu_csv_path = resolved_base / "gpu_hours.csv"
```

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
