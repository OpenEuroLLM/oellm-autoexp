# Tools

## progress_tracker.py

Reads a sweep YAML config and produces a per-Slurm-job status table with training metrics, GPU hours, overhead analysis, and cluster attribution.

### Usage

**0.1B:**
```bash
megatron_exec tools/progress_tracker.py config/experiments/slaing00/multilingual_scaling/0.1B_ne.yaml --results-dir /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.1B_ne/training --machine LEO --compute-storage --csv /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.1B_ne/training/progress_0.1B.csv --md /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.1B_ne/training/progress_0.1B.md
```

**0.9B:**
```bash
megatron_exec tools/progress_tracker.py config/experiments/slaing00/multilingual_scaling/0.9B_ne.yaml --results-dir /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.9B_ne/training --machine LEO --compute-storage --csv /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.9B_ne/training/progress_0.9B.csv --md /leonardo_work/OELLM_prod2026/experiments/multilingual_scaling/0.9B_ne/training/progress_0.9B.md
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--results-dir` | from config | Path to training results directory (where per-run subdirs live) |
| `--machine` | `LEO` | Local cluster tag written to CSV (`LEO` on Leonardo, `MN5` on MareNostrum) |
| `--csv PATH` | — | Write per-job CSV to this path |
| `--md PATH` | — | Write per-run markdown summary to this path |
| `--compute-storage` | off | Measure checkpoint directory sizes via `du -sb` (slow on large trees) |
| `--skip-first-iters` | 50 | Skip first N iterations when averaging throughput (warmup) |
| `--max-iters` | 500 | Max iterations to use for throughput average |
| `--max-elapsed-ms` | 6000 | Drop iterations with elapsed time above this threshold (ms) |
| `--monitor-dirs` | auto | Monitor-state session directories (auto-discovered from `monitor_state/` if omitted) |

### Outputs

**Terminal** — per-job detail table and per-run summary with status, progress, throughput, GPU hours, overhead, and remaining estimates.

**CSV** — one row per Slurm job. Key columns:

| Column | Description |
|---|---|
| `Run` | Experiment name |
| `JobID` | Slurm job ID |
| `Machine` | Cluster this job ran on (`LEO` / `MN5`) |
| `RunClusters` | Aggregate cluster tag for the whole run (`LEO` / `MN5` / `MIX`) |
| `Owner` | Submitting user (from sacct for local jobs; inferred from log paths for cross-cluster jobs) |
| `StartTime` | Job start timestamp from sacct |
| `Elapsed` | Job wall-clock elapsed time from sacct |
| `GPU-h(LEO)` / `GPU-h(MN5)` | GPU hours attributed to each cluster |
| `GPU-h` | Total GPU hours for this job |
| `Overhead%` | `(TTFI + LowTP) / GPU-h` |
| `Remaining-GPU-h` | Estimated GPU hours remaining (latest job row only) |

**Markdown** — one row per run with aggregated GPU hours, overhead, progress, and storage.

### Overhead

- **TTFI** (time-to-first-iteration): GPU hours from job start to first logged training step. Requires sacct timing — unavailable for cross-cluster jobs.
- **LowTP**: GPU hours at throughput below 30% of the run average. Computed from log timestamps, available for all jobs.
- **Overhead%** = `(TTFI_GPU-h + LowTP_GPU-h) / GPU-h × 100`

### Cross-cluster notes

MN5 and Leonardo share Slurm job ID sequences, so a given ID may refer to different jobs on each cluster. The tracker detects collisions via GPU count heuristics and:

- Tags affected jobs with the foreign cluster name
- Infers GPU-h from log throughput (`efficient_GPU-h + LowTP_GPU-h`)
- Reports `TTFI = N/A` (requires sacct start time, unavailable cross-cluster)
- Infers job owner from filesystem paths in the log
