# Multilingual Scaling Experiments (slaing00)

Qwen3-dense multilingual scaling runs on Leonardo: 0.1B, 0.2B, 0.4B, and 0.9B NE models.
Experiment parameters from: https://docs.google.com/spreadsheets/d/17LXPs6O9_6cL7P2KVEJF8LTy5zhy0qLNXEVGRCYOzhY/edit?gid=0#gid=0

## Running (as slaing00)

Submit via the orchestrator sbatch files, e.g.:

```bash
sbatch config/experiments/slaing00/multilingual_scaling/0.9B_ne_orch_center.sbatch
```

Or run directly:

```bash
PYTHONPATH=. python3 scripts/run_autoexp.py \
    --config-name experiments/slaing00/multilingual_scaling/0.9B_ne \
    backend.megatron.aux.priority_tier=center
```

## Adapting for a new user

Copy the configs to your own directory and update the following hardcoded paths:

### 1. Output directory (`job.base_output_dir`)

Each config has a line like:

```yaml
base_output_dir: "/leonardo_work/OELLM_prod2026/slaing00/multilingual_scaling/<model>/training/${job.name}"
```

Change `slaing00` to your Leonardo username (or choose any path under `$WORK` that you own).
The directory will be created automatically by the framework at job start.

### 2. Data cache (`backend.megatron.data_cache_path`)

Each config has:

```yaml
data_cache_path: /leonardo_work/OELLM_prod2026/slaing00/cache/MEGATRON_262k/${job.name}
```

Change `slaing00` to your own username. **Create the parent directory before submitting:**

```bash
mkdir -p /leonardo_work/OELLM_prod2026/<your_username>/cache/MEGATRON_262k
```

Each job gets its own subdirectory (`${job.name}`) to avoid write conflicts between concurrent runs.

### 3. Sbatch orchestrator files

The `.sbatch` files in this directory hardcode the repo path:

```bash
cd /leonardo_work/OELLM_prod2026/slaing00/oellm-autoexp
```

and the config name:

```bash
--config-name experiments/slaing00/multilingual_scaling/0.9B_ne
```

Update both to reflect your checkout location and config directory.

### 4. W&B (optional)

`wandb_project` and `wandb_entity` are set to the shared project (`dense_multilingual_scaling` / `openeurollm-project`). Leave these unchanged if you have access to the shared W&B org; change them if you want runs logged to your own project.

### Summary checklist

| Field | File(s) | What to change |
|---|---|---|
| `job.base_output_dir` | `*.yaml` | Replace `slaing00` with your username |
| `backend.megatron.data_cache_path` | `*.yaml` | Replace `slaing00` with your username; `mkdir -p` the parent |
| `cd ...` path | `*.sbatch` | Point to your repo checkout |
| `--config-name` | `*.sbatch` | Point to your config directory |
