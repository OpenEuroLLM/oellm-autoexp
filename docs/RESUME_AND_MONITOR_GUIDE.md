# Resume Jobs and Monitor Guide

This document summarizes how to resume timed-out or failed jobs, find sweep indices, understand checkpoint behavior, and work with the monitor for multi-stage (stable → decay) training.

---

## Table of Contents

1. [Stable vs Decay Jobs](#stable-vs-decay-jobs)
2. [Resuming Jobs with the Resume Script](#resuming-jobs-with-the-resume-script)
3. [Manual Decay Resume (Advanced)](#manual-decay-resume-advanced)
4. [Finding Sweep Indices](#finding-sweep-indices)
5. [Checkpoint Saving and Loading](#checkpoint-saving-and-loading)
6. [Monitor and Job Submission](#monitor-and-job-submission)
7. [Common Issues and Fixes](#common-issues-and-fixes)
8. [Config Reference: train_iters and save_extra_steps](#config-reference-train_iters-and-save_extra_steps)

---

## Stable vs Decay Jobs

### Stable Jobs

- Run the initial training phase (e.g., up to 120BT end point).
- Save checkpoints to their own output dir: `results/.../moe_abl_nexp_X_lrY_gbszZ_seed1234_stable/checkpoints/`
- Do **not** have `ckpt_step` set in config (defaults to `None`).
- When resuming, Megatron uses `latest_checkpointed_iteration.txt` to find the latest checkpoint.

### Decay Jobs (decay80BT, decay120BT)

- Continue training from the stable phase at a branching point (80BT or 120BT).
- Load from the **stable sibling's** checkpoint at `start_iter` (the branching iteration).
- Have `ckpt_step: ${backend.megatron.aux.start_iter}` in config.
- Save checkpoints to their own output dir: `results/.../moe_abl_nexp_X_lrY_gbszZ_seed1234_decay80BT/checkpoints/`

---

## Resuming Jobs with the Resume Script

The `resume_timed_out.py` script supports both **stable** and **decay** jobs:

- **Stable jobs** → loads from their own checkpoint directory.
- **Decay jobs** → loads from the corresponding stable sibling's checkpoint directory (resolved automatically via the config's `${sibling.stable.*}` reference).

### List available indices

```bash
# List stable indices with job names
python scripts/resume_timed_out.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
  --config-dir config \
  --list-stable

# List decay indices with job names
python scripts/resume_timed_out.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
  --config-dir config \
  --list-decay
```

### Resume specific indices

```bash
# Resume stable jobs (comma-separated or ranges, spaces after commas OK)
python scripts/resume_timed_out.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
  --config-dir config \
  --indices 0,3,6,48

# Resume decay jobs (loads from stable sibling's checkpoint automatically)
python scripts/resume_timed_out.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
  --config-dir config \
  --indices 1,2,4,5

# Mix stable and decay indices freely
python scripts/resume_timed_out.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
  --config-dir config \
  --indices 0,1,2,3
```

### Dry-run and execution modes

```bash
# Dry-run: print commands without executing
python scripts/resume_timed_out.py ... --dry-run

# Sequential: run jobs one at a time (wait for each to finish)
python scripts/resume_timed_out.py ... --sequential
```

Without `--dry-run` or `--sequential`, the script prints the commands for you to copy-paste into separate terminals.

Each resume command includes:

- `--array-subset <index>` (single index per run)
- `++backend.megatron.load=<checkpoint_dir>` (own dir for stable, stable sibling's dir for decay)

**No need to specify `ckpt_step`** for stable jobs — Megatron uses `latest_checkpointed_iteration.txt`.

---

## Manual Decay Resume (Advanced)

### Resume from decay job's own checkpoint (decay ran and saved checkpoints)

If a decay job timed out after saving its own checkpoints, you need to resume from the **decay** checkpoint directory (not the stable sibling's). This requires clearing `ckpt_step`:

```bash
python scripts/run_autoexp.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
  --config-dir config \
  --array-subset <DECAY_INDEX> \
  ++backend.megatron.load=results/.../moe_abl_nexp_X_lrY_gbszZ_seed1234_decay80BT/checkpoints \
  ++backend.megatron.ckpt_step=null
```

**Why `ckpt_step=null`?** Decay config sets `ckpt_step` to `start_iter` (branching point). When resuming from the decay dir, you must clear it so Megatron uses `latest_checkpointed_iteration.txt` instead.

### Force a specific checkpoint (e.g., avoid tracker overwrite)

```bash
++backend.megatron.load=results/.../checkpoints \
++backend.megatron.ckpt_step=15500
```

---

## Finding Sweep Indices

```bash
# List stable indices
python scripts/resume_timed_out.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_filtering \
  --config-dir config \
  --list-stable

# List decay indices with job names
python scripts/resume_timed_out.py \
  --config-name experiments/swagatam/test_moe_130M_300BT_filtering \
  --config-dir config \
  --list-decay
```

Search the output for the job name (e.g., `moe_abl_nexp_8_lr0.001_gbsz1024_seed1234_decay80BT`) to get its index.

**Note:** Indices depend on the current sweep config. If you add batch sizes or learning rates, indices shift. Re-run `--list-stable` / `--list-decay` after config changes.

---

## Checkpoint Saving and Loading

### How Megatron chooses which checkpoint to load

1. If `ckpt_step` is set → load that specific iteration.
2. Else → read `latest_checkpointed_iteration.txt` in the load directory.

### Multiple runs overwriting the tracker

If two jobs write to the same checkpoint directory, the one that saves **last** overwrites `latest_checkpointed_iteration.txt`, even if its iteration is lower.

**Example:** Job A saves iter 15500. Job B (restart from 13500) saves iter 14000 later. The tracker becomes 14000. A new resume then loads from 14000 instead of 15500.

**Mitigations:**

- Run only one resume per config at a time.
- Or explicitly set `++backend.megatron.ckpt_step=<latest_iter>` when resuming.
- Or fix the tracker: `echo 15500 > .../checkpoints/latest_checkpointed_iteration.txt`

### Inspecting checkpoints

```bash
ls results/.../moe_abl_nexp_8_lr0.008_gbsz1024_seed1234_stable/checkpoints/
cat results/.../checkpoints/latest_checkpointed_iteration.txt
```

---

## Monitor and Job Submission

### Which monitor starts decay jobs?

Decay jobs are started by the **original** monitor session (e.g., `1773407491`) that ran the full sweep. Resumed runs create **new** sessions with only the stable job.

**To start decay jobs:** Run the monitor for the original session:

```bash
python scripts/monitor_autoexp.py --session 1773917843
```

### Will the monitor start the same job twice?

**No.** The monitor only submits jobs when `runtime.submitted` is `False`. Once submitted, it stays `True` until the job is restarted (e.g., on "DUE TO TIME LIMIT") or marked finished.

### Will decay jobs resume from their own checkpoints when the monitor restarts them?

**No.** The monitor resubmits using the same job definition, which loads from the **stable** checkpoint. Decay jobs always restart from the branching point, not from their own latest checkpoint. To resume from a decay checkpoint, use the manual `run_autoexp` commands above.

### Expected job count from resume script

The resume script creates **one SLURM job per index** (one `run_autoexp` call per `--array-subset` index). For 20 stable indices, you get 20 jobs. If you see many more, likely causes:

- Resume script run multiple times
- Original monitor also running and submitting decay jobs

---

## Common Issues and Fixes

### `InterpolationKeyError: sibling.stable.job.base_output_dir not found`

**Cause:** Running a decay job with `--array-subset` only (no stable in plan), so sibling interpolation fails.

**Fix:** Pass an explicit load path: `++backend.megatron.load=results/.../moe_abl_nexp_X_lrY_..._stable/checkpoints`

### Resume loaded from older checkpoint (e.g., 14000 instead of 15500)

**Cause:** Another run overwrote `latest_checkpointed_iteration.txt` with a lower iteration.

**Fix:** Use `++backend.megatron.ckpt_step=15500` or update the tracker file manually.

### `failed to find checkpoint ... in wandb`

**Cause:** Megatron tries to link the loaded checkpoint to a WandB artifact, but the checkpoint was never uploaded (e.g., `WANDB_MODE=offline`).

**Impact:** Harmless. The checkpoint was loaded correctly; only WandB artifact linking failed.

---

## Config Reference: train_iters and save_extra_steps

For `test_moe_130M_300BT_filtering` (seq_length=4096, cooldown_decay_fraction=0.2):

**Formulae:**

- `train_iters` = ceil(tokens / (seq×gbsz))
- `exit_interval` = ceil(120B / (seq×gbsz)) × 0.8
- `save_extra_steps` = [ceil(80B/(seq×gbsz))×0.8, ceil(120B/(seq×gbsz))×0.8]


| gbsz | train_iters (300B) | exit_interval | save_extra_steps [80BT, 120BT] |
| ---- | ------------------ | ------------- | ------------------------------ |
| 64   | 1,144,410          | 366,211       | [244,140, 366,211]             |
| 128  | 572,205            | 183,105       | [122,070, 183,105]             |
| 256  | 286,103            | 91,552        | [61,035, 91,552]               |
| 512  | 143,051            | 45,776        | [30,517, 45,776]               |
| 1024 | 71,526             | 22,888        | [15,259, 22,888]               |


---

## Quick Reference


| Task                         | Command                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| List stable indices          | `resume_timed_out.py ... --list-stable`                                                                                 |
| List decay indices           | `resume_timed_out.py ... --list-decay`                                                                                  |
| Resume stable job            | `resume_timed_out.py ... --indices N`                                                                                   |
| Resume decay (from stable)   | `resume_timed_out.py ... --indices N` (auto-resolves stable sibling's checkpoint)                                       |
| Resume decay (from own ckpt) | `run_autoexp.py ... --array-subset N ++backend.megatron.load=<decay_dir>/checkpoints ++backend.megatron.ckpt_step=null` |
| Start decay jobs             | `monitor_autoexp.py --session <original_session_id>`                                                                    |
| Force specific checkpoint    | Add `++backend.megatron.ckpt_step=<iter>`                                                                               |


## Evaluation one checkpoint:

- need to change the array-subset number to all decay indices
  - 1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35,37,38,40,41,43,44,46,47,49,50,52,53,55,56,58,59
- correspondingly change the bsz_{x} and eval_iters={y} value to match the same number of tokens
- also make the job.log_path reflect the checkpoint number??

```python
python scripts/run_autoexp.py \
--config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
--config-dir config \
--array-subset 23 \
++backend.megatron.load=results/moe_bsz512_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz512_seed1234_decay120BT/checkpoints \
~backend.megatron.data_path \
++backend.megatron.split=null \
'++backend.megatron.train_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all]' \
'++backend.megatron.valid_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all]' \
++backend.megatron.test_data_path=null \
++backend.megatron.ckpt_step=null \
++backend.megatron.skip_train=true \
++backend.megatron.eval_iters=400 \
++job.log_path='${job.base_output_dir}/slurm-%j-eval.log' \
++job.log_path_current='${job.base_output_dir}/current-eval.log'
```



## try with a specific checkpoint number within the folder

say 51000 -- 38770832

```python
python scripts/run_autoexp.py \
--config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
--config-dir config \
--array-subset 23 \
++backend.megatron.load=results/moe_bsz512_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz512_seed1234_decay120BT/checkpoints \
~backend.megatron.data_path \
++backend.megatron.split=null \
'++backend.megatron.train_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all]' \
'++backend.megatron.valid_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all]' \
++backend.megatron.test_data_path=null \
++backend.megatron.ckpt_step=51000 \
++backend.megatron.skip_train=true \
++backend.megatron.eval_iters=400 \
++job.log_path='${job.base_output_dir}/slurm-%j-eval-51000.log' \
++job.log_path_current='${job.base_output_dir}/current-eval-51000.log'
```



will it work for stable?

- note the array subset changes to 21, and we do 45000 (job -- 38770907) and 40000 (job -- 38771061)

```python
python scripts/run_autoexp.py \
--config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
--config-dir config \
--array-subset 21 \
++backend.megatron.load=results/moe_bsz512_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz512_seed1234_stable/checkpoints \
~backend.megatron.data_path \
++backend.megatron.split=null \
'++backend.megatron.train_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all]' \
'++backend.megatron.valid_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all]' \
++backend.megatron.test_data_path=null \
++backend.megatron.ckpt_step=45000 \
++backend.megatron.skip_train=true \
++backend.megatron.eval_iters=400 \
++job.log_path='${job.base_output_dir}/slurm-%j-eval-45000.log' \
++job.log_path_current='${job.base_output_dir}/current-eval-45000.log'
```

```python
python scripts/run_autoexp.py \
--config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
--config-dir config \
--array-subset 21 \
++backend.megatron.load=results/moe_bsz512_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz512_seed1234_stable/checkpoints \
~backend.megatron.data_path \
++backend.megatron.split=null \
'++backend.megatron.train_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all]' \
'++backend.megatron.valid_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all]' \
++backend.megatron.test_data_path=null \
++backend.megatron.ckpt_step=40000 \
++backend.megatron.skip_train=true \
++backend.megatron.eval_iters=400 \
++job.log_path='${job.base_output_dir}/slurm-%j-eval-40000.log' \
++job.log_path_current='${job.base_output_dir}/current-eval-40000.log'
```

```python
python scripts/run_autoexp.py \
--config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
--config-dir config \
--array-subset 21 \
++backend.megatron.load=results/moe_bsz512_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz512_seed1234_stable/checkpoints \
~backend.megatron.data_path \
++backend.megatron.split=null \
'++backend.megatron.train_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all]' \
'++backend.megatron.valid_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all]' \
++backend.megatron.test_data_path=null \
++backend.megatron.ckpt_step=35000 \
++backend.megatron.skip_train=true \
++backend.megatron.eval_iters=400 \
++job.log_path='${job.base_output_dir}/slurm-%j-eval-35000.log' \
++job.log_path_current='${job.base_output_dir}/current-eval-35000.log'
```


## full determinism by dropping bf16, flash attention
- first on 2 nodes 38852188
```
python scripts/run_autoexp.py \
--config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
--config-dir config \
--array-subset 21 \
++backend.megatron.load=results/moe_bsz512_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz512_seed1234_stable/checkpoints \
~backend.megatron.data_path \
++backend.megatron.split=null \
'++backend.megatron.train_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all]' \
'++backend.megatron.valid_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all]' \
++backend.megatron.test_data_path=null \
++backend.megatron.ckpt_step=35000 \
++backend.megatron.skip_train=true \
++backend.megatron.eval_iters=400 \
++backend.megatron.bf16=false \
++backend.megatron.fp16=false \
++backend.megatron.moe_grouped_gemm=false \
++backend.megatron.use_flash_attn=false \
++backend.megatron.attention_backend=unfused \
++backend.megatron.no_load_optim=true \
++job.log_path='${job.base_output_dir}/slurm-%j-eval-35000-fp32-nodes-2.log' \
++job.log_path_current='${job.base_output_dir}/current-eval-35000.log'
```

- then on 4 nodes
```
python scripts/run_autoexp.py \
--config-name experiments/swagatam/test_moe_130M_300BT_bsz_512.yaml \
--config-dir config \
--array-subset 21 \
++backend.megatron.load=results/moe_bsz512_lr_sparsity_grid_A130M_120BT/moe_abl_nexp_64_lr0.001_gbsz512_seed1234_stable/checkpoints \
~backend.megatron.data_path \
++backend.megatron.split=null \
'++backend.megatron.train_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all]' \
'++backend.megatron.valid_data_path=[/leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all]' \
++backend.megatron.test_data_path=null \
++backend.megatron.ckpt_step=35000 \
++backend.megatron.skip_train=true \
++backend.megatron.eval_iters=400 \
++backend.megatron.bf16=false \
++backend.megatron.fp16=false \
++backend.megatron.moe_grouped_gemm=false \
++backend.megatron.use_flash_attn=false \
++backend.megatron.attention_backend=unfused \
++backend.megatron.no_load_optim=true \
++job.log_path='${job.base_output_dir}/slurm-%j-eval-35000-fp32-nodes-4.log' \
++job.log_path_current='${job.base_output_dir}/current-eval-35000.log'
```