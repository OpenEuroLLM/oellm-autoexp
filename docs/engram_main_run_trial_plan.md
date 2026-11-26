# Main-run Engram trial plan: 62,000 → 63,000

Status: **plan only, nothing executed.** The trial tests the per-stream Engram gates (`bg_read`) plus the
scale-invariant query / `logit_scale` fix on the main run's real trajectory for 1,000 steps, with a reset
to 62,000 if the loss regresses.

## 1. Objective

The offline rescore says the fix restores gate **spread** but not selection correlation. The remaining
question is whether the opened per-stream gates help the actual loss. The main run is the real trajectory
and already has 264 nodes; 1,000 steps is ~0.6% of the remaining compute.

## 2. Current state

- Main run: p264 job `48871798`, 264 nodes, host011. `ckpt` 61,000, rolling 61,382.
- Resolved config: `train_iters` 236733, `exit_interval` 236733, `eval_interval` 1000, `eval_iters` 10,
  `arch_metrics_interval` 500, `save_interval` 1000, rolling every 50, `global_batch_size` 1024,
  `micro_batch_size` 1.
- Engram: `engram_branch_gates: false`, `engram_branch_query: read`, `engram_gate_signed_sqrt: true`,
  `engram_clip_rownorm: 7.0`. The fix keys (`engram_gate_logit_scale`, `engram_gate_query_eps`) are
  **not** in the main run's Megatron yet.
- 62,000 is both a persistent save and a validation point (both intervals are 1000).

## 3. Design

- **Arm:** `bg_read` (per-stream gates, query read) + `engram_gate_logit_scale: true` +
  `engram_gate_query_eps: 1e-12`.
- **`bg_read` is already "all gates":** `engram_branch_gates` adds a gate for **every** mHC stream
  (all 4) at every Engram layer. `bg_stream` is the same gate set with the query taken from the stream
  instead of the read — not a different gate set. The `fix` is the single gate. The separate
  `engram_gate_scale` term (currently false) is not part of `engram_branch_gates`; enable it with
  `--gate-scale` only if the trial is extended to it. The mHC read/write/mix gates are separate too.
- **Init:** function-preserving. `bg_read` at init is exactly the current single gate (stream params are
  copies); `logit_scale` = the mean query RMS at 62,000 (not 53,000); the query eps is a constant.
- **Length:** 1,000 steps, 62,000 → 63,000; `exit_interval: 63000`; validation at 63,000.
- **Baseline:** the main run's own 61,000 → 62,000 (previous 1k steps, same data order).
- **Reset:** if the loss regresses, keep the 62,000 checkpoint and resume the main run without the fix.
- **`bg_stream`:** only if `bg_read` wins; not in the first trial.

## 4. Prerequisites (before the trial)

1. **Megatron fix in the main repo.** Fetch worktree commit `bd6b57856` (`engram_gate_query_eps`,
   `engram_gate_logit_scale`, `query_rms` / `read_sum` metrics) into `submodules/Megatron-LM` of the main
   repo, then commit the superproject gitlink. The main run's current Megatron lacks these keys.
2. **Trial config in the main repo** (§6) plus a fresh snapshot `sync_to_marenostrum.sh autoexp_engramfix7`
   so `SYNC_STAMP` matches.
3. **62,000 checkpoint:** the main run's persistent save `iter_0062000` and the 62,000 validation.
4. **User stops the main run at 62,000.** Per AGENTS.md, no Slurm cancel without explicit permission.

## 5. Step-by-step (not executed)

### Step 1 — stop at 62,000 (user action)
- The main run reaches 62,000; persistent save `iter_0062000`; the 62,000 validation.
- The user stops job `48871798`. The main tree is not edited.

### Step 2 — mean query RMS at 62,000
- Export the 62,000 checkpoint with the standard HF export (the same export the offline rescore used).
- Run the diag on the export:
  ```
  $R/autoexp_env/bin/python scripts/rluukkon/engram_gate_diag.py \
    --model-dir <export_62000> --out <diag_62000> \
    --docs 16 --fixed-query --logit-scales 0.025,0.1,0.3,1.0 --sweep-layer 2
  ```
  Read `query_rms` for the gate layers (2 and 52), read and stream. Those become the `logit_scale` values.
  Pass `--docs 16` explicitly (an empty env var caused the earlier `invalid int value: ''` failure).

### Step 3 — surgery on 62,000
- Run the surgery with the 62,000 means:
  ```
  SRC=<main ckpt dir> IT=62000 OUT_ROOT=<surgery root> \
  READ_SCALE=2:<mean2_read>,52:<mean52_read> \
  STREAM_SCALE=2:<mean2_stream>,52:<mean52_stream> \
  bash scripts/korbi/production/engram_branch_fix_surgery.sh
  ```
- The trial uses `bg_read`; the `fix` checkpoint (single gate) is available as a control if needed.

### Step 4 — trial config and snapshot
- Add the trial config (§6) to the main repo; sync to `autoexp_engramfix7`; read `SYNC_STAMP`.

### Step 5 — trial job
- Enqueue on the p264 pool / host011, 264 nodes, the main run's account/QOS, `pool_priority: 100`.
- `load` = the `bg_read` 62,000 checkpoint; `exit_interval: 63000`; 1,000 steps.

### Step 6 — evaluation
- Loss: the trial's 1,000-step loss vs the baseline's 61,000 → 62,000 (paired, same data order).
- Arch metrics: per-stream gate spread, `query_rms`, `read_sum` every 10 steps.
- Validation at 63,000 (`eval_iters: 10`) vs the 62,000 baseline.
- Downstream export/eval at 63,000 if the loss is not decisive.

### Step 7 — decision
- **Win:** loss improves beyond noise and the gates open → continue the main run on the trial tree; then
  consider `bg_stream`.
- **Neutral:** loss within noise, gates open → the gate is not the bottleneck; continue the main run
  without the fix; drop the per-stream gates.
- **Loss:** loss regresses → reset to 62,000; resume the main run without the fix.

## 6. Trial config (draft)

```yaml
# @package _global_
defaults:
  - v4b_learned_mn5
  - _self_

aux:
  surgery: /gpfs/scratch/ehpc533/hybrid_nucleus/production/engram_branch_fix/surgery
  kv_root: /gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116

backend:
  megatron:
    load: ${aux.surgery}/bg_read/ckpt
    ckpt_step: 62000
    engram_branch_gates: true
    engram_branch_query: read
    engram_gate_logit_scale: true
    engram_gate_query_eps: 1e-12
    exit_interval: 63000
    arch_metrics_interval: 10
    save_extra_steps: [63000]

job:
  name: ambitious_v4b_learned_trial_bg_read
  base_output_dir: ${aux.kv_root}/runs/v4b_learned/trial_bg_read
  metadata:
    pool_priority: 100

slurm:
  sbatch:
    nodes: 264
```

The trial writes only `${job.base_output_dir}` (`runs/v4b_learned/trial_bg_read`); the main tree
`runs/v4b_learned/main` is untouched.

## 7. Risks and rollback

- The main run is stopped at 62,000 by the user; the main tree is untouched; the trial has its own tree.
- The function-preserving init depends on the 62,000 mean query RMS. If the export/diag is wrong, the
  step-0 loss jumps and the trial is invalid.
- The 1k steps are ~0.6% of the remaining compute; the reset keeps the loss.
- The host004 arms remain the controlled tiebreaker if the trial is ambiguous.

## 8. Open questions

- Confirm the main run's stop at 62,000 (user).
- Confirm 1,000 steps (62,000 → 63,000) is the trial length.
- Confirm `bg_read` first and `bg_stream` only on a win.
- Set the reset threshold (e.g. loss delta > +0.0005 beyond noise).
- Decide the host004 arms' role (tiebreaker or drop).
- Land the Megatron commit `bd6b57856` in the main repo before the snapshot.
