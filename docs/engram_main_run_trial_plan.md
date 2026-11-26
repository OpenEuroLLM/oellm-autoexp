# Main-run Engram trial plan: 62,000 → 63,000

Status: **plan only, nothing executed.** The trial tests the per-stream Engram gates (`bg_read`) plus the
scale-invariant query / `logit_scale` fix on the main run's real trajectory for 1,000 steps, with a reset
to 62,000 if the loss regresses.

## 1. Objective

The offline rescore says the fix restores gate **spread** but not selection correlation. The remaining
question is whether the opened per-stream gates help the actual loss. The main run is the real trajectory
and already has 264 nodes; 1,000 steps is ~0.6% of the remaining compute.

## 2. Current state

- Main run: p264 job `48871798`, 264 nodes, host011. `ckpt` 62,000, rolling 62,450; the next
  boundary is 63,000.
- Resolved config: `train_iters` 236733, `exit_interval` 236733, `eval_interval` 1000, `eval_iters` 10,
  `arch_metrics_interval` 500, `save_interval` 1000, rolling every 50, `global_batch_size` 1024,
  `micro_batch_size` 1.
- Engram: `engram_branch_gates: false`, `engram_branch_query: read`, `engram_gate_signed_sqrt: true`,
  `engram_clip_rownorm: 7.0`. The fix keys (`engram_gate_logit_scale`, `engram_gate_query_eps`) are
  **not** in the main run's Megatron yet.
- 63,000 is both a persistent save and a validation point (both intervals are 1000).

## 3. Design

- **Arm:** `bg_learned` (per-stream gates, learned query mix) + `engram_gate_logit_scale: true` +
  `engram_gate_query_eps: 1e-12`.
- **Why learned:** the query is `read + lam*(streams-read)` with `lam=0` at init, so step 0 is exactly
  the current single gate (function-preserving, no spike) and the model learns how much stream to mix
  in. This is the best-effort compromise: it can reach the original stream query without the abrupt
  `bg_stream` spike. `bg_read` and `bg_stream` are the controls.
- **Norm in all cases:** the query/key `hidden_norm`/`key_norm` and the value `conv_norm` are applied
  in every query/gate mode; `hidden_norm` uses the fix's `engram_gate_query_eps` (1e-12), so the query
  norm is scale-invariant, and `logit_scale` sets the size. `engram_output_scale` is a separate boost.
  The `learned` mix is learned before the norm.
- **All gates:** `engram_branch_gates` adds a gate for **every** mHC stream (all 4) at every Engram
  layer; `learned`, `read` and `stream` differ only in the query source. The `fix` is the single gate.
  The separate `engram_gate_scale` (false) and the mHC read/write/mix gates are separate mechanisms.
- **Init and spike:** layer 2 `logit_scale = 0.3` (spread ~0.43); layer 52 keeps the mean (~0.47, it
  is already open). This is a **gate-opening** value, not the function-preserving mean query RMS,
  because `lam=0` + a dead gate gives zero gradient to `lam` (double zero -> non-learnable). With the
  gate opened, `lam` starts at 0 (the read) and learns the stream; expect a short spike.
- **Length:** 1,200 steps, 62,800 → 64,000; `exit_interval: 64000`; early validation at 63,000 (200
  steps) and 64,000 (1,200 steps).
- **Baseline:** the main run's own 62,800 → 63,000 and 62,800 → 64,000 (paired, same data order).
- **Reset:** if the loss regresses, keep the main run's 62,800 rolling save and continue without the fix.
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
- The main run reaches 62,800; the rolling save `ckpt_rolling/iter_0062800`; the fork translation
  persists it into the fork's own ckpt (the rolling save itself is overwritten later).
- The user stops job `48871798`. The main tree is not edited.

### Step 2 — mean query RMS at 62,000
- Export the 62,800 rolling save with the standard HF export (the same export the offline rescore used).
- Run the diag on the export:
  ```
  $R/autoexp_env/bin/python scripts/rluukkon/engram_gate_diag.py \
    --model-dir <export_62800> --out <diag_62800> \
    --docs 16 --fixed-query --logit-scales 0.025,0.1,0.3,1.0 --sweep-layer 2
  ```
  Read `query_rms` for layers 2 and 52; layer 2's `logit_scale` is fixed at 0.3, layer 52 keeps the mean.
  Pass `--docs 16` explicitly (an empty env var caused the earlier `invalid int value: ''` failure).

### Step 3 — surgery on 62,000
- Run the surgery with layer 2 = 0.3 and layer 52 = the mean:
  ```
  SRC=<main ckpt_rolling dir> IT=62800 OUT_ROOT=<fork root> \
  READ_SCALE=2:0.3,52:<mean52_read> \
  STREAM_SCALE=2:0.3,52:<mean52_stream> \
  bash scripts/korbi/production/engram_branch_fix_surgery.sh
  ```
- The trial uses `bg_learned` (the `--query-mix` surgery + the gate-opening `logit_scale`); the `bg_read`
  and `bg_stream` checkpoints with the same `logit_scale` are the controls; the `fix` too.

### Step 4 — trial config and snapshot
- Add the trial config (§6) to the main repo; sync to `autoexp_engramfix7`; read `SYNC_STAMP`.

### Step 5 — trial job
- Enqueue the trial on the 128-node c136 host (the 134 nodes available now), in parallel with the
  main run's normal continuation on its 264 nodes.
- `load` = the `bg_learned` 62,800 checkpoint; `exit_interval: 64000`; 1,200 steps.

### Step 6 — evaluation
- Loss: the trial's loss at 63,000 (200 steps) and 64,000 (1,200 steps) vs the main run's **same steps**
  (62,800 → 63,000 and 62,800 → 64,000) on the 264
  nodes, paired by data order. That is the clean comparison; the previous 1k steps is only a fallback.
- Spike: the loss at the first ~50 steps and the recovery by ~500 steps; reset if it has not recovered.
- Arch metrics: per-stream gate spread, `query_rms`, `read_sum`, and the `query_mix` (`lam`) every 10
  steps; if `lam` stays at zero, the stream adds nothing.
- Validation at 64,000 (`eval_iters: 10`) vs the 63,000 baseline.
- Downstream export/eval at 64,000 if the loss is not decisive.

### Step 7 — decision
- **Win:** the trial's loss beats the main run's same steps beyond noise and the gates open → switch
  the main run's tree to the trial tree at 64,000.
- **Neutral:** loss within noise, gates open → the gate is not the bottleneck; keep the main tree; drop
  the per-stream gates.
- **Loss:** loss regresses → keep the main tree; the trial tree is discarded.

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
    load: ${aux.surgery}/bg_learned/ckpt
    ckpt_step: 62800
    engram_branch_gates: true
    engram_branch_query: learned
    engram_gate_logit_scale: true
    engram_gate_query_eps: 1e-12
    exit_interval: 64000
    arch_metrics_interval: 10
    save_extra_steps: [64000]

job:
  name: ambitious_v4b_learned_trial_bg_learned
  base_output_dir: ${aux.kv_root}/runs/v4b_learned/trial_bg_learned
  metadata:
    pool_priority: 100

slurm:
  sbatch:
    nodes: 128
```

The trial writes only `${job.base_output_dir}` (`runs/v4b_learned/trial_bg_read`); the main tree
`runs/v4b_learned/main` is untouched.

## 7. Risks and rollback

- The main run continues on its 264 nodes; the trial runs in parallel on 128 nodes. The main tree is
  untouched; the trial has its own tree. The switch to the trial tree happens only on a win.
- `learned` at `lam=0` with a **dead** gate has zero gradient to `lam` (double zero -> non-learnable).
  Layer 2's `logit_scale = 0.3` opens the gate at init; then `lam` learns and expect a short spike. The `bg_read`
  and `bg_stream` controls tell whether the read or the stream itself helps.
- The `logit_scale` init depends on the 62,800 mean query RMS. If the export/diag is wrong, the step-0
  loss jumps and the trial is invalid.
- The 1k steps are ~0.6% of the remaining compute; the reset keeps the loss.
- The host004 arms remain the controlled tiebreaker if the trial is ambiguous.

## 8. Open questions

- Confirm the fork trigger at 62,800 (the rolling save; the user stops nothing, the main continues).
- Confirm 1,200 steps (62,800 → 64,000) with the early eval at 63,000.
- Confirm `bg_learned` with layer 2 = 0.3 and the `bg_read`/`bg_stream` controls with the same
  `logit_scale`.
- Set the reset threshold (e.g. loss delta > +0.0005 beyond noise) and the spike tolerance.
- Confirm the `learned` mix's `lam` is free to learn (the `--query-mix` surgery, zero Adam state).
- Decide the host004 arms' role (tiebreaker or drop).
- Land the Megatron commit `bd6b57856` in the main repo before the snapshot.
