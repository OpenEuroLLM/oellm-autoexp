# Main-run Engram fork trial plan: 62,900 → 64,000

Status (2026-11-27): **the fork is the main run** (decision after the 64k tie, §6b); started 2026-11-26 on c136 host004 (snapshot `autoexp_engramfix7`). The trial
tests the per-stream Engram gates (`bg_learned`) plus the scale-invariant query / `logit_scale` fix on the main
run's real trajectory for 1,100 steps, in parallel on 128 nodes, with the main run's own same steps as the paired
baseline.

## 1. Objective

The offline rescore says the fix restores gate **spread** but not selection correlation. The remaining
question is whether the opened per-stream gates help the actual loss. The main run is the real trajectory;
1,100 steps is ~0.5% of the remaining compute.

## 2. Current state

- Main run: p264 job `48871798`, 264 nodes, host011. Rolling is at **62,900**; the next persistent save
  and validation point is 63,000 (`eval_interval` 1000).
- **62,900 fork base secured**: the rolling save was hardlinked to
  `runs/v4b_learned/main/ckpt_eval/iter_0062900` (`cp -al`, 1,024 distcp files, 1.1 T, link count 2), so
  the rolling overwrite cannot take it away. The HF export of this directory is in progress.
- Resolved config: `train_iters` 236733, `exit_interval` 236733, `eval_interval` 1000, `eval_iters` 10,
  `arch_metrics_interval` 500, `save_interval` 1000, rolling every 50, `global_batch_size` 1024,
  `micro_batch_size` 1.
- Engram: `engram_branch_gates: false`, `engram_branch_query: read`, `engram_gate_signed_sqrt: true`,
  `engram_clip_rownorm: 7.0`. The fix keys (`engram_gate_logit_scale`, `engram_gate_query_eps`) are
  **not** in the main run's Megatron yet.

## 3. Design

- **Arm:** `bg_learned` (per-stream gates, learned query mix) + `engram_gate_logit_scale: true` +
  `engram_gate_query_eps: 1e-12`.
- **Why learned:** the query is `read + lam*(streams-read)` with `lam=0` at init, so step 0 is exactly
  the current single gate (function-preserving, no spike) and the model learns how much stream to mix
  in. This is the best-effort compromise: it can reach the original stream query without the abrupt
  `bg_stream` spike. `bg_read` and `bg_stream` are the controls.
- **Norm in all cases:** the query/key `hidden_norm`/`key_norm` and the value `conv_norm` are applied
  in every query/gate mode; `hidden_norm` uses the fix's `engram_gate_query_eps` (1e-12), so the query
  norm is scale-invariant, and `logit_scale` sets the size. The `learned` mix is learned before the norm.
- **All gates:** `engram_branch_gates` adds a gate for **every** mHC stream (all 4) at every Engram
  layer; `learned`, `read` and `stream` differ only in the query source. The `fix` is the single gate.
- **Init and spike:** layer 2 `logit_scale = 0.3` (spread ~0.43); layer 52 keeps the mean (~0.47, it
  is already open). This is a **gate-opening** value, not the function-preserving mean query RMS,
  because `lam=0` + a dead gate gives zero gradient to `lam` (double zero -> non-learnable). With the
  gate opened, `lam` starts at 0 (the read) and learns the stream; expect a short spike.
- **Length:** 1,100 steps, 62,900 → 64,000; `exit_interval: 64000`; early validation at 63,000 (100
  steps) and 64,000 (1,100 steps).
- **Baseline:** the main run's own 62,900 → 63,000 and 62,900 → 64,000 (paired, same data order).
- **Reset:** if the loss regresses, keep the main run's 62,900 rolling save and continue without the fix.

## 4. Prerequisites (done 2026-11-26)

1. **62,900 persistent:** hardlinked to `ckpt_eval/iter_0062900`.
2. **Megatron fix in the main repo:** `bd6b57856` cherry-picked onto the main run's Megatron line as
   `1054d2b75` (oellm/moonep-short-routing, pushed; 16/16 `test_engram_branch_gates.py`). Between the main
   run's Megatron (`a7d461f43`, snapshot `autoexp_v4bbf16c`) and the fork's there are only the LDM FP16 commits
   (#8, unused by v4b) and the single-set validation fix `feba2e6d8`. Mirrors regenerated (superproject 810f50d0).
3. **Surgery tool with `--logit-scale` / `--query-mix`** ported from the worktree (78c1fbd1).
4. **Prep job and fork config** (ca3c05d8); snapshot `autoexp_engramfix7`.

## 5. Step-by-step

### Step 1: fork base at 62,900 (done)
- The 62,900 rolling save is hardlinked to `ckpt_eval/iter_0062900`; the main tree is not edited and the main
  run keeps going on its 264 nodes.

### Steps 2-3: export, diag, query RMS, surgery (one job)
`config/experiments/korbi/production/engram_branch_fix_fork_prep_mn5.yaml` (1 node, c136 / cns143) runs
`scripts/korbi/production/engram_branch_fix_fork_prep.sh`:
1. `export_run.sh <main> 62900 hf_exports/ambitious_v4b_learned_fp8_62900` (fp8 experts, from `ckpt_eval`).
2. `engram_gate_diag.py --docs 16 --flores 0 --gsm8k 0 --fixed-query --logit-scales 0.025,0.1,0.3,1.0
   --sweep-layer 2` -> `engram_gate_diag/ambitious_v4b_learned_fp8_62900_fixed_query`.
3. Mean `|h~|` (`hn_L<i>`, all tokens of the five validation sets) per Engram layer -> `query_rms.json`;
   the init is layer 2 = 0.3, the other layer = its mean.
4. `add_engram_branch_gates.py --query-mix --logit-scale <init>` on `ckpt_eval/iter_0062900` ->
   `production/engram_branch_fix/fork_bg_learned_62900/ckpt/iter_0062900` + tracker (`logit_scale_init.txt`
   records the values).
5. The GSM8K / FLORES part of the diag (offline record; the fork does not wait for it).

### Step 4: fork (enqueued, waits on the tracker)
`config/experiments/korbi/recall_surgery/v4b_learned_fork_bg_learned_mn5.yaml`: base = the main run's own
config (`v4b_learned_fix_bf16rs_mn5`). A local render diffed against the main run's command
(`config-mn5-pool-p264.2520.yaml`) differs only in: `--engram-branch-gates`, `--engram-branch-query learned`,
`--engram-gate-logit-scale`, `--engram-gate-query-eps 1e-12`, `--load <fork>/ckpt`, `--ckpt-step 62900`,
`--exit-interval 64000`, `--arch-metrics-interval 10`, and the fork's own save / rolling / tb / wandb dirs and
name. 128 nodes (gbs 1024 = 2 micro-batches per GPU; reference sampler dp 512), c136 host004, cns143 /
acc_xlarge, 12 h, pool priority 100. Persistent saves + validation at 63,000 and 64,000; rolling saves every 50
into the fork's `ckpt_rolling` (a restart resumes there).

### Step 5: evaluation
- Loss: the fork's loss at 63,000 (100 steps) and 64,000 (1,100 steps) vs the main run's **same steps**
  (62,900 -> 63,000 and 62,900 -> 64,000), paired by data order. That is the clean comparison.
- Spike: the loss at the first ~50 steps and the recovery by ~500 steps; reset if it has not recovered.
- Arch metrics: per-stream gate spread, `query_rms`, `read_sum`, and the `query_mix` (`lam`) every 10
  steps; if `lam` stays at zero, the stream adds nothing.
- Validation at 63,000 and 64,000 vs the main run's.
- Downstream export/eval at 64,000 if the loss is not decisive.

### Step 6: decision
- **Win:** the fork's loss beats the main run's same steps beyond noise and the gates open -> switch
  the main run's tree to the fork tree at 64,000.
- **Neutral:** loss within noise, gates open -> the gate is not the bottleneck; keep the main tree.
- **Loss:** loss regresses -> keep the main tree; the fork is discarded.

## 6. Paths

- fork checkpoint: `/gpfs/scratch/ehpc533/hybrid_nucleus/production/engram_branch_fix/fork_bg_learned_62900`
- fork run: `/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/fork_bg_learned_62900`
- export: `/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4b_learned_fp8_62900`
- diag: `/gpfs/scratch/ehpc533/hybrid_nucleus/engram_gate_diag/ambitious_v4b_learned_fp8_62900_fixed_query`

The fork writes only its own dirs; the main tree `runs/v4b_learned/main` is untouched.

## 6b. Outcome (2026-11-27)

- The fork ran 62,900 -> 63,376 on 128 nodes of c136 host004, and from there on host011's 256 nodes in place of
  the main run (main stopped at 64,083 with a clean save). Speed on 256 nodes: about 16.0-16.2 s/step (main 15.7).
  Memory use: 0.90-0.96 of the cap (main 0.83).
- Paired eval-only validation (same 8-node layout and samples, eval_iters 3), fork - v4b:
  - 63k: +0.0001 to +0.0006 per set (`all` +0.0004). The in-training long_64k +0.015 was the 128- vs 264-node
    layout.
  - 64k: `all` +0.00003, per set -0.0003 to +0.0005: a tie.
- Layer-2 Engram off (engram_output_scale 0 on layer 2), the cost per model:
  - v4b: +0.375 (63k) / +0.392 (64k).
  - fork: +0.356 / +0.365.
  - Layer 2 carries a lot in both; the fork leans on it slightly less, not more.
- Gate scales (read from the saves):
  - layer 2, streams 0/1 hold about 0.27-0.28; streams 2/3 decay towards v4b's effective 0.015 (63,600: 0.05 /
    0.14);
  - layer 52 sharpens by itself (0.47 -> 0.58-0.77);
  - lam stays about +-0.01.
- Decision (user, 2026-11-27): continue the fork as the main run despite the tie (no cost, and hope for a later
  gain). The v4b main tree stays as the fallback at 64,083.
- Inference: per-stream gates in the HF export (0e0716ae; gate math bit-identical to Megatron) and in vLLM
  (9c9ecd37d0; passes the MoE-free parity components). A separate vLLM attention discrepancy predates this work
  and affects all v4b exports (single attention component up to 0.05 in fp32, TF32 or not).
- Incidents:
  - 00:16 GPU MMU fault in an eager bf16 `abs` kernel (as05r3b20; likely the clamped-SwiGLU backward); restart
    from 63,600.
  - 01:09 a healthy restart killed by the pool after my preempt of its node check (template fix 6dd3c23a).

## 7. Risks and rollback

- The main run continues on its 264 nodes; the fork runs in parallel on 128 nodes. The main tree is
  untouched; the fork has its own tree. The switch to the fork tree happens only on a win.
- `learned` at `lam=0` with a **dead** gate has zero gradient to `lam` (double zero -> non-learnable).
  Layer 2's `logit_scale = 0.3` opens the gate at init; then `lam` learns and expect a short spike.
- The `logit_scale` init depends on the 62,900 mean query RMS. If the export/diag is wrong, the step-0
  loss jumps and the trial is invalid.
- The 1,100 steps are ~0.5% of the remaining compute; the reset keeps the loss.
- The host004 arms remain the controlled tiebreaker if the trial is ambiguous.

## 8. Open questions

- Confirm the fork trigger at 62,900 (done; the main run continues).
- 1,100 steps (62,900 → 64,000) with the early eval at 63,000.
- Set the reset threshold (e.g. loss delta > +0.0005 beyond noise) and the spike tolerance.
- Decide the host004 arms' role (tiebreaker or drop).
- Land the Megatron commit `bd6b57856` in the main repo before the snapshot.

## 9. Observed gate values (2026-11-27)

Read from the exported `gate_terms.logit_scale` tensors (`hf_exports/ambitious_v4b_fork_bg_fp8_*`,
4 streams per Engram layer; init was one value per layer, broadcast to all streams):

| layer | init | @64,000 | @66,000 |
|---|---|---|---|
| 2 | 0.30 | 0.253, 0.354, 0.003, 0.026 | 0.194, **0.606**, 0.0007, 0.010 |
| 52 | 0.47 | 0.634, 0.749, 0.598, 0.804 | 0.615, 0.770, 0.649, **0.828** |

- Layer 52 opened all four streams (+30–75 % over init) and held them: a learned increase in the
  Engram read magnitude, not a schedule.
- Layer 2 diverged: two streams fell to ~0.003–0.026, one opened to 0.61. Note what ~0 means here:
  `logit_scale` scales the **pre-sqrt logit**, so `logit_scale = 0` gives `gate = sigmoid(0) = 0.5`
  — the *neutral* point, not a closed gate. Those two streams stopped modulating and sit at half
  strength; only stream 2 is actively query-dependent.
- `query_mix` (λ, init 0) moved only to ±0.01–0.04: the read/stream blend is nudged, not rewritten.

Consequence for the readout: the fork is functionally distinct from the main run, yet the recall
suite is a tie (fork@64,000 mean 57.74 vs main@64,000 58.02, −0.28 pp inside the 1.3–1.7 pp floor,
no row collapse). A large change in Engram read magnitude produced no measurable collateral damage
on QA — consistent with the Engram contributing 0.47 % of active FLOPs.

Still open: whether the revived layer-2 gates make that Engram carry information. That is the
paired validation in `config/experiments/korbi/recall_surgery/v4b_learned_fork_bg_learned_eval_mn5.yaml`
(main / fork x layer-2 Engram on/off, seven fixed validation sets, same samples per arm).
