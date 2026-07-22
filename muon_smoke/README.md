# Muon 50M smoke test — where everything lands

Everything below is inside your worktree: `oellm-muon/`
(reachable as either `/scratch/project_465002530/users/laingsam/oellm-muon/`
or `/pfs/lustrep3/scratch/project_465002530/users/laingsam/oellm-muon/` — same files).

## Curated step logs (what I write — read these first)

`oellm-muon/muon_smoke/`

| File | Step | What to look for |
|------|------|------------------|
| `00_schema_regen.log`       | regenerate v0.17 schema | ends with the muon keys added, no errors |
| `01_optimizer_construct.log`| build Muon+Adam optimizer on GPU | `HAVE_EMERGING_OPTIMIZERS: True`, "optimizer built OK" |
| `02_dry_run_command.txt`    | rendered Megatron command | the full `python pretrain_gpt.py --optimizer muon ...` line |
| `03_smoke_step.log`         | ~10-20 real training iters | `iteration  N/... | lm loss: ...` lines decreasing |
| `RESULT.md`                 | my one-page summary | pass/fail per step + next step |

Each log is the raw stdout+stderr of that step, prefixed so they sort in order.

## Framework's own output (Megatron writes here itself)

`oellm-muon/outputs/muon_50M_50BT/<job.name>/`
- `checkpoints/`      — torch_dist checkpoints (only if smoke step saves)
- `tensorboard/`      — TensorBoard event files (loss/throughput curves)
- `logs/`             — Megatron's own stdout/stderr per run
- wandb offline run dir (WANDB_MODE=offline — nothing uploaded)

## Nothing is written outside this worktree + your cache dir
- caches: node-local `/tmp` (triton) and your `HF_HOME` (pinned per-command)
- data cache: `/scratch/project_465002530/users/laingsam/muon_test_cache/`
