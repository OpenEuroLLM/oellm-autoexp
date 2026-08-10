# Muon + real Qwen3 0.1B_ne smoke test — Capella

Companion to `RESULT.md` (generic 50M proxy model). This one uses the real
architecture from `multilingual_scaling` (`qwen3_dense_0.1B_ne.yaml`, 16L/512h,
97M params) instead of the proxy, still against our own converted Nemotron
sample data (`DATA_CONVERSION.md`). Config: `muon_qwen3_0.1B_longctx.yaml`.
Standalone — no dependency on the real scaling-laws sweep (see that config's
comments: `/sweep: none`, own `base_output_dir`, own `wandb_project`, only
imports the model architecture fields).

## Verdict: PASS (2026-08-08, job 3846248). Full pipeline confirmed end-to-end on Capella — real Nemotron data + real Qwen3 0.1B_ne architecture + Muon + working checkpoint save, on a full H100.

All 50 iterations completed, `lm loss` 10.906 → 6.715 monotonically, zero
NaN/skipped iterations. All 3 checkpoints (`save_interval: 20`, iters 20/40/50)
saved successfully and verified present on disk (`iter_0000050/` has a real
~1GB `.distcp` weights file + `common.pt` + `.metadata`). Job exited
`COMPLETED`, `0:0`, 14m32s.

One new minor/non-fatal issue surfaced by switching to the async save path:
`wandb_finalize_fn` → `wandb_writer.run.log_artifact(...)` raises
`AttributeError: 'NoneType' object has no attribute 'log_artifact'` during
final async-save-queue teardown (`maybe_finalize_async_save(terminate=True)`
at process exit) — `wandb_writer.run` is already `None` by the time this
queued finalize callback fires. Happens strictly *after* the checkpoint write
itself already succeeded, and didn't affect the job's exit code. Only
reachable via the async save path (`async_save: true`), so never hit by the
sync-save crashes documented below. Not investigated further since it's
cosmetic — flagging in case it matters for wandb artifact tracking later.

Jobs 3833304, 3833446, 3838181 (all `capella`, full H100, not MIG) — every one
of them trained 20 clean iterations identically: `lm loss` 10.906 → 7.552,
grad norm 12.7 → 1.3, zero NaN/skipped iterations, ~50-75k tok/s/GPU. Then
crashed at the very first checkpoint save (`save_interval: 20`).

## Bugs found and fixed

1. **`norm_epsilon` (shared model file, not ours to edit)** —
   `qwen3_dense_0.1B_ne.yaml` (also used by `slaing00/multilingual_scaling`)
   still uses the pre-v0.17 field name. Our own `layernorm_epsilon` override
   doesn't remove the stale key from the merged config, and compoconf rejects
   unknown keys regardless of value. Fixed via a Hydra delete override at
   invocation time, not a file edit:
   ```
   '~backend.megatron.norm_epsilon'
   ```
   (see `04_qwen3_real_data_test_capella.sbatch`).

2. **`lr_wsd_decay_iters` unset → `AssertionError: wsd_decay_steps is not None`** —
   `qwen3_dense_0.1B_ne.yaml` sets `lr_decay_style: WSD` but leaves the decay
   length to be computed per-stage by the staged-sweep DAG (see
   `muon_50M_50BT.yaml`'s stable/decay sweep block). Our config is a flat
   single "smoke" stage, not routed through that DAG, so it's never set.
   Fixed with an explicit override in our own yaml: `lr_wsd_decay_iters: 0`
   (same value the 50M config's stable stage uses — "no decay in stable
   phase").

3. **Checkpoint save crash — took 3 attempts to root-cause properly:**
   - Symptom: `ImportError: cannot import name 'get_write_results_queue'
     from nvidia_resiliency_ext...` at the first `save_interval` hit, every
     time.
   - **Attempt 1 (wrong)**: took the error message's own suggestion at face
     value (`async_strategy: mcore`). Didn't work — traced why afterward:
     `torch.py`'s sync-save wrapper (`TorchDistSaveShardedStrategy.save()`,
     used whenever `async_save: False`) hardcodes
     `"nvrx" if HAVE_NVRX else "mcore"` and ignores the config's
     `async_strategy` value entirely for that code path.
   - **Attempt 2 (still wrong)**: added `async_save: true` alongside
     `async_strategy: mcore`, reasoning the async code path *does* respect
     `async_strategy` (verified via `serialization.py`/`checkpointing.py`
     call chain). Compose was clean, but it crashed identically again.
     Resolved config config showed `async_save: true`, but Megatron's own
     printed args dump showed `async_save ... False` at runtime — turned out
     Megatron's `arguments.py:1523-1530` silently disables `async_save`
     unless `use_persistent_ckpt_worker` is also `True`, with a `UserWarning`
     (`--async-save is not supported without --use-persistent-ckpt-worker.
     Disabling --async-save.`) that was sitting in the log the whole time —
     found only by actually reading the full log instead of grepping just
     for Traceback/Error.
   - **Fix (applied, not yet run)**: all three together:
     ```yaml
     async_strategy: mcore
     async_save: true
     use_persistent_ckpt_worker: true
     ```
     This routes checkpoint saving through the genuinely-async path with the
     `mcore` strategy, which doesn't touch the broken `nvidia_resiliency_ext`
     install in the sandbox at all.

## Why the sandbox package is broken (not fixed, just routed around)

The sandbox's installed `nvidia_resiliency_ext` imports enough to pass
Megatron's top-level `try/except` (`HAVE_NVRX = True`), but is missing
`get_write_results_queue`, needed deeper in the `nvrx` async strategy specifically.
Root cause is a partial/incompatible package version in the sandbox — not
touched here (user explicitly wanted the sandbox left alone); worked around
entirely via config instead.

## Next
- (Optional, cosmetic) look into the `wandb_finalize_fn` `AttributeError` if
  wandb checkpoint-artifact tracking turns out to matter later.
- Multi-GPU: deliberately not attempted yet (see conversation) — separate
  follow-up, since Muon's distributed optimizer path is unverified at
  `dp_cp>1` and `/slurm: local` is currently hardcoded single-GPU.
