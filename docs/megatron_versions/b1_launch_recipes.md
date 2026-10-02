# Complete B1 continuation recipes on JUPITER

Use these recipes instead of the inherited `revival_1` configurations. Those select different head learning rates and z-loss coefficients. The source-only September 25 release remains immutable; this release adds explicit B1 configurations and a standalone preparer with restore checks.

| Variant | Configuration | Exact Megatron revision | Deferred head gradients |
|---|---|---|---|
| Historical B1 reborn | `b1_reborn_legacy.yaml` | `3fc1452853978f7d6ff2add16d887bb6ed30470a` | Original, unfixed implementation; reproduce the historical run only |
| Corrected B1 reborn | `b1_reborn_patched.yaml` | `799ff4e6e1c3f6762283c97cd76ab4eb68073720` | Validated deferred-gradient patch, without test-only gradient instrumentation |

Both set deferral **on** and gradient reduction overlap **on**. Selecting the variant changes the source pin, not the B1 numerical recipe. Historical chronology: the original B1 used unfixed deferral through156k; the immediate156k continuation used deferral off; patched deferral was promoted at192k. Names containing `defer_fix` alone do not identify which implementation was used.

Shared settings are global LR `3e-4`, no decoupled head/embedding LR, Adam betas `.9/.95`, epsilon `1e-8`, weight decay `.1`, embedding and residual/pre-norm decay multipliers0, QK and final-norm decay multipliers1, z-loss `1e-5`, native FP32 fused cross-entropy, delayed hybrid FP8 with1024-entry maximum amax history, TP4/PP4, GBS4096, microbatch2, and sequence4096. Parameters and the post-load contract are recorded in `versions/b1_recipes.json`. Default topology is512 GH200 nodes,2048 GPUs. The original sample-based WSD schedule is preserved, including its eventual linear decay; these recipes do not select the separately proposed cooldown experiments.

## Prepare a collaborator checkout

```bash
git clone --branch oellm-32b-b1-recipes-20260928 https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
git submodule update --init submodules/Megatron-LM
```

Activate your host's dedicated autoexp environment with the dependencies from this repository's installation instructions. The login-side Python runs orchestration only. Training runs with `/opt/venv/bin/python` inside the validated JUPITER training SIF, SHA-256 `4d19cd6920652cdddba9bc39eb90053eb980c2885a7ce3f1c74398dd8e7b0f31`. Supply your accessible copy of that image, the training data blend manifest and its referenced tokenized data, the matching prebuilt data cache, tokenizer, account and hardware exclusion file. We do not distribute datasets, checkpoints, a Python environment or the SIF in Git. Paths must be absolute, shell-safe, and accessible inside the container's `/e` bind.

Supply an explicit, complete **trusted** native Megatron checkpoint directory, including Adam, RNG and data/scheduler state. The default shard count is2048; use `--checkpoint-shards 4096` for an appropriate4096-shard source such as conservative v1. Existence/count checks are followed by the native loader and fail-closed post-load audit; they are not a substitute for reading the actual tensors. Never point this at an in-progress checkpoint.

```bash
# Set these to your own accessible resources; every output belongs to RUN_ROOT.
python tools/b1_run.py prepare \
  --variant patched \
  --run-root "$RUN_ROOT" \
  --checkpoint "$CHECKPOINT_PARENT/iter_0060000" \
  --data-manifest "$DATA_MANIFEST" --data-cache "$DATA_CACHE" \
  --tokenizer "$TOKENIZER" --image "$TRAINING_SIF" \
  --exclude-file "$EXCLUDE_FILE" --account "$SLURM_ACCOUNT" \
  --end-step 100000
```

This prints a plan and queries live Slurm walltime policy without creating a run or submitting a job. Repeat the same command with `--execute` to prepare it. Preparation verifies the image checksum, copies the selected source into a new isolated checkout, links the seed read-only by convention into an owned checkpoint tree, and writes the configuration, restore contract and SHA-256 manifest. It never checks out another revision in the shared submodule. Use `--variant legacy` and a different run root to prepare historical B1.

Default continuation retains Adam moments and age and immediately uses the normal B1 schedule. For the60k continuation experiments' low-start10k ramp, explicitly add `--ramp-updates 10000 --ramp-floor 0.1`; this starts at `3e-5` and reaches `3e-4`. Ramp position follows consumed samples across rolling resumes, so a restart never repeats the ramp. Use `--wandb-project oellm_32b_dense_loss-increase_debug` for experiments; the default project is `oellm_32B_dense`. Optional `--name` gives an unambiguous run name. A bounded experiment and a full production continuation are different endpoints; select `--end-step` intentionally.

## Inspect and launch

```bash
python tools/b1_run.py verify --run-root "$RUN_ROOT"
python tools/b1_run.py launch --run-root "$RUN_ROOT"       # renders only
# Inspect the emitted sbatch/config and resource estimate, then:
python tools/b1_run.py launch --run-root "$RUN_ROOT" --execute
```

Run the last command in your named tmux session and keep the native autoexp monitor attached. `launch --execute` records a persistent launch intent before starting autoexp and refuses a second initial launch from that root. Autoexp retains its own compute-cost confirmation prompts. If the monitor process is interrupted, use the exact `scripts/monitor_autoexp.py --session ... --monitor-state-dir ...` command printed by autoexp to reattach to the existing session; do not prepare or submit a duplicate run. Inspect the intent/state if failure occurs before submission. A root `STOP` file blocks subsequent launcher entry; it does not itself cancel a live Slurm job or stop an already attached monitor.

These launch configurations use the existing native autoexp failure/exclusion policy, checkpoint-on-signal relay, permanent saves every2000 updates and arm-owned rolling saves every125 updates. Each initial launch rechecks the maximum partition/account/QoS walltime and refuses changed policy instead of silently retaining a stale limit. No shorter application duration exit is inherited. Retention of native rolling saves remains Megatron's responsibility. Workspace-specific mirroring, downstream-evaluation campaigns, report generators, monitor supervision and offline W&B upload services are **separate services**; these recipes do not install another user's watchdogs. W&B and TensorBoard data are written below the owned run root; configure your W&B sync service for that directory. The default downstream checkpoint hook is empty so it cannot invoke private workspace paths.

At startup, the wrapper checks loaded arguments, constructed model/DDP settings, scheduler and data position, and Adam age. It reapplies fresh B1 parameter-group hyperparameters after checkpoint loading while verifying sampled Adam moments remain unchanged. It writes per-rank evidence in `restore-audit/<SLURM_JOB_ID>/` and emits `B1_POST_LOAD_CONTRACT_PASSED`. Any mismatch aborts before training. The legacy variant intentionally retains its historical gradient defect; use the patched variant for new training.

## Reproduce the packaging checks

```bash
PYTHONPATH=. python -m pytest -q --no-cov \
  tests/integration/test_b1_recipes.py \
  tests/integration/test_b1_restore_contract.py \
  tests/integration/test_ce_fp32_config.py
```

The tests compose both full recipes through the actual typed loader and render their Megatron commands, check optimizer-setting restoration and checkpoint override rejection, and verify ramp behavior on resume and during eventual cooldown. Source-level GPU validation of the corrected gradient patch is documented in [the source milestone](b1_reborn_20260925.md). Packaging tests and dry-run rendering do not constitute a new GPU training run of the packaged wrapper with the historical source.

## Optimized TWEO variant

The additive `oellm-32b-b1-tweo-20261002` release also provides `--variant tweo`, preserving these legacy/patched variants. Use the [TWEO guide](b1_tweo_20261002.md) for its exact source, calibrated80k profile, coefficient-ramp semantics and numerical checks.
