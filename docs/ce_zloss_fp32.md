# Native CE/z-loss FP32 combination

Branch `prod/oellm_32b_dense_revival_1` pins a Megatron submodule containing the tested native CE backward correction. Initialize/update the pinned submodule; merely switching the outer repository branch does not update an already checked-out submodule.

```bash
git clone --branch prod/oellm_32b_dense_revival_1 https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
git submodule update --init submodules/Megatron-LM
git submodule status submodules/Megatron-LM
sha256sum submodules/Megatron-LM/megatron/core/fusions/fused_cross_entropy.py
```

Expected source SHA-256: `600f811d11c51c1a4bcc74a541de68e9a5b6aed9d5a08acd9ec88c04eadb9c8c`. Existing checkouts should first preserve their uncommitted work, fast-forward the outer branch, then update the submodule. The previous original implementation is pinned by autoexp commit `ced56c6`; the new submodule changes native CE for all configurations that use this checkout. A backend flag alone does not toggle between the original and corrected native implementation.

The correction preserves FP32 through the CE/z-gradient sum and lets autograd cast at the BF16 input boundary. The CE backend stays `native`; Transformer Engine CE is not enabled. The z coefficient stays `1e-4`. Forward CE/z calculations were already FP32; this fixes premature backward rounding rather than changing the reported loss definition.

## Reproduce a patched experiment

Use an existing activated autoexp environment and the JUPITER training container/data paths inherited by the revival configuration. Do not install dependencies into a system environment. The overlay `experiments/oellm_32b_dense/oellm_32b_dense_revival_1_ce_fp32` retains the TP4/PP4/DP128 FP8 recipe, batch sizes and body/head learning rates. It requires explicit private outputs, a full checkpoint root with its tracker, remaining-update count and absolute end step:

```bash
export PROJECT_DIR="$PWD"
export CE_RUN_ROOT=/path/to/your/private/ce-fp32-run
export CE_RUN_NAME=your-ce-fp32-run
export CE_LOAD_PATH=/path/to/full/checkpoint-root
export CE_REMAINING_STEPS=4000
export CE_END_STEP=110000
export CE_WALLTIME=06:00:00
export SLURM_ACCOUNT=your_authorized_account
export JUPITER_EXCLUDE_NODES=/e/project1/e-sta-openeurollm/production_training/jupiter_exclude_nodes.txt

PYTHONPATH=. python scripts/run_autoexp.py \
  --config-name experiments/oellm_32b_dense/oellm_32b_dense_revival_1_ce_fp32 \
  --dry-run
```

The example assumes a full 106000 checkpoint and requests 4000 updates; set both end/count consistently for your actual checkpoint. Review the rendered script, account, allocation, checkpoint paths, source/submodule revision and output locations. Submit an authorized experiment by removing `--dry-run` and adding `--monitor-state-dir "$CE_RUN_ROOT/training_core/_monitor_state"` in your named tmux workflow. The monitor retains the production log-event restart policy and refreshes node exclusions when resubmitting. The JUPITER launcher explicitly sets `SLURM_MPI_TYPE=none`, `--mpi=none` and `--cpus-per-task=288`, as validated after the maintenance change.

The example writes offline W&B records to `oellm_32b_dense_loss-increase_debug`, preserves optimizer/RNG loading, and saves every 250 steps plus rolling saves every 125. It disables inherited external checkpoint-evaluation and registry-write hooks so a collaborator's test does not invoke the shared production campaign. It does not silently seed the private output tree or overwrite the supplied input checkpoint root. Existing autoexp restart logic selects saved progress for subsequent attempts; retain the monitor state and all output paths.

The current matched campaign additionally uses bounded terminal-state recovery, full-state verification, budget guards and selection checks implemented in `SLAMPAI/oellm-workflows`, `helper_scripts/run_ce_training.sh` and `ce_precision/training/restart_policy.py`. Those campaign controls remain in that repository; this overlay is a portable experiment recipe using autoexp's existing production log policy. Main production lifecycle still goes through `oellm-workflows/oellm_32B_loss-increase_debug/production/revival_ctl.sh`; this document does not authorize an additional production instance.

## Validation and limits

The pinned CE file is byte-identical to the tested candidate: TP4 GPU numerical probes; matched 64k/104k real-token backwards; full-topology optimizer/RNG/FP8 load/update/save checks; and patched job 1728364, which completed the matched continuation through 106000 on September 9, 2026. The first 2000 updates have lower mean training CE and a steeper downward fit than original; they do not establish that this correction explains the historical degradation. See the recorded reports under `SLAMPAI/oellm-workflows/oellm_32B_loss-increase_debug/ce_precision/` for test details and retained limitations.

The overlay has CPU configuration/command tests covering native CE, full-state flags, output/project isolation and launcher settings. Publication does not introduce another kernel variation. Existing live experiment checkouts remain pinned to their original revisions plus the identical tested patch, so committing this branch does not alter an in-flight job.
