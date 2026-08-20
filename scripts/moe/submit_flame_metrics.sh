#!/usr/bin/env bash
# Submit MoE-metrics jobs for the CMU-FLAME FLAME-MoE family (Megatron torch_dist).
# One sbatch job per model via a separate run_autoexp.py invocation with
# sweep=none + --submit-and-exit (mirrors submit_hf_metrics.sh; avoids the
# monitor_state session-dir race that bit the parallel Megatron sweep).
#
# Checkpoints must already be on disk (compute nodes are offline) — run
#   bash scripts/moe/download_flame_moe.sh
# on a login node first. Same FLAME_ROOT must be used in both scripts.
#
# Usage:
#   bash scripts/moe/submit_flame_metrics.sh                       # all 7 models
#   MODELS="FLAME-MoE-38M-100M" bash scripts/moe/submit_flame_metrics.sh
set -euo pipefail

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-OELLM_prod2026}"
export SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"
export OELLM_CONTAINERS_DIR="${OELLM_CONTAINERS_DIR:-/leonardo_work/OELLM_prod2026/container_images}"
export HF_HOME="${HF_HOME:-/tmp}"   # unused in megatron_indexed mode; base.yaml expects it

FLAME_ROOT="${FLAME_ROOT:-/leonardo_scratch/large/userexternal/ajha0001/flame_moe}"

MODELS="${MODELS:-\
FLAME-MoE-38M-100M \
FLAME-MoE-98M-349M \
FLAME-MoE-115M-459M \
FLAME-MoE-290M-1.3B \
FLAME-MoE-419M-2.2B \
FLAME-MoE-721M-3.8B \
FLAME-MoE-1.7B-10.3B}"

submit_one () {
  local model="$1"
  local load_dir="${FLAME_ROOT}/${model}"
  echo "================================================================"
  echo "Submitting: ${model}"
  echo "================================================================"
  if [[ ! -f "${load_dir}/latest_checkpointed_iteration.txt" ]]; then
    echo "  SKIP: no checkpoint at ${load_dir} (run download_flame_moe.sh first)" >&2
    return
  fi
  python scripts/run_autoexp.py \
    --config-name="experiments/abhash/flame_moe_metrics" \
    --submit-and-exit \
    metadata.model="${model}" \
    sweep=none
}

for m in $MODELS; do
  submit_one "$m"
done
