#!/usr/bin/env bash
# Single-bucket dry-run submission of the MoE metrics job.
# Picks one bucket from the global_batch_aux grid (nexp=8) and submits it,
# bypassing the sweep so only ONE sbatch job is launched.
#
# Usage:
#   bash scripts/moe/submit_dummy_metrics.sh
#
# Override the bucket from the CLI, e.g.:
#   ALGO=deepseek_bias NEXP=64 bash scripts/moe/submit_dummy_metrics.sh
set -euo pipefail

ALGO="${ALGO:-global_batch_aux}"   # global_batch_aux | deepseek_bias
NEXP="${NEXP:-8}"                  # 8 | 64

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-OELLM_prod2026}"
export SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"
export OELLM_CONTAINERS_DIR="${OELLM_CONTAINERS_DIR:-/leonardo_work/OELLM_prod2026/container_images}"
export HF_HOME="${HF_HOME:-/tmp}"

echo "Submitting dummy bucket: algo=${ALGO}, nexp=${NEXP}"

python scripts/run_autoexp.py \
  --config-name="experiments/abhash/moe_metrics_120BT_${ALGO}" \
  metadata.nexp="${NEXP}" \
  sweep=none
