#!/usr/bin/env bash
# Submit the full MoE-metrics sweep over the two load-balance ablation grids:
#   algo ∈ {global_batch_aux, deepseek_bias}
#   nexp ∈ {8, 64}
# → 4 SLURM jobs total, all submitted by a SINGLE run_autoexp.py invocation.
#
# (An earlier parallel-per-algo design hit a monitor_state collision when both
# pythons created the same session-id directory; folding both algos into one
# sweep removes that race entirely.)
#
# Usage:
#   bash scripts/moe/submit_all_metrics.sh
set -euo pipefail

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-OELLM_prod2026}"
export SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"
export OELLM_CONTAINERS_DIR="${OELLM_CONTAINERS_DIR:-/leonardo_work/OELLM_prod2026/container_images}"
export HF_HOME="${HF_HOME:-/tmp}"

python scripts/run_autoexp.py \
  --config-name="experiments/abhash/moe_metrics_120BT_loadbal_ablation"
