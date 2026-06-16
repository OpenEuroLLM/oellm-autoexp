#!/usr/bin/env bash
# Submit MoE-metrics jobs for the released HF MoE models.
#
#   - Qwen/Qwen3-30B-A3B         (1 job, main)
#   - stepfun-ai/Step-3.5-Flash  (1 job, main)
#   - allenai/OLMoE-1B-7B-0924   (N jobs — every 10th step branch + main)
#
# Each job is a separate run_autoexp.py invocation with sweep=none + a
# metadata.revision override; this avoids the monitor_state race that bit us
# in the Megatron sweep (two parallel pythons sharing an epoch-second
# session-id dir would each submit the union of all configured buckets).
#
# Usage:
#   bash scripts/moe/submit_hf_metrics.sh                # all 3 models
#   MODELS=olmoe bash scripts/moe/submit_hf_metrics.sh   # just OLMoE
#   OLMOE_EVERY=20 MODELS=olmoe bash scripts/moe/submit_hf_metrics.sh
set -euo pipefail

MODELS="${MODELS:-qwen3 step35 olmoe}"
OLMOE_EVERY="${OLMOE_EVERY:-10}"

# Compute nodes are offline. Every revision below must already live in
# $HF_HOME — run `OLMOE_EVERY=${OLMOE_EVERY} bash scripts/moe/download_hf_models.sh`
# on a login node first (same OLMOE_EVERY value keeps cache and submit in sync).

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-OELLM_prod2026}"
export SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"
export OELLM_CONTAINERS_DIR="${OELLM_CONTAINERS_DIR:-/leonardo_work/OELLM_prod2026/container_images}"
export HF_HOME="${HF_HOME:-/leonardo_scratch/large/userexternal/ajha0001/HF_CACHE}"

submit_one () {
  local config="$1"
  local revision="$2"
  echo "================================================================"
  echo "Submitting: ${config}   revision=${revision}"
  echo "================================================================"
  # --submit-and-exit: fire the sbatch job and return immediately instead of
  # entering the blocking monitor loop, so all revisions queue back-to-back.
  python scripts/run_autoexp.py \
    --config-name="experiments/abhash/${config}" \
    --submit-and-exit \
    metadata.revision="${revision}" \
    sweep=none
}

for m in $MODELS; do
  case "$m" in
    qwen3)
      submit_one hf_moe_metrics_qwen3_30b main
      ;;
    step35)
      submit_one hf_moe_metrics_step35_flash main
      ;;
    olmoe)
      # OLMOE_ALL=1 → every cached revision (read offline from the HF cache
      # refs/ dir). Otherwise every-${OLMOE_EVERY}th via the Hub API.
      if [[ "${OLMOE_ALL:-0}" == "1" ]]; then
        echo "Listing ALL cached OLMoE revisions (from local HF cache)..."
        revs=$(python scripts/moe/list_olmoe_revisions.py --all --from-cache)
      else
        echo "Listing OLMoE revisions (every ${OLMOE_EVERY}th)..."
        revs=$(python scripts/moe/list_olmoe_revisions.py --every "${OLMOE_EVERY}")
      fi
      n=$(echo "$revs" | wc -l)
      echo "Will submit $n OLMoE jobs."
      for r in $revs; do
        submit_one hf_moe_metrics_olmoe "$r"
      done
      ;;
    *)
      echo "Unknown model key: $m" >&2
      exit 1
      ;;
  esac
done
