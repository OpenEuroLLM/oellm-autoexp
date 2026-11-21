#!/bin/bash
# HF side of the t16 parity check: the arm's HF export (t16_hf_export_mn5.sh, sparse attention in the model) scores
# the very samples Megatron's fixed validation scores (paired_valid_check) on a seeded subset of the long-document
# validation set at 64k, where the selection prunes keys (documents >= 64k). The subset is built once (flock); the
# Megatron side is t16_megatron_fixedvalid_mn5 on the same subset. One MN5 node:
#
#   t16_hf_parity_mn5.sh <arm> [step]
#
# EXPORT_SUITE / EXPORT_PREFIX as t16_hf_export_mn5.sh; the result goes to results/hf_<prefix>_<arm>_<step>.json for
# another suite (results/hf_<arm>_<step>.json for t16, unchanged).
set -euo pipefail
arm=$1; step=${2:-3000}
R=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}
D=/gpfs/projects/ehpc533/users/korbip/data
SUB=$R/campaign_midtraining/t16_parity/v8ml-valid-long64k-sub32
run=$R/production/${EXPORT_SUITE:-t16_s3_sparse_fixed}/$arm
out=$R/hf_exports/${EXPORT_PREFIX:-t16_s3}_${arm}_${step}
tag=$([ -n "${EXPORT_PREFIX:-}" ] && echo "${EXPORT_PREFIX}_" || true)
config=$(readlink -f "$(ls -t "$run"/current_*.yaml | head -1)")
[ -f "$out/.check_done" ] || { echo "export $out not checked yet" >&2; exit 1; }
mkdir -p "$SUB" "$R/campaign_midtraining/t16_parity/results"
(
  flock 9
  if [ ! -f "$SUB/long_64k_text_document.idx" ]; then
    # 32 samples: a multiple of the evaluation's data-parallel size (4 nodes x 4 GPUs)
    python -m oellm_autoexp.hf_export.make_valid_subset --samples 32 --seq-len 65536 --seed 0 \
      "$D/v8ml-valid/long_64k_text_document=$SUB/long_64k_text_document"
  fi
) 9>"$SUB/.lock"
python -m oellm_autoexp.hf_export.paired_valid_check --model-dir "$out" --config "$config" \
  --prefix "$SUB/long_64k_text_document" --seq-len 65536 \
  --out "$R/campaign_midtraining/t16_parity/results/hf_${tag}${arm}_${step}.json"
echo "[t16_hf_parity] $arm done"
