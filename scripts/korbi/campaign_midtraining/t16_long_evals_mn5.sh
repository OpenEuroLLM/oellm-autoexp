#!/bin/bash
# Long-context evals of a t16 arm with its sparse attention active (the step-3000 HF export, t16_hf_export_mn5.sh):
#   nll   per-position NLL on 8 held-out documents >= 64k (nll_by_position.py, chunked prefill through the cache)
#   mdqa  multi-document QA: a SQuAD question on a passage at depth far / mid / near among other passages, 8k / 32k /
#         64k context, greedy generation, F1 (multidoc_qa.py)
# One MN5 node:
#
#   t16_long_evals_mn5.sh <arm> <nll|mdqa> [step]
set -euo pipefail
arm=$1; task=$2; step=${3:-3000}
R=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}
D=/gpfs/projects/ehpc533/users/korbip/data
DS=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/data/downstream
model=$R/hf_exports/t16_s3_${arm}_${step}
out=$R/campaign_midtraining/t16_evals/$task/$arm
[ -f "$model/.check_done" ] || { echo "export $model not checked yet" >&2; exit 1; }
mkdir -p "$out"
echo "[t16_long_evals] $arm $task: $model -> $out"
case $task in
  nll)  python scripts/korbi/campaign_midtraining/nll_by_position.py --model "$model" \
          --docs "$D/v8ml-valid/long_64k_text_document" --out "$out/nll.json" --lengths 65536 --docs-n 8 ;;
  mdqa) python scripts/korbi/campaign_midtraining/multidoc_qa.py --model "$model" --squad "$DS/squad/dev-v2.0.json" \
          --out "$out" --lengths 8192 32768 65536 --n 30 ;;
  *) echo "unknown task $task" >&2; exit 2 ;;
esac
echo "[t16_long_evals] $arm $task done"
