#!/bin/bash
# Long-context evals of a t16 arm with its sparse attention active (the step-3000 HF export, t16_hf_export_mn5.sh):
#   nll   per-position NLL on 8 held-out documents >= 64k (nll_by_position.py, chunked prefill through the cache)
#   mdqa  multi-document QA: a SQuAD question on a passage at depth far / mid / near among other passages, 8k / 32k /
#         64k context, greedy generation, F1 (multidoc_qa.py)
#   mdqa3 the same with 3 worked examples (span answers from a base model), the gold passage alone as the ceiling,
#         60 questions per cell
#   retr  synthetic retrieval, generative (RULER-style, offline): passkey, multi-key NIAH, variable tracking at
#         4k-64k, exact match (retrieval_gen.py)
# One MN5 node:
#
#   t16_long_evals_mn5.sh <arm> <nll|mdqa|mdqa3|retr|nll_long|mdqa3_128k|mdqa3_256k|retr_long>[_tok] [step]
set -euo pipefail
arm=$1; task=$2; step=${3:-3000}
R=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}
D=/gpfs/projects/ehpc533/users/korbip/data
DS=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/data/downstream
model=$R/hf_exports/t16_s3_${arm}_${step}
out=$R/campaign_midtraining/t16_evals/$task/$arm
# <task>_tok: the same task with per-token selection for the prompt's last 128 queries and in decode
# (OELLM_SPARSE_SELECT=token, oellm_ambitious/sparse_attention.py); the default is the training's block-wise selection
if [[ $task == *_tok ]]; then
  export OELLM_SPARSE_SELECT=token OELLM_SPARSE_TAIL=128
  task=${task%_tok}
fi
[ -f "$model/.check_done" ] || { echo "export $model not checked yet" >&2; exit 1; }
mkdir -p "$out"
echo "[t16_long_evals] $arm $task: $model -> $out"
case $task in
  nll)  python scripts/korbi/campaign_midtraining/nll_by_position.py --model "$model" \
          --docs "$D/v8ml-valid/long_64k_text_document" --out "$out/nll.json" --lengths 65536 --docs-n 8 ;;
  mdqa) python scripts/korbi/campaign_midtraining/multidoc_qa.py --model "$model" --squad "$DS/squad/dev-v2.0.json" \
          --out "$out" --lengths 8192 32768 65536 --n 30 ;;
  mdqa3) python scripts/korbi/campaign_midtraining/multidoc_qa.py --model "$model" --squad "$DS/squad/dev-v2.0.json" \
          --out "$out" --lengths 8192 32768 65536 --depths far=0.05 mid=0.5 near=0.95 only --shots 3 --n 60 ;;
  retr) python scripts/korbi/campaign_midtraining/retrieval_gen.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out" --lengths 4096 8192 16384 32768 65536 --n 30 ;;
  # beyond the training length (the s3 testbed trains at 64k): 128k / 256k, separate jobs per length
  nll_long) python scripts/korbi/campaign_midtraining/nll_by_position.py --model "$model" \
          --docs "$D/v8ml-valid/long_64k_text_document" --out "$out/nll.json" --lengths 131072 262144 --docs-n 8 ;;
  mdqa3_128k|mdqa3_256k) L=$([ $task = mdqa3_128k ] && echo 131072 || echo 262144)
        python scripts/korbi/campaign_midtraining/multidoc_qa.py --model "$model" --squad "$DS/squad/dev-v2.0.json" \
          --out "$out" --lengths $L --depths far=0.05 mid=0.5 near=0.95 --shots 3 --n 30 ;;
  retr_long) python scripts/korbi/campaign_midtraining/retrieval_gen.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out" --lengths 131072 262144 --n 21 --tasks passkey kv ;;
  *) echo "unknown task $task" >&2; exit 2 ;;
esac
echo "[t16_long_evals] $arm $task done"
