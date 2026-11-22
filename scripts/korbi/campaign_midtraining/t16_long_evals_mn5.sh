#!/bin/bash
# Long-context evals of a t16 arm with its sparse attention active (the step-3000 HF export, t16_hf_export_mn5.sh):
#   nll   per-position NLL on 8 held-out documents >= 64k (nll_by_position.py, chunked prefill through the cache)
#   mdqa  multi-document QA: a SQuAD question on a passage at depth far / mid / near among other passages, 8k / 32k /
#         64k context, greedy generation, F1 (multidoc_qa.py)
#   mdqa3 the same with 3 worked examples (span answers from a base model), the gold passage alone as the ceiling,
#         60 questions per cell
#   retr  synthetic retrieval, generative (RULER-style, offline): passkey, multi-key NIAH, variable tracking at
#         4k-64k, exact match (retrieval_gen.py)
#   beyond 256k (t18, 2026-11-19): nll_512k / nll_1m (the 10 / 3 held-out documents that long), retr_vt_long
#         (variable tracking at 128k / 256k), retr_512k / retr_1m (passkey, multi-key, variable tracking), e2e (prefill
#         and decode cost of the whole model at 128k-1M, long_e2e_bench.py)
#   realistic: real64 (LongBench v1's 14 tasks + InfiniteBench cut to 64k, 50 each), real_nqa64 (NarrativeQA's 100
#         longest), real_inf128k / real_inf256k (InfiniteBench en.mc / en.qa / code.debug cut to 128k / 256k)
# One MN5 node (or one GPU: the evals load the model on the visible devices):
#
#   t16_long_evals_mn5.sh <arm> <nll|mdqa|mdqa3|retr|nll_long|mdqa3_128k|mdqa3_256k|retr_long|...>[_tok][_k<N>] [step]
#
# EVAL_EXPORT_PREFIX (default t16_s3) / EVAL_OUT (default t16_evals): another suite's exports and result tree.
set -euo pipefail
arm=$1; task=$2; step=${3:-3000}
R=${HYBRID_OUTPUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus}
D=/gpfs/projects/ehpc533/users/korbip/data
DS=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/data/downstream
model=$R/hf_exports/${EVAL_EXPORT_PREFIX:-t16_s3}_${arm}_${step}
out=$R/campaign_midtraining/${EVAL_OUT:-t16_evals}/$task/$arm
# <task>_tok: the same task with per-token selection for the prompt's last 128 queries and in decode
# (OELLM_SPARSE_SELECT=token, oellm_ambitious/sparse_attention.py); the default is the training's block-wise selection
# <task>_k<N>: N selected pages at inference (OELLM_SPARSE_TOPK) instead of the trained 32
if [[ $task =~ _k([0-9]+)$ ]]; then
  export OELLM_SPARSE_TOPK=${BASH_REMATCH[1]}
  task=${task%_k*}
fi
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
  retr_long60) python scripts/korbi/campaign_midtraining/retrieval_gen.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out" --lengths 131072 262144 --n 60 --tasks passkey kv ;;
  nll_512k|nll_1m) L=$([ $task = nll_512k ] && echo 524288 || echo 1048576)
        N=$([ $task = nll_512k ] && echo 8 || echo 3)
        python scripts/korbi/campaign_midtraining/nll_by_position.py --model "$model" \
          --docs "$D/v8ml-valid/long_64k_text_document" --out "$out/nll.json" --lengths $L --docs-n $N ;;
  retr_vt_long) python scripts/korbi/campaign_midtraining/retrieval_gen.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out" --lengths 131072 262144 --n 30 --tasks vt ;;
  retr_512k|retr_1m) L=$([ $task = retr_512k ] && echo 524288 || echo 1048576)
        python scripts/korbi/campaign_midtraining/retrieval_gen.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out" --lengths $L --n 30 --tasks passkey kv vt ;;
  e2e) python scripts/korbi/campaign_midtraining/long_e2e_bench.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out/e2e.json" --lengths 131072 262144 524288 1048576 ;;
  # realistic tasks (long_realistic_eval.py): LongBench v1 (<= 64k) and InfiniteBench (cut to 64k / 128k / 256k)
  real64) python scripts/korbi/campaign_midtraining/long_realistic_eval.py --model "$model" \
          --data "$D/long_evals" --out "$out" --max-len 65536 --n 50 ;;
  real_nqa64) python scripts/korbi/campaign_midtraining/long_realistic_eval.py --model "$model" \
          --data "$D/long_evals" --out "$out" --tasks narrativeqa --max-len 65536 --n 100 --select longest ;;
  real_inf128k|real_inf256k) L=$([ $task = real_inf128k ] && echo 131072 || echo 262144)
        python scripts/korbi/campaign_midtraining/long_realistic_eval.py --model "$model" \
          --data "$D/long_evals" --out "$out" --tasks en_mc en_qa code_debug --max-len $L --n 50 ;;
  # 3x the samples where the default counts were too noisy to separate arms (2026-11-22): InfiniteBench 150 per task,
  # key-value retrieval 180 per length at 128k / 256k and 90 at 512k / 1M (passkey saturates, vt is ~0 at s3 scale)
  real_inf128k_x3|real_inf256k_x3) L=$([ $task = real_inf128k_x3 ] && echo 131072 || echo 262144)
        python scripts/korbi/campaign_midtraining/long_realistic_eval.py --model "$model" \
          --data "$D/long_evals" --out "$out" --tasks en_mc en_qa code_debug --max-len $L --n 150 ;;
  kv_long_x3) python scripts/korbi/campaign_midtraining/retrieval_gen.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out" --lengths 131072 262144 --n 180 --tasks kv ;;
  kv_512k_x3|kv_1m_x3) L=$([ $task = kv_512k_x3 ] && echo 524288 || echo 1048576)
        python scripts/korbi/campaign_midtraining/retrieval_gen.py --model "$model" --raw "$DS/raw.jsonl" \
          --out "$out" --lengths $L --n 90 --tasks kv ;;
  *) echo "unknown task $task" >&2; exit 2 ;;
esac
echo "[t16_long_evals] $arm $task done"
