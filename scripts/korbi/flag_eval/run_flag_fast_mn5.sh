#!/bin/bash
# Tested MN5 vLLM path: node-local EP, host-table prefetch, cached bounded request batches.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export_dir=$1 out=$2 shard=$3 nshards=$4 limit=$5; shift 5
source scripts/korbi/vllm/setup_vllm_uv.sh
source scripts/korbi/vllm/uva_tables_cleanup.sh
export OELLM_MHC_IMPL=${OELLM_MHC_IMPL:-triton}
export OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_FUSION_IMPL=${OELLM_FUSION_IMPL:-reference}
export OELLM_DECODE_GRAPHS=${OELLM_DECODE_GRAPHS:-0}
export OELLM_GRAPH_MAX_ROWS=${OELLM_GRAPH_MAX_ROWS:-256}
export OELLM_ENGRAM_DECODE_IMPL=${OELLM_ENGRAM_DECODE_IMPL:-batched}
export OELLM_ENGRAM_TABLES=${OELLM_ENGRAM_TABLES:-ram}
export OELLM_ENGRAM_LOOKUP=${OELLM_ENGRAM_LOOKUP:-leader}
export OELLM_ENGRAM_MADVISE=${OELLM_ENGRAM_MADVISE:-normal}
args=()
[ "$limit" = 0 ] || args+=(--limit "$limit")
"$INFER_PY" scripts/korbi/flag_eval/flag_lmeval.py \
 --export "$export_dir" --out "$out" --shard "$shard" --num-shards "$nshards" \
 --tp "${FLAG_TP:-1}" --dp "${FLAG_DP:-4}" --ep 1 \
 --max-num-seqs "${FLAG_MAX_NUM_SEQS:-256}" \
 --max-num-batched-tokens "${FLAG_TOKEN_BUDGET:-32768}" \
 --request-batch-size "${FLAG_REQUEST_BATCH_SIZE:-256}" --response-cache \
 "${args[@]}" --names "$@"
