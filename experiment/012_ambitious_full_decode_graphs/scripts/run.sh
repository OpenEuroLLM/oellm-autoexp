#!/bin/bash
# 012: whole-step decode CUDA graphs (vLLM FULL_DECODE_ONLY) + UVA Engram tables vs 010's graph islands
set -uo pipefail
source scripts/korbi/vllm/setup_vllm_uv.sh
OUT=$PWD/experiment/012_ambitious_full_decode_graphs/results/${RESULT_TAG:-run1}
mkdir -p "$OUT"
MODEL=${MODEL:-/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_6000}
PROMPTS=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_batch/experiment/008_ambitious_inference_speed/results/batching/ep4/prompts.json
{ hostname; df -h /dev/shm /tmp; free -g; nvidia-smi -L; } > "$OUT/node.txt" 2>&1
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1 OELLM_ENGRAM_DECODE_IMPL=batched
export OELLM_FUSION_IMPL=triton OELLM_ENGRAM_MADVISE=normal VLLM_LOGGING_LEVEL=INFO
VARIANTS=${VARIANTS:-fullgraph,islands}
rm -f /dev/shm/oellm_engram_* 2>/dev/null  # stale tables of a killed run
if [ "${VARIANTS#*k8}" != "$VARIANTS" ]; then
  "$INFER_PY" experiment/012_ambitious_full_decode_graphs/scripts/check_fp8_w8a16.py 2>&1 | grep -vE "Warning|warn" | tail -8 | tee "$OUT/check_fp8_w8a16.log"
  grep -q FP8_W8A16_EXACT "$OUT/check_fp8_w8a16.log" || { echo "fp8 w8a16 check failed"; exit 1; }
fi
status=0
for v in ${VARIANTS//,/ }; do
  off=""; [ "${v#*-}" != "$v" ] && off=${v#*-}
  base=${v%%-*}
  export OELLM_DECODE_KERNELS=reference OELLM_MOE_FP8= OELLM_DECODE_KERNELS_OFF=${off//+/,}
  if [ "${base#fullgraph}" != "$base" ]; then
    export OELLM_ENGRAM_TABLES=uva OELLM_DECODE_GRAPHS=0 OELLM_ENGRAM_LOOKUP=replicated
    [ "${base#fullgraph_k}" != "$base" ] && export OELLM_DECODE_KERNELS=triton
    [ "$base" = fullgraph_k8 ] && export OELLM_MOE_FP8=w8a16
  else
    export OELLM_ENGRAM_TABLES=ram OELLM_DECODE_GRAPHS=1 OELLM_ENGRAM_LOOKUP=leader
  fi
  timeout ${CASE_TIMEOUT:-2400} "$INFER_PY" experiment/012_ambitious_full_decode_graphs/scripts/bench.py \
    --model "$MODEL" --prompts "$PROMPTS" --out "$OUT" --variant "$v" --cases "${CASES:-node1,gpu1,gpu8,gpu32}" > "$OUT/$v.log" 2>&1 || { echo "$v failed rc=$?"; status=1; }
  grep -E '"case"' "$OUT/$v.log" || tail -30 "$OUT/$v.log"
  sleep 5
done
rm -f /dev/shm/oellm_engram_* 2>/dev/null
echo "BENCH012_DONE status=$status"
exit $status
