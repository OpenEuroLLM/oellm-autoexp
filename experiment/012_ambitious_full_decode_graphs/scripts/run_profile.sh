#!/bin/bash
# 012: nsys timeline of full-graph decode, one request per GPU
set -uo pipefail
source scripts/korbi/vllm/setup_vllm_uv.sh
OUT=$PWD/experiment/012_ambitious_full_decode_graphs/results/${RESULT_TAG:-profile1}
mkdir -p "$OUT"
MODEL=${MODEL:-/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_6000}
PROMPTS=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_batch/experiment/008_ambitious_inference_speed/results/batching/ep4/prompts.json
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1 OELLM_ENGRAM_DECODE_IMPL=batched
export OELLM_FUSION_IMPL=triton OELLM_ENGRAM_MADVISE=normal VLLM_LOGGING_LEVEL=INFO
export OELLM_ENGRAM_TABLES=uva OELLM_DECODE_GRAPHS=0 OELLM_ENGRAM_LOOKUP=replicated
export OELLM_DECODE_KERNELS=${OELLM_DECODE_KERNELS:-reference}
rm -f /dev/shm/oellm_engram_* 2>/dev/null
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$OUT/decode" \
  "$INFER_PY" experiment/012_ambitious_full_decode_graphs/scripts/nsys_decode.py --model "$MODEL" --prompts "$PROMPTS" \
  > "$OUT/profile.log" 2>&1
echo "profile rc=$?"
grep decode_tok_s "$OUT/profile.log"
for t in "$OUT"/decode*.nsys-rep; do
  nsys export --type=sqlite --force-overwrite=true -o "${t%.nsys-rep}.sqlite" "$t" >/dev/null 2>&1
  "$INFER_PY" experiment/012_ambitious_full_decode_graphs/scripts/analyze.py "${t%.nsys-rep}.sqlite" > "${t%.nsys-rep}.analysis.json" 2>&1
  head -c 4000 "${t%.nsys-rep}.analysis.json"
done
rm -f /dev/shm/oellm_engram_* 2>/dev/null
echo PROFILE012_DONE
