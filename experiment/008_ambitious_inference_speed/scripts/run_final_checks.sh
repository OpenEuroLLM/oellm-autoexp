#!/bin/bash
# Independent final stage reusing quality allocation 48458754 after its reference eval.
set -euo pipefail
BASE=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
RESULT=$PWD/experiment/008_ambitious_inference_speed/results/final
mkdir -p "$RESULT"
# Validate the production wrapper/runtime and done-row resume against completed smoke rows.
FLAG_REQUEST_BATCH_SIZE=64 bash scripts/korbi/flag_eval/run_flag_fast_mn5.sh \
 "$BASE" /gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_infer008_smoke 0 1 17 \
 vllm/hellaswag/0 vllm/gsm8k/4 2>&1 | tee "$RESULT/production_resume.log"
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_MADVISE=normal OELLM_ENGRAM_LOOKUP=leader
for layout in ep4 tp4; do
 "$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_vllm.py \
 --model "$BASE" --layout "$layout" --global-token-budget 8192 --out "$RESULT/$layout" \
 2>&1 | tee "$RESULT/$layout.log"
done
