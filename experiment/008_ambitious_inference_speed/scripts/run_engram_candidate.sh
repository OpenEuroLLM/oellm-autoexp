#!/bin/bash
set -euo pipefail
cd /gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_engram
source scripts/korbi/vllm/setup_vllm_uv.sh
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_LOOKUP=leader OELLM_ENGRAM_MADVISE=normal
export OELLM_ENGRAM_DECODE_IMPL=reference VLLM_LOGGING_LEVEL=INFO
RESULT=$PWD/experiment/008_ambitious_inference_speed/results/engram_candidate
mkdir -p "$RESULT"
# Exactly the same 1024 requests as the completed baseline sweep.
cp /gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_batch/experiment/008_ambitious_inference_speed/results/batching/ep4/prompts.json "$RESULT/prompts.json"
"$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_batching.py \
 --model /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700 \
 --layout ep4 --out "$RESULT" --requests 1024 --decode 32 --engram-compare
bash experiment/008_ambitious_inference_speed/scripts/run_engram_quality.sh
bash experiment/008_ambitious_inference_speed/scripts/run_single_profile.sh
