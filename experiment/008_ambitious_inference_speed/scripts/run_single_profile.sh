#!/bin/bash
set -euo pipefail
export OELLM_ENGRAM_DECODE_IMPL=batched
RESULT=$PWD/experiment/008_ambitious_inference_speed/results/engram_single_profile
mkdir -p "$RESULT"
cp experiment/008_ambitious_inference_speed/results/engram_candidate/prompts.json "$RESULT/prompts.json"
"$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_batching.py \
 --model /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700 \
 --layout ep4 --out "$RESULT" --profile-single
