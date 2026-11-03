#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
RESULT=$PWD/experiment/008_ambitious_inference_speed/results/optimized
BASE=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
FAST=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700_hf008
PROMPTS=$R/autoexp_infer008/experiment/008_ambitious_inference_speed/results/ep4/prompts.json
mkdir -p "$RESULT"
source experiment/008_ambitious_inference_speed/scripts/setup_runtime.sh
# The new directory hard-links immutable weights; model code is atomically replaced.
[ -d "$FAST" ] || cp -al "$BASE" "$FAST"
for file in configuration_oellm_ambitious.py modeling_oellm_ambitious.py mhc_kernels.py; do
  cp "oellm_autoexp/hf_export/oellm_ambitious/$file" "$FAST/$file.new"
  mv "$FAST/$file.new" "$FAST/$file"
done
for file in configuration_oellm_hybrid.py modeling_oellm_hybrid.py; do
  cp "oellm_autoexp/hf_export/oellm_hybrid/$file" "$FAST/$file.new"
  mv "$FAST/$file.new" "$FAST/$file"
done
for mode in reference fast; do
  model=$BASE; [ "$mode" = reference ] || model=$FAST
  "$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_hf_full.py \
    --model "$model" --case "$mode" --prompts "$PROMPTS" --out "$RESULT/hf_$mode.json" \
    2>&1 | tee "$RESULT/hf_$mode.log"
done
unset OELLM_HF_MOE OELLM_HF_ENGRAM_LOOKUP
export OELLM_MHC_IMPL=reference OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_ENGRAM_TABLES=mmap OELLM_ENGRAM_LOOKUP=replicated
"$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_vllm.py \
  --model "$BASE" --layout tp4 --global-token-budget 2048 --out "$RESULT/tp4_tuned" \
  2>&1 | tee "$RESULT/tp4_tuned.log"
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_MADVISE=normal OELLM_ENGRAM_LOOKUP=leader
for layout in ep4 tp4; do
  "$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_vllm.py \
    --model "$BASE" --layout "$layout" --out "$RESULT/$layout" \
    2>&1 | tee "$RESULT/$layout.log"
done
bash experiment/008_ambitious_inference_speed/scripts/run_eval_smoke.sh
