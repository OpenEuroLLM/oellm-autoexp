#!/bin/bash
set -euo pipefail
source scripts/korbi/vllm/setup_vllm_uv.sh
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1 OELLM_ENGRAM_DECODE_IMPL=batched
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_LOOKUP=leader OELLM_ENGRAM_MADVISE=normal
export OELLM_FUSION_IMPL=reference OELLM_DECODE_GRAPHS=0 VLLM_LOGGING_LEVEL=INFO
OUT=$PWD/experiment/010_ambitious_fusion_graphs/results
mkdir -p "$OUT"
nvidia-smi -q > "$OUT/gpus_start.txt"
"$INFER_PY" experiment/010_ambitious_fusion_graphs/scripts/check_kernels.py > "$OUT/kernel_checks.log" 2>&1
"$INFER_PY" experiment/010_ambitious_fusion_graphs/scripts/bench.py \
 --model /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700 \
 --prompts /gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008_batch/experiment/008_ambitious_inference_speed/results/batching/ep4/prompts.json \
 --out "$OUT/paired" > "$OUT/paired.log" 2>&1
export OELLM_FUSION_IMPL=triton OELLM_DECODE_GRAPHS=1
export FLAG_MAX_NUM_SEQS=256 FLAG_TOKEN_BUDGET=32768 FLAG_REQUEST_BATCH_SIZE=64
bash scripts/korbi/flag_eval/run_flag_fast_mn5.sh \
 /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700 \
 /gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_fusion010 0 1 17 \
 vllm/hellaswag/0 vllm/gsm8k/4 > "$OUT/eval.log" 2>&1
nvidia-smi -q > "$OUT/gpus_end.txt"
