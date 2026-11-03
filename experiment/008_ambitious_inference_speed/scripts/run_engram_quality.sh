#!/bin/bash
set -euo pipefail
export OELLM_ENGRAM_DECODE_IMPL=batched
export FLAG_MAX_NUM_SEQS=256 FLAG_TOKEN_BUDGET=32768 FLAG_REQUEST_BATCH_SIZE=64
bash scripts/korbi/flag_eval/run_flag_fast_mn5.sh \
 /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700 \
 /gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_infer008_batched 0 1 17 \
 vllm/hellaswag/0 vllm/gsm8k/4
