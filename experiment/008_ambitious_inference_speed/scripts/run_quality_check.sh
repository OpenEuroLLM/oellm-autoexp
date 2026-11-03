#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
BASE=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
FAST=${BASE}_hf008
RESULT=$PWD/experiment/008_ambitious_inference_speed/results/quality
PROMPTS=$R/autoexp_infer008/experiment/008_ambitious_inference_speed/results/ep4/prompts.json
mkdir -p "$RESULT"
source experiment/008_ambitious_inference_speed/scripts/setup_runtime.sh
"$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_hf_full.py \
 --model "$FAST" --case fast --no-fused-mhc --prompts "$PROMPTS" \
 --out "$RESULT/hf_grouped_only.json" 2>&1 | tee "$RESULT/hf_grouped_only.log"
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics/infer008_ref_${SLURM_JOB_ID:-local}
export HF_EVALUATE_OFFLINE=1 HF_ALLOW_CODE_EVAL=1 NLTK_DATA=$R/nltk_data
export OELLM_MHC_IMPL=reference OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_LOOKUP=leader
"$INFER_PY" scripts/korbi/flag_eval/flag_lmeval.py \
 --export "$BASE" --out /gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_infer008_reference \
 --shard 0 --num-shards 1 --tp 1 --dp 4 --ep 1 --max-num-seqs 32 \
 --max-num-batched-tokens 2048 --request-batch-size 64 --response-cache \
 --limit 17 --names vllm/hellaswag/0 vllm/gsm8k/4 \
 2>&1 | tee "$RESULT/lmeval_reference.log"
bash experiment/008_ambitious_inference_speed/scripts/run_final_checks.sh
