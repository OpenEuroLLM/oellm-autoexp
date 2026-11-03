#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
MODEL=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700
RESULT=$PWD/experiment/008_ambitious_inference_speed/results/optimized
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics/infer008_${SLURM_JOB_ID:-local}
export HF_EVALUATE_OFFLINE=1 HF_ALLOW_CODE_EVAL=1 NLTK_DATA=$R/nltk_data
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_LOOKUP=leader
"$INFER_PY" scripts/korbi/flag_eval/flag_lmeval.py \
 --export "$MODEL" --out /gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_infer008_smoke \
 --shard 0 --num-shards 1 --tp 1 --dp 4 --ep 1 --max-num-seqs 32 \
 --max-num-batched-tokens 2048 --request-batch-size 64 --response-cache \
 --limit 17 --names vllm/hellaswag/0 vllm/gsm8k/4 \
 2>&1 | tee "$RESULT/lmeval_ep4_smoke.log"
