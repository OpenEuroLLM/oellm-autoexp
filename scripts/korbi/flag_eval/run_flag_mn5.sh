#!/bin/bash
# FLAG lm-eval rows on one MareNostrum 5 node (config/experiments/korbi/production/flag_eval_mn5.yaml):
# vLLM (submodules/vllm on the 0.32 wheels) installed into node-local /tmp first, the image's
# lm-eval 0.4.13 (JUPITER: 0.4.12 + a DP patch; rows run TP + EP here, no DP), the eval
# extras the image lacks from $R/pylibs_flag (ifeval, polymath, rouge), offline datasets from
# the JUPITER HF_HOME copy.
#   run_flag_mn5.sh <export dir> <out dir> <shard> <num shards> <limit, 0 = all> <name> [<name> ...]
set -euo pipefail
# FLAG_SUITE: rows from another suite file of the same schema (e.g. config/eval_tasks/flag_evals_extra.yaml)
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
cd "$(dirname "$0")/../../.."
export_dir=$1 out=$2 shard=$3 nshards=$4 limit=$5; shift 5
source scripts/korbi/vllm/uva_tables_cleanup.sh
SHA=$(grep -m1 "^submodules/vllm " SYNC_STAMP | awk '{print $3}')
export DEST=/tmp/pylibs_vllm_$SHA
bash scripts/korbi/vllm/build_pylibs_vllm.sh
export PYTHONPATH="$DEST:$PWD:$PWD/flash-linear-attention:$R/pylibs_flag"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING} VLLM_WORKER_MULTIPROC_METHOD=spawn
# a row's first step compiles / autotunes the Triton kernels for its batch shapes: hellaswag's
# first step (40k requests) passed vLLM's 300 s step timeout and killed the engine (2026-11-02)
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
export HF_HOME=${FLAG_HF_HOME:-/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home}
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics/job_${SLURM_JOB_ID:-local}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 NLTK_DATA=$R/nltk_data TOKENIZERS_PARALLELISM=false
python3 scripts/korbi/flag_eval/flag_lmeval.py --export "$export_dir" --out "$out" \
    --shard "$shard" --num-shards "$nshards" $([ "$limit" = 0 ] || echo --limit "$limit") \
    ${FLAG_SUITE:+--suite "$FLAG_SUITE"} --names "$@"
