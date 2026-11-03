#!/bin/bash
set -euo pipefail
# One allocation, two independent nodes; no model collectives cross the nodes.
case ${SLURM_NODEID:-0} in
 0) layout=ep4 ;;
 1) layout=tp4 ;;
 *) echo 'Expected exactly two nodes' >&2; exit 1 ;;
esac
source scripts/korbi/vllm/setup_vllm_uv.sh
export OELLM_MHC_IMPL=triton OELLM_LOCAL_EXPERT_LOAD=1
export OELLM_ENGRAM_TABLES=ram OELLM_ENGRAM_LOOKUP=leader OELLM_ENGRAM_MADVISE=normal
export VLLM_LOGGING_LEVEL=INFO
RESULT=$PWD/experiment/008_ambitious_inference_speed/results/batching/$layout
mkdir -p "$RESULT"
printf '%s\n' "job=$SLURM_JOB_ID node=$(hostname) layout=$layout" > "$RESULT/allocation.txt"
"$INFER_PY" experiment/008_ambitious_inference_speed/scripts/bench_batching.py \
 --model /gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700 \
 --layout "$layout" --out "$RESULT" --requests 1024 --decode 32
