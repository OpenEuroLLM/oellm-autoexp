#!/bin/bash
#SBATCH --job-name=dllm-tok-fwedu
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --array=0-15
# Tokenize the FineWeb-Edu sample-10BT JSONL shards with the Qwen3 tokenizer into Megatron
# .bin/.idx files (EOD = <|endoftext|> appended to every document), one array task per shard.
#
# Prepare the shards first (see oellm_autoexp/dllm/data_prep.py), then submit from the
# repository root with the log path (SBATCH directives cannot expand variables):
#   sbatch -o "$OUTPUT_DIR/dllm-ablations/logs/p2/tok-%A_%a.log" \
#       oellm_autoexp/dllm/tokenize_fineweb_edu.sbatch
# Needs DATA_DIR and HF_HOME (with Qwen/Qwen3-0.6B-Base prefetched); runs offline with the CPU venv
# .venv-dllm (torch + transformers), because apptainer cannot create user namespaces on genoa nodes.
set -euo pipefail

SHARD=$(printf %02d "${SLURM_ARRAY_TASK_ID}")
INPUT="${DATA_DIR}/raw/fineweb-edu-jsonl/shard_${SHARD}.jsonl"
PREFIX="${DATA_DIR}/qwen3_tokenized/fineweb_edu_10BT/shard_${SHARD}"
mkdir -p "$(dirname "${PREFIX}")"
test -s "${INPUT}" || { echo "shard ${SHARD}: missing input ${INPUT}" >&2; exit 1; }

echo "shard ${SHARD}: ${INPUT} -> ${PREFIX}_text_document.{bin,idx} on $(hostname)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=submodules/MegaDLMs \
  .venv-dllm/bin/python submodules/MegaDLMs/tools/preprocess_data.py \
    --input "${INPUT}" \
    --json-keys text \
    --output-prefix "${PREFIX}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model Qwen/Qwen3-0.6B-Base \
    --append-eod \
    --workers "${SLURM_CPUS_PER_TASK}"
test -s "${PREFIX}_text_document.bin" && test -s "${PREFIX}_text_document.idx" \
  || { echo "shard ${SHARD}: missing output ${PREFIX}_text_document.{bin,idx}" >&2; exit 1; }
echo "shard ${SHARD}: done"
