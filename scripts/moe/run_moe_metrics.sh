#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for scripts/moe/compute_moe_metrics.py with practical defaults.
#
# Example:
#   bash scripts/moe/run_moe_metrics.sh \
#     --load /path/to/checkpoint \
#     --vocab-file /path/to/vocab.json \
#     --merge-file /path/to/merges.txt \
#     --plot-dir /path/to/plots
#
# Any extra args passed after "--" are forwarded to compute_moe_metrics.py.
#
# Example with overrides:
#   bash scripts/moe/run_moe_metrics.sh \
#     --load /path/to/checkpoint \
#     --vocab-file /path/to/vocab.json \
#     --merge-file /path/to/merges.txt \
#     --plot-dir /path/to/plots \
#     --num-batches 500 --seq-length 1024 --dataset-percentage 10.0 \
#     -- --moe-compare-ckpts 500,1000,1500

LOAD=""
VOCAB_FILE=""
MERGE_FILE=""
PLOT_DIR=""

NUM_BATCHES=2000
SEQ_LENGTH=2048
BATCH_SIZE=1
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
DATASET_SPLIT="test"
DATASET_PERCENTAGE="100.0"
OUTPUT_JSON="saturation_results.json"
COACT_JSON="coactivation_results.json"

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/moe/run_moe_metrics.sh \
    --load <checkpoint_dir> \
    --vocab-file <vocab_file> \
    --merge-file <merge_file> \
    --plot-dir <plot_dir> \
    [--num-batches 2000] \
    [--seq-length 2048] \
    [--batch-size 1] \
    [--dataset-name wikitext] \
    [--dataset-config wikitext-2-raw-v1] \
    [--dataset-split test] \
    [--dataset-percentage 100.0] \
    [--output-json saturation_results.json] \
    [--coactivation-output coactivation_results.json] \
    [-- <extra compute_moe_metrics.py args>]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --load)
      LOAD="$2"
      shift 2
      ;;
    --vocab-file)
      VOCAB_FILE="$2"
      shift 2
      ;;
    --merge-file)
      MERGE_FILE="$2"
      shift 2
      ;;
    --plot-dir)
      PLOT_DIR="$2"
      shift 2
      ;;
    --num-batches)
      NUM_BATCHES="$2"
      shift 2
      ;;
    --seq-length)
      SEQ_LENGTH="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --dataset-config)
      DATASET_CONFIG="$2"
      shift 2
      ;;
    --dataset-split)
      DATASET_SPLIT="$2"
      shift 2
      ;;
    --dataset-percentage)
      DATASET_PERCENTAGE="$2"
      shift 2
      ;;
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --coactivation-output)
      COACT_JSON="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$LOAD" || -z "$VOCAB_FILE" || -z "$MERGE_FILE" || -z "$PLOT_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

python scripts/moe/compute_moe_metrics.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --load "$LOAD" \
  --use-checkpoint-args \
  --ckpt-format torch_dist \
  --vocab-file "$VOCAB_FILE" \
  --merge-file "$MERGE_FILE" \
  --legacy-tokenizer \
  --no-bias-dropout-fusion \
  --moe-data-type hf_dataset \
  --moe-dataset-name "$DATASET_NAME" \
  --moe-dataset-config "$DATASET_CONFIG" \
  --moe-dataset-split "$DATASET_SPLIT" \
  --moe-dataset-percentage "$DATASET_PERCENTAGE" \
  --moe-seq-length "$SEQ_LENGTH" \
  --moe-batch-size "$BATCH_SIZE" \
  --moe-num-batches "$NUM_BATCHES" \
  --moe-output-json "$OUTPUT_JSON" \
  --moe-coactivation-output "$COACT_JSON" \
  --moe-plot \
  --moe-plot-dir "$PLOT_DIR" \
  "${EXTRA_ARGS[@]}"
