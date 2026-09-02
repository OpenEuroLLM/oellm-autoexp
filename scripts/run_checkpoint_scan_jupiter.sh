#!/usr/bin/env bash
# Run full checkpoint-state scans inside the production JUPITER container.
#
# This script is normally launched through
# config/experiments/oellm_32b_dense/checkpoint_stats_jupiter.yaml, which
# supplies the production container, binds, environment, account, and partition.
# It can also be called manually from an equivalent allocation.

set -euo pipefail

MODE="${1:-model_all}"
OUTPUT_ROOT="${2:?usage: run_checkpoint_scan_jupiter.sh MODE OUTPUT_ROOT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"

FLAGSHIP_ROOT="/e/scratch/e-sta-openeurollm/production_training/oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4/checkpoints"
CONT4_ROOT="/e/scratch/e-sta-openeurollm/production_training/oellm_32b_dense_prod_dataopt5_cont4_bf16_seed1234/checkpoints"
CONT4B_ROOT="/e/scratch/e-sta-openeurollm/production_training/oellm_32b_dense_prod_dataopt5_cont4b_bf16_seed1234/checkpoints"

MODEL_ROOT="$OUTPUT_ROOT/model"
OPTIMIZER_ROOT="$OUTPUT_ROOT/optimizer"

export PYTHONPATH="$REPO_ROOT/submodules/Megatron-LM:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo "[checkpoint-scan] mode       : $MODE"
echo "[checkpoint-scan] repo       : $REPO_ROOT"
echo "[checkpoint-scan] python     : $PYTHON_BIN"
echo "[checkpoint-scan] output     : $OUTPUT_ROOT"
echo "[checkpoint-scan] container  : ${APPTAINER_NAME:-${SINGULARITY_NAME:-unknown}}"
"$PYTHON_BIN" -c "import torch, transformer_engine; print('[checkpoint-scan] torch', torch.__version__, 'TE', transformer_engine.__version__)"

require_checkpoint() {
    local checkpoint="$1"
    if [[ ! -f "$checkpoint/.metadata" ]]; then
        echo "[checkpoint-scan] FATAL: missing checkpoint metadata: $checkpoint/.metadata" >&2
        exit 2
    fi
}

require_flagship_model_checkpoints() {
    require_checkpoint "$FLAGSHIP_ROOT/iter_0060000"
    require_checkpoint "$FLAGSHIP_ROOT/iter_0064000"
    require_checkpoint "$FLAGSHIP_ROOT/iter_0068000"
    require_checkpoint "$FLAGSHIP_ROOT/iter_0072000"
    require_checkpoint "$FLAGSHIP_ROOT/iter_0075126"
}

require_bf16_model_checkpoints() {
    require_checkpoint "$CONT4_ROOT/iter_0060000"
    require_checkpoint "$CONT4B_ROOT/iter_0064000"
    require_checkpoint "$CONT4B_ROOT/iter_0068000"
}

scan_model_flagship() {
    require_flagship_model_checkpoints
    "$PYTHON_BIN" scripts/scan_checkpoint_stats.py \
        "$FLAGSHIP_ROOT" \
        --iterations 60000,64000,68000,72000,75126 \
        --run flagship \
        --optimizer off \
        --comparison-sample-elements 16384 \
        --strict \
        --fail-on-nonfinite \
        --output-dir "$MODEL_ROOT/flagship"
}

scan_model_bf16() {
    require_bf16_model_checkpoints
    "$PYTHON_BIN" scripts/scan_checkpoint_stats.py \
        "$CONT4_ROOT/iter_0060000" \
        "$CONT4B_ROOT/iter_0064000" \
        "$CONT4B_ROOT/iter_0068000" \
        --run bf16 \
        --optimizer off \
        --comparison-sample-elements 16384 \
        --strict \
        --fail-on-nonfinite \
        --output-dir "$MODEL_ROOT/bf16"
}

compare_model_runs() {
    "$PYTHON_BIN" scripts/compare_checkpoint_stats.py \
        --scan "flagship=$MODEL_ROOT/flagship" \
        --scan "bf16=$MODEL_ROOT/bf16" \
        --pair bf16:flagship \
        --baseline-iteration 60000 \
        --output-dir "$MODEL_ROOT/comparison"
}

scan_optimizer_flagship() {
    require_flagship_model_checkpoints
    "$PYTHON_BIN" scripts/scan_checkpoint_stats.py \
        "$FLAGSHIP_ROOT" \
        --iterations 60000,64000,68000,72000,75126 \
        --run flagship_optimizer \
        --include optimizer \
        --optimizer on \
        --optimizer-states exp_avg,exp_avg_sq \
        --comparison-sample-elements 4096 \
        --no-channels \
        --strict \
        --fail-on-nonfinite \
        --output-dir "$OPTIMIZER_ROOT/flagship"
}

scan_optimizer_bf16() {
    require_bf16_model_checkpoints
    "$PYTHON_BIN" scripts/scan_checkpoint_stats.py \
        "$CONT4_ROOT/iter_0060000" \
        "$CONT4B_ROOT/iter_0064000" \
        "$CONT4B_ROOT/iter_0068000" \
        --run bf16_optimizer \
        --include optimizer \
        --optimizer on \
        --optimizer-states exp_avg,exp_avg_sq \
        --comparison-sample-elements 4096 \
        --no-channels \
        --strict \
        --fail-on-nonfinite \
        --output-dir "$OPTIMIZER_ROOT/bf16"
}

compare_optimizer_runs() {
    "$PYTHON_BIN" scripts/compare_checkpoint_stats.py \
        --scan "flagship_optimizer=$OPTIMIZER_ROOT/flagship" \
        --scan "bf16_optimizer=$OPTIMIZER_ROOT/bf16" \
        --pair bf16_optimizer:flagship_optimizer \
        --baseline-iteration 60000 \
        --no-plot \
        --output-dir "$OPTIMIZER_ROOT/comparison"
}

case "$MODE" in
    model_flagship)
        scan_model_flagship
        ;;
    model_bf16)
        scan_model_bf16
        ;;
    model_all)
        scan_model_flagship
        scan_model_bf16
        compare_model_runs
        ;;
    optimizer_flagship)
        scan_optimizer_flagship
        ;;
    optimizer_bf16)
        scan_optimizer_bf16
        ;;
    optimizer_all)
        scan_optimizer_flagship
        scan_optimizer_bf16
        compare_optimizer_runs
        ;;
    all)
        scan_model_flagship
        scan_model_bf16
        compare_model_runs
        scan_optimizer_flagship
        scan_optimizer_bf16
        compare_optimizer_runs
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        echo "valid modes: model_flagship model_bf16 model_all optimizer_flagship optimizer_bf16 optimizer_all all" >&2
        exit 2
        ;;
esac

echo "[checkpoint-scan] completed: $MODE"
echo "[checkpoint-scan] artifacts: $OUTPUT_ROOT"
