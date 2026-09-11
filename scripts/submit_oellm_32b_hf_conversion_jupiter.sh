#!/usr/bin/env bash
# Prepare and submit a production dense-32B Megatron -> Hugging Face conversion
# on JUPITER. No model data is copied to the submission host.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_RUN_NAME="oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4"
RUN_NAME="${OELLM_32B_RUN_NAME:-$DEFAULT_RUN_NAME}"
ITERATION="${OELLM_32B_CKPT_ITER:-latest}"
CHECKPOINT_ROOT="${OELLM_32B_CHECKPOINT_ROOT:-/e/scratch/e-sta-openeurollm/production_training}"
TRAINING_ROOT="${OELLM_32B_TRAINING_ROOT:-/e/project1/e-sta-openeurollm/production_training}"
TOKENIZER="${OELLM_32B_TOKENIZER:-/e/data1/datasets/playground/mmlaion/oellm/oellm_tokenizer_256k}"
BRIDGE_IMAGE="${OELLM_32B_BRIDGE_IMAGE:-/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-bridge-bridge_aarch64_202605191331.sif}"
HF_OUTPUT_OVERRIDE=""
JOB_DIR_OVERRIDE=""
DRY_RUN=false
SUBMIT_AND_EXIT=false

usage() {
    echo "Usage: $0 [--run-name NAME] [--iteration ITER|latest] [--output DIR]" >&2
    echo "          [--job-dir DIR] [--dry-run] [--submit-and-exit]" >&2
    echo "" >&2
    echo "Optional path overrides: OELLM_32B_CHECKPOINT_ROOT, OELLM_32B_TRAINING_ROOT," >&2
    echo "OELLM_32B_TOKENIZER, OELLM_32B_BRIDGE_IMAGE, OELLM_32B_HF_OUTPUT," >&2
    echo "OELLM_32B_CONVERSION_JOB_DIR." >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)
            RUN_NAME="${2:?--run-name requires a value}"
            shift 2
            ;;
        --iteration)
            ITERATION="${2:?--iteration requires a value}"
            shift 2
            ;;
        --output)
            HF_OUTPUT_OVERRIDE="${2:?--output requires a value}"
            shift 2
            ;;
        --job-dir)
            JOB_DIR_OVERRIDE="${2:?--job-dir requires a value}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --submit-and-exit)
            SUBMIT_AND_EXIT=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

CHECKPOINT_RUN_DIR="$CHECKPOINT_ROOT/$RUN_NAME/checkpoints"
TRAINING_RUN_DIR="$TRAINING_ROOT/$RUN_NAME"

if [[ "$ITERATION" == "latest" ]]; then
    TRACKER="$CHECKPOINT_RUN_DIR/latest_checkpointed_iteration.txt"
    if [[ -f "$TRACKER" ]]; then
        TRACKED_ITER="$(tr -dc '0-9' < "$TRACKER")"
        if [[ -n "$TRACKED_ITER" ]]; then
            printf -v TRACKED_DIR 'iter_%07d' "$((10#$TRACKED_ITER))"
            if [[ -f "$CHECKPOINT_RUN_DIR/$TRACKED_DIR/.metadata" ]]; then
                ITERATION="$TRACKED_DIR"
            else
                echo "WARNING: tracker points to incomplete $TRACKED_DIR; scanning completed checkpoints" >&2
            fi
        fi
    fi
    if [[ "$ITERATION" == "latest" ]]; then
        LATEST_NUM=-1
        shopt -s nullglob
        for CANDIDATE in "$CHECKPOINT_RUN_DIR"/iter_*; do
            [[ -f "$CANDIDATE/.metadata" ]] || continue
            CANDIDATE_NUM="${CANDIDATE##*/iter_}"
            [[ "$CANDIDATE_NUM" =~ ^[0-9]+$ ]] || continue
            if (( 10#$CANDIDATE_NUM > LATEST_NUM )); then
                LATEST_NUM=$((10#$CANDIDATE_NUM))
            fi
        done
        shopt -u nullglob
        if (( LATEST_NUM < 0 )); then
            echo "FATAL: no complete iter_* checkpoint under $CHECKPOINT_RUN_DIR" >&2
            exit 2
        fi
        printf -v ITERATION 'iter_%07d' "$LATEST_NUM"
    fi
elif [[ "$ITERATION" =~ ^[0-9]+$ ]]; then
    printf -v ITERATION 'iter_%07d' "$((10#$ITERATION))"
elif [[ ! "$ITERATION" =~ ^iter_[0-9]{7}$ ]]; then
    echo "FATAL: iteration must be an integer, iter_NNNNNNN, or latest: $ITERATION" >&2
    exit 2
fi

CHECKPOINT="${OELLM_32B_CHECKPOINT:-$CHECKPOINT_RUN_DIR/$ITERATION}"
TRAINING_CONFIG="${OELLM_32B_TRAINING_CONFIG:-$TRAINING_RUN_DIR/logs/current.yaml}"
HF_OUTPUT="${HF_OUTPUT_OVERRIDE:-${OELLM_32B_HF_OUTPUT:-$TRAINING_RUN_DIR/hf/$ITERATION}}"
CONVERSION_JOB_DIR="${JOB_DIR_OVERRIDE:-${OELLM_32B_CONVERSION_JOB_DIR:-$TRAINING_RUN_DIR/hf_conversion/$ITERATION}}"

if [[ -z "${OELLM_32B_TRAINING_CONFIG:-}" && ! -f "$TRAINING_CONFIG" ]]; then
    NEWEST_CONFIG=""
    shopt -s nullglob
    for CANDIDATE in "$TRAINING_RUN_DIR"/logs/config-*.yaml; do
        if [[ -z "$NEWEST_CONFIG" || "$CANDIDATE" -nt "$NEWEST_CONFIG" ]]; then
            NEWEST_CONFIG="$CANDIDATE"
        fi
    done
    shopt -u nullglob
    if [[ -n "$NEWEST_CONFIG" ]]; then
        TRAINING_CONFIG="$NEWEST_CONFIG"
    fi
fi

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        echo "FATAL: missing $label: $path" >&2
        exit 2
    fi
}

require_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "$path" ]]; then
        echo "FATAL: missing $label: $path" >&2
        exit 2
    fi
}

require_dir "$CHECKPOINT" "checkpoint directory"
require_file "$CHECKPOINT/.metadata" "completed torch_dist metadata"
require_file "$TRAINING_CONFIG" "resolved training config"
require_dir "$TOKENIZER" "production tokenizer"
require_file "$BRIDGE_IMAGE" "Megatron-Bridge container"
require_file "$REPO_ROOT/submodules/Megatron-Bridge/src/megatron/bridge/__init__.py" "Megatron-Bridge submodule"

if [[ ! -f "$TOKENIZER/tokenizer.json" && ! -f "$TOKENIZER/tokenizer.model" ]]; then
    echo "FATAL: tokenizer has neither tokenizer.json nor tokenizer.model: $TOKENIZER" >&2
    exit 2
fi
if [[ -e "$HF_OUTPUT" ]]; then
    echo "FATAL: output already exists (converter never overwrites): $HF_OUTPUT" >&2
    exit 2
fi

# The patch is idempotent and is needed when Bridge contains optional model
# imports that the OpenEuroLLM Megatron fork does not provide.
PATCH_PYTHON="$(command -v python3 || command -v python || true)"
if [[ -z "$PATCH_PYTHON" ]]; then
    echo "FATAL: python3/python is required to prepare Megatron-Bridge" >&2
    exit 2
fi
"$PATCH_PYTHON" "$REPO_ROOT/container/megatron/patch_bridge_lazy_imports.py" \
    "$REPO_ROOT/submodules/Megatron-Bridge/src/megatron/bridge/models"

export OELLM_AUTOEXP_ROOT="$REPO_ROOT"
export OELLM_32B_CKPT_ITER="$ITERATION"
export OELLM_32B_CHECKPOINT="$CHECKPOINT"
export OELLM_32B_TRAINING_CONFIG="$TRAINING_CONFIG"
export OELLM_32B_HF_OUTPUT="$HF_OUTPUT"
export OELLM_32B_CONVERSION_JOB_DIR="$CONVERSION_JOB_DIR"
export OELLM_32B_TOKENIZER="$TOKENIZER"
export OELLM_32B_BRIDGE_IMAGE="$BRIDGE_IMAGE"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-e-sta-openeurollm}"

echo "[32b-hf] run             : $RUN_NAME"
echo "[32b-hf] checkpoint      : $CHECKPOINT"
echo "[32b-hf] training config : $TRAINING_CONFIG"
echo "[32b-hf] tokenizer       : $TOKENIZER"
echo "[32b-hf] output          : $HF_OUTPUT"
echo "[32b-hf] job artifacts   : $CONVERSION_JOB_DIR"
echo "[32b-hf] container       : $BRIDGE_IMAGE"

ARGS=(--config-name experiments/oellm_32b_dense/convert_hf_jupiter)
if [[ "$DRY_RUN" == true ]]; then
    ARGS+=(--dry-run)
elif [[ "$SUBMIT_AND_EXIT" == true ]]; then
    ARGS+=(--submit-and-exit)
fi

exec "$REPO_ROOT/scripts/oellm_32b.sh" "${ARGS[@]}"
