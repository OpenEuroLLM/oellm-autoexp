#!/usr/bin/env bash
# Scan only the surviving flagship checkpoints before iteration 60,000.
# Run inside the same JUPITER production container/environment as
# run_checkpoint_scan_jupiter.sh.

set -euo pipefail

RUN_ID="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${1:-/e/home/jusers/luukkonen1/jupiter/e-sta-workdir/oellm-autoexp/checkpoint_scan_prior_${RUN_ID}/artifacts}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
CHECKPOINT_ROOT="/e/scratch/e-sta-openeurollm/production_training/oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4/checkpoints"
ITERATIONS="4000,8000,12000,16000,20000,24000,28000,32000,34454,36000,40000,43200,44000,48000,52000,56000,57940"

export PYTHONPATH="$REPO_ROOT/submodules/Megatron-LM:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

for iteration in ${ITERATIONS//,/ }; do
    printf -v checkpoint "%s/iter_%07d" "$CHECKPOINT_ROOT" "$iteration"
    if [[ ! -f "$checkpoint/.metadata" ]]; then
        echo "[prior-checkpoint-scan] FATAL: missing checkpoint metadata: $checkpoint/.metadata" >&2
        exit 2
    fi
done

echo "[prior-checkpoint-scan] repo       : $REPO_ROOT"
echo "[prior-checkpoint-scan] python     : $PYTHON_BIN"
echo "[prior-checkpoint-scan] checkpoints: $ITERATIONS"
echo "[prior-checkpoint-scan] output     : $OUTPUT_ROOT"
echo "[prior-checkpoint-scan] container  : ${APPTAINER_NAME:-${SINGULARITY_NAME:-unknown}}"
"$PYTHON_BIN" -c "import torch, transformer_engine; print('[prior-checkpoint-scan] torch', torch.__version__, 'TE', transformer_engine.__version__)"

"$PYTHON_BIN" scripts/scan_checkpoint_stats.py \
    "$CHECKPOINT_ROOT" \
    --iterations "$ITERATIONS" \
    --run flagship_prior \
    --optimizer off \
    --comparison-sample-elements 16384 \
    --strict \
    --fail-on-nonfinite \
    --output-dir "$OUTPUT_ROOT/flagship_prior"

"$PYTHON_BIN" scripts/compare_checkpoint_stats.py \
    --scan "flagship_prior=$OUTPUT_ROOT/flagship_prior" \
    --output-dir "$OUTPUT_ROOT/comparison"

echo "[prior-checkpoint-scan] completed"
echo "[prior-checkpoint-scan] artifacts: $OUTPUT_ROOT"
