#!/usr/bin/env bash
# Submit one lm-eval job per (model checkpoint × task group).
# All evaluation parameters live here; the sbatch script is kept generic.
# set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Model checkpoints to evaluate
# ---------------------------------------------------------------------------
EXPORT_ROOT="/shared_silo/scratch/rluukkon/oellm/oellm-autoexp/exports"

# Glob pattern selecting which runs to evaluate (applied to run dir basenames).
# Override on command line: RUN_FILTER='*bilingual*' bash run_evals.sh
RUN_FILTER="${RUN_FILTER:-*bilingual*}"

# ---------------------------------------------------------------------------
# Backend & model args
# ---------------------------------------------------------------------------
export LOCAL_LMEVAL="${LOCAL_LMEVAL:-${REPO_ROOT}/submodules/lm-evaluation-harness}"

export MODEL_BACKEND="vllm"
export DTYPE="auto"
export GPU_MEM_UTIL="0.9"
export BATCH_SIZE="auto"

# vllm tensor/data parallelism (unset = vllm default)
# export TENSOR_PARALLEL_SIZE="8"
# export DATA_PARALLEL_SIZE="1"

# ---------------------------------------------------------------------------
# Task groups with their canonical fewshot settings
# Each entry: "TASKS_CSV|NUM_FEWSHOT"
# ---------------------------------------------------------------------------
EVAL_GROUPS=(
    "arc_challenge_mt_fi,arc_challenge|25"
    "truthfulqa,ogx_goldenswagx_fi,goldenswag|0"
    "ogx_hellaswagx_fi|10"
    "ogx_truthfulqax_mc2_fi,ogx_mmlux_FI,gsm8k,ogx_gsm8kx_fi|5"
    "ogx_flores200-trans-eng_Latn-fin_Latn,ogx_flores200-trans-fin_Latn-eng_Latn|5"
)

# ---------------------------------------------------------------------------
# Sbatch settings
# ---------------------------------------------------------------------------
SBATCH_SCRIPT="${SCRIPT_DIR}/eval_lmeval_preemptible.sbatch"

export INCLUDE_PATH="${INCLUDE_PATH:-${SCRIPT_DIR}/tasks}"

SLURM_EXTRA="${SLURM_EXTRA:---mem=256G --partition=amd-tw-verification --time=24:00:00}"

SUBMIT_SLEEP_SECONDS="${SUBMIT_SLEEP_SECONDS:-1}"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
TOTAL=0
SKIPPED=0

for RUN_DIR in "${EXPORT_ROOT}"/${RUN_FILTER}/; do
    [ -d "$RUN_DIR" ] || continue
    RUN_NAME=$(basename "$RUN_DIR")

    for MODEL_PATH in "${RUN_DIR}"iter_*/; do
        [ -d "$MODEL_PATH" ] || continue
        MODEL_PATH="${MODEL_PATH%/}"
        ITER_NAME=$(basename "$MODEL_PATH")
        MODEL_NAME="${RUN_NAME}__${ITER_NAME}"

        for group in "${EVAL_GROUPS[@]}"; do
            TASKS="${group%%|*}"
            FEWSHOT="${group##*|}"
            TASK_SLUG=$(echo "$TASKS" | tr ',:|/ ' '_')
            OUTDIR="${SCRATCH:-/shared_silo/scratch/${USER}}/eval_results/lmeval_${TASK_SLUG}_${MODEL_NAME}"

            if find "$OUTDIR" -maxdepth 2 -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
                echo "skip existing  model=${MODEL_NAME}  fewshot=${FEWSHOT}  tasks=${TASKS}"
                SKIPPED=$(( SKIPPED + 1 ))
                continue
            fi

            job_id=$(
                MODEL="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
                TASKS="${TASKS}" \
                NUM_FEWSHOT="${FEWSHOT}" \
                sbatch --parsable \
                    --export=ALL \
                    ${SLURM_EXTRA} \
                    "${SBATCH_SCRIPT}"
            )

            echo "job ${job_id}  model=${MODEL_NAME}  fewshot=${FEWSHOT}  tasks=${TASKS}"
            TOTAL=$(( TOTAL + 1 ))

            if [ "$SUBMIT_SLEEP_SECONDS" -gt 0 ]; then
                sleep "$SUBMIT_SLEEP_SECONDS"
            fi
        done
    done
done

echo ""
echo "Submitted ${TOTAL} job(s), skipped ${SKIPPED} completed group(s)."
