#!/bin/bash
# One item of config/eval_tasks/math_cot.yaml for one export, inside a 1-node (4 GPU) allocation.
#   run.sh EXPORT MODE TASKS
#     MODE   sampled  temperature sampling, the release prompt unchanged
#            cot      the same, with the prompt continued by PREFILL (default " <think>\n")
#            nll      per-token NLL of the reference solutions (no generation; one GPU)
#     TASKS  comma-separated: AIME24, AIME25, AMC23, MATH500
# Env: MATH_COT_OUT (default $FLAG_WORK/math_cot), FLAG_WORK (views), SAMPLES (0 = per-task default),
#      TEMPERATURE, TOP_P, PREFILL, EVAL_SIF. Writes $MATH_COT_OUT/<export>/<export>[_cot].<tasks>.shard<i>.jsonl
#      (or .nll.jsonl), then <MODE>.<TASKS>.ok; grade with grade.py inside the same image.
set -eu
EXPORT=$1; MODE=$2; TASKS=$3
HERE=$(cd "$(dirname "$0")" && pwd)
FLAG_WORK=${FLAG_WORK:-/e/scratch/e-sta-openeurollm/$USER/flag-evals}
OUT=${MATH_COT_OUT:-$FLAG_WORK/math_cot}/$EXPORT
VIEW=$FLAG_WORK/views/identity/$EXPORT
SIF=${EVAL_SIF:-/e/project1/e-sta-openeurollm/container/oellm-eval-vllm.sif}
mkdir -p "$OUT/logs"
[ -f "$VIEW/config.json" ] || { echo "error: no identity view $VIEW (run flag-prepare / views first)" >&2; exit 1; }

# Own writable home per process (vLLM/FlashInfer caches): /e/home is not mounted in the image.
run() {
    local gpu=$1; shift
    local home=$OUT/home/${SLURM_JOB_ID:-local}_$gpu; mkdir -p "$home"
    singularity exec --nv --cleanenv --containall --no-mount bind-paths,hostfs,cwd,home --home "$home" \
        --env TMPDIR=/tmp --env HF_HUB_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        --env RAY_USAGE_STATS_ENABLED=0 --env OMP_NUM_THREADS=8 --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        --bind /e/scratch/e-sta-openeurollm:/e/scratch/e-sta-openeurollm --bind "$HERE:$HERE" \
        "$SIF" env CUDA_VISIBLE_DEVICES=$gpu \
        python3 "$HERE/math_sample.py" --view "$VIEW" --name "$EXPORT" --out "$OUT" --tasks "$TASKS" "$@"
}
SAMPLING=(--samples "${SAMPLES:-0}" --temperature "${TEMPERATURE:-0.6}" --top_p "${TOP_P:-0.95}")
case "$MODE" in
    sampled) EXTRA=() ;;
    cot)     EXTRA=(--prefill "${PREFILL:- <think>\n}" --tag _cot) ;;
    nll)     run 0 --nll > "$OUT/logs/nll.$TASKS.log" 2>&1
             touch "$OUT/$MODE.$TASKS.ok"; exit 0 ;;
    *)       echo "error: MODE must be sampled, cot or nll" >&2; exit 2 ;;
esac
pids=()
for i in 0 1 2 3; do
    run $i --shard $i --nshards 4 "${SAMPLING[@]}" "${EXTRA[@]}" > "$OUT/logs/$MODE.$TASKS.shard$i.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
touch "$OUT/$MODE.$TASKS.ok"
