#!/bin/bash
# One half (vllm | lighteval) of the FLAG suite INSIDE an existing allocation -- for oellm-autoexp
# stages, where the job cannot sbatch (see config/experiments/korbi/chain_flag_130M_jupiter.yaml).
#
#   flag_stage.sh NAME HALF
#
# Runs in the batch shell of a multi-node allocation (slurm config jupiter_driver: no outer srun):
#   1. builds the model views (locked: the two halves may start together),
#   2. renders this half's launcher (oellm-eval scripts/jupiter_flag_evals.sh, HALVES=HALF),
#   3. runs each eval as an `srun` step on one whole node, as many at once as there are nodes,
#      with the logs where `flag-status` expects them (slurm_logs/oellm-eval-<job>-<index>.*),
#   4. retries the failed evals once, in the same allocation,
#   5. records the run in $FLAG_STATE_DIR/NAME.env; the half that finishes last collects the CSV.
# It exits 0 even if evals failed: the stage must not be restarted as a whole for a few failures.
# `eval_checkpoints.sh flag-status NAME` lists them; `flag-rerun NAME` resubmits them as an array.
#
# Environment: EXPORT_ROOT EVAL_REPO FLAG_WORK FLAG_STATE_DIR FLAG_TIME COLLECT_RAW FLAG_DATASETS_CSV
#   FLAG_INDICES_VLLM / FLAG_INDICES_LIGHTEVAL   comma list of eval indices (default: all)
#   FLAG_RETRIES (default 1)
set -euo pipefail
NAME="${1:?usage: flag_stage.sh NAME vllm|lighteval}" HALF="${2:?vllm|lighteval}"
case "$HALF" in vllm|lighteval) ;; *) echo "error: HALF must be vllm or lighteval" >&2; exit 2 ;; esac
HERE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
source "$HERE/flag_lib.sh"
EXPORT="$EXPORT_ROOT/$NAME"
STATE="$FLAG_STATE_DIR/$NAME.env"
UP=$(tr '[:lower:]' '[:upper:]' <<< "$HALF")
JOB="${SLURM_JOB_ID:?not inside a Slurm allocation}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
[ -f "$EXPORT/config.json" ] || { echo "error: no export $EXPORT" >&2; exit 1; }
mkdir -p "$FLAG_STATE_DIR" "$FLAG_WORK"

rev=$(flag_check_installed)
export FLAG_WORK TIME="$FLAG_TIME" HALVES="$HALF" ACCOUNT="${SLURM_JOB_ACCOUNT:-unused}"
( flock 9; "$EVAL_REPO/scripts/jupiter_flag_evals.sh" views "$EXPORT" > /dev/null ) 9> "$FLAG_WORK/.views.lock"
launcher=$("$EVAL_REPO/scripts/jupiter_flag_evals.sh" render "$NAME" | awk '$1 == "sbatch" {print $2}')
[ -f "$launcher" ] || { echo "error: render produced no $HALF launcher" >&2; exit 1; }
run_dir=$(dirname "$launcher")
n_evals=$(( $(wc -l < "$run_dir/jobs.csv") - 1 ))
indices_var="FLAG_INDICES_$UP"
if [ -n "${!indices_var:-}" ]; then IFS=, read -ra indices <<< "${!indices_var}"; else mapfile -t indices < <(seq 0 $((n_evals - 1))); fi
echo "[flag_stage] $NAME $HALF: ${#indices[@]} of $n_evals evals on $NODES node(s); run $run_dir"

log() { echo "$run_dir/slurm_logs/oellm-eval-$JOB-$1.$2"; }
failed() {  # same rule as flag_status.py
    grep -q "\[error\] Evaluation failed" "$(log "$1" out)" "$(log "$1" err)" 2>/dev/null && return 0
    ! tail -c 200 "$(log "$1" out)" 2>/dev/null | grep -q "finished\.*[[:space:]]*$"
}
run_round() {  # <index>... : one srun step per eval, at most NODES at a time
    # Same step options as templates/jupiter.sbatch's training steps, which see all 4 GPUs.
    local i
    for i in "$@"; do
        while [ "$(jobs -rp | wc -l)" -ge "$NODES" ]; do wait -n || true; done
        SLURM_ARRAY_TASK_ID=$i SLURM_ARRAY_JOB_ID=$JOB \
            srun --nodes=1 --ntasks=1 --exclusive --cpus-per-task="${SLURM_CPUS_ON_NODE:-288}" \
                 --job-name="flag-$HALF-$i" \
                 bash "$launcher" > "$(log "$i" out)" 2> "$(log "$i" err)" &
    done
    wait || true
}

mkdir -p "$run_dir/slurm_logs"
todo=("${indices[@]}")
for attempt in $(seq 0 "${FLAG_RETRIES:-1}"); do
    [ ${#todo[@]} -gt 0 ] || break
    if [ "$attempt" -gt 0 ]; then
        echo "[flag_stage] retry $attempt: ${todo[*]}"
        for i in "${todo[@]}"; do for s in out err; do mv -f "$(log "$i" $s)" "$(log "$i" $s).attempt$attempt" 2>/dev/null || true; done; done
    fi
    run_round "${todo[@]}"
    next=(); for i in "${todo[@]}"; do failed "$i" && next+=("$i"); done
    todo=("${next[@]+"${next[@]}"}")
done
echo "[flag_stage] $HALF done: $(( ${#indices[@]} - ${#todo[@]} ))/${#indices[@]} ok${todo:+; failed: ${todo[*]}}"

# Record this half; the second half to arrive collects (the state file is shared, so lock it).
(
    flock 9
    { grep -q "^NAME=" "$STATE" 2>/dev/null || printf 'NAME=%s\nEXPORT=%s\nEVAL_REV=%s\n' "$NAME" "$EXPORT" "$rev"
      printf '%s_RUN=%s\n%s_JOB=%s\n%s_DONE=%s\n' "$UP" "$run_dir" "$UP" "$JOB" "$UP" "$(date -Is)"; } >> "$STATE"
    if grep -q "^VLLM_DONE=" "$STATE" && grep -q "^LIGHTEVAL_DONE=" "$STATE"; then
        subset=$([ -n "${FLAG_INDICES_VLLM:-}${FLAG_INDICES_LIGHTEVAL:-}" ] && echo --collect || true)
        FLAG_NO_QUEUE=1 python3 "$HERE/flag_status.py" "$STATE" $subset || echo "[flag_stage] collection failed (see above)"
    fi
) 9> "$STATE.lock"
exit 0
