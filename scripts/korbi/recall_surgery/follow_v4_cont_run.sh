#!/bin/bash
# Follow the open-ended v4 continuation run (no exit-interval: it trains to train_iters 236733 or a
# manual stop). Reports iteration, s/step, memory, loss, saves, validations and errors; exits early on an
# error pattern or when the run ends. Read-only: it never touches a Slurm job.
#
#   bash scripts/korbi/recall_surgery/follow_v4_cont_run.sh [run_id] [interval_s] [max_s]
set -uo pipefail

POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
RUN=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/runs/main
RID=${1:-1445}
INTERVAL=${2:-900}
MAX=${3:-43200}          # 12 h
D=$POOL/runs/mn5-pool-p264.$RID
LOG=$RUN/slurm-mn5-pool-p264.$RID.log
T0=$(date +%s)
SEEN_SAVES=0

echo "follow .$RID  (interval ${INTERVAL}s, budget $((MAX/3600))h)  log=$LOG"
while :; do
    NOW=$(date +%s); EL=$((NOW - T0))
    if [ ! -e "$D/rc" ] && [ ! -s "$LOG" ]; then
        echo "[$((EL/60))m] no log yet"; sleep "$INTERVAL"; continue
    fi
    IT=$(grep -oE "iteration +[0-9]+/" "$LOG" | tail -1 | grep -oE "[0-9]+")
    LAST=$(grep -E "iteration +[0-9]+/" "$LOG" | tail -1)
    STEP=$(printf '%s' "$LAST" | grep -oE "elapsed time per iteration \(ms\): [0-9.]+" | grep -oE "[0-9.]+$")
    MEM=$(printf '%s' "$LAST" | grep -oE "mem usages: [0-9.]+" | grep -oE "[0-9.]+$")
    LOSS=$(printf '%s' "$LAST" | grep -oE "lm loss: [0-9.E+-]+" | awk '{print $3}')
    ERR=$(grep -cE "oom_kill|Out Of Memory|Triton Error|RuntimeError|Traceback" "$LOG" 2>/dev/null)
    VAL=$(grep -cE "validation" "$LOG" 2>/dev/null)
    SAVES=$(grep -cE "successfully saved checkpoint" "$LOG" 2>/dev/null)
    RC=$(cat "$D/rc" 2>/dev/null || echo ACTIVE)
    NEW_SAVES=$((SAVES - SEEN_SAVES))
    echo "[$((EL/60))m] it=$IT/236733 step=${STEP:-?}ms mem=${MEM:-?} loss=${LOSS:-?} saves=$SAVES(+$NEW_SAVES) validations=$VAL errors=$ERR rc=$RC"
    if [ "$NEW_SAVES" -gt 0 ]; then
        grep -E "successfully saved checkpoint" "$LOG" | tail -"$NEW_SAVES" |
            sed 's/.*saved checkpoint/    saved/' | cut -c1-110
        SEEN_SAVES=$SAVES
    fi
    if [ "$ERR" -gt 0 ]; then
        echo "=== error lines"
        grep -nE "oom_kill|Out Of Memory|Triton Error|RuntimeError|Traceback" "$LOG" | tail -5 | cut -c1-160
        exit 4
    fi
    if [ "$RC" != ACTIVE ]; then
        echo "=== run .$RID ended rc=$RC at iteration $IT"
        tail -5 "$LOG" | cut -c1-150
        exit 0
    fi
    if [ "$EL" -ge "$MAX" ]; then
        echo "budget spent at it=$IT (still running)"; exit 2
    fi
    sleep "$INTERVAL"
done
