#!/bin/bash
# Follow the v4 continuation LINE, not one run id.
#
# Why: run ids change on every pool restart (a node/IB failure, a GPFS stall, a crash). A follower pinned
# to one rid keeps printing the last iteration of a dead log and cannot say "the line continues as .NNNN",
# so it reports healthy while nothing trains. This resolves the live run of the line by name each cycle.
#
#   bash scripts/korbi/recall_surgery/follow_v4_cont_line.sh [name] [interval_s] [max_s]
#
# Alerts (exit non-zero, so the notification wakes the session) on: an error pattern in the live log, a
# stalled iteration count, the line having no live run, or a restart (reported, then followed).
set -uo pipefail

POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
RUN=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/runs/main
NAME=${1:-ambitious_v4_cont_main_continue}
INTERVAL=${2:-300}
MAX=${3:-21600}
T0=$(date +%s)
CUR_RID=""
LAST_IT=""
LAST_PROGRESS=$T0
SEEN_SAVES=0

live_rid() {
    python3 - "$POOL" "$NAME" <<'PYLIVE'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
best = 0
for d in glob.glob(os.path.join(pool, "runs", "mn5-pool-p264.*")):
    try:
        if json.load(open(os.path.join(d, "run.json"))).get("name") != name:
            continue
    except Exception:
        continue
    if os.path.exists(os.path.join(d, "rc")):
        continue
    best = max(best, int(d.rsplit(".", 1)[-1]))
print(best or "")
PYLIVE
}

echo "follow line '$NAME'  (interval ${INTERVAL}s, budget $((MAX/3600))h)"
while :; do
    NOW=$(date +%s); EL=$((NOW - T0))
    RID=$(live_rid)
    if [ -z "$RID" ]; then
        echo "[$((EL/60))m] ALERT: no live run of '$NAME' in the pool (all ended) -- nothing is training"
        exit 3
    fi
    if [ "$RID" != "$CUR_RID" ]; then
        [ -n "$CUR_RID" ] && echo "[$((EL/60))m] RESTART: line moved .$CUR_RID -> .$RID (previous run ended)"
        CUR_RID=$RID
        SEEN_SAVES=0
        LAST_IT=""
        LAST_PROGRESS=$NOW
        echo "[$((EL/60))m] following .$RID"
    fi
    D=$POOL/runs/mn5-pool-p264.$RID
    LOG=$RUN/slurm-mn5-pool-p264.$RID.log
    if [ ! -s "$LOG" ]; then
        echo "[$((EL/60))m] .$RID: no log yet (starting)"; sleep "$INTERVAL"; continue
    fi
    IT=$(grep -oE "iteration +[0-9]+/" "$LOG" | tail -1 | grep -oE "[0-9]+")
    LAST=$(grep -E "iteration +[0-9]+/" "$LOG" | tail -1)
    STEP=$(printf '%s' "$LAST" | grep -oE "elapsed time per iteration \(ms\): [0-9.]+" | grep -oE "[0-9.]+$")
    MEM=$(printf '%s' "$LAST" | grep -oE "mem usages: [0-9.]+" | grep -oE "[0-9.]+$")
    LOSS=$(printf '%s' "$LAST" | grep -oE "lm loss: [0-9.E+-]+" | awk '{print $3}')
    HARD=$(grep -cE "oom_kill|Out Of Memory|Triton Error|RuntimeError|Traceback" "$LOG" 2>/dev/null)
    # Calibrated on .1445: every start emits one benign "NCCL WARN Call to ibv_query_port_speed failed"
    # per GPU (8192 lines), so a generic "NCCL WARN" count alerts on every launch. Only the NET/IB
    # transport errors mark a real fabric fault (10 lines in .1445, immediately before the kill).
    NCCL=$(grep -cE "NCCL WARN NET/IB|transport/net_ib/p2p\.cc" "$LOG" 2>/dev/null)
    NCCL_INFO=$(grep -c "ibv_query_port_speed" "$LOG" 2>/dev/null)
    SAVES=$(grep -cE "successfully saved checkpoint" "$LOG" 2>/dev/null)
    VAL=$(grep -cE "validation" "$LOG" 2>/dev/null)
    NEW=$((SAVES - SEEN_SAVES))
    echo "[$((EL/60))m] .$RID it=$IT/236733 step=${STEP:-?}ms mem=${MEM:-?} loss=${LOSS:-?} saves=$SAVES(+$NEW) val=$VAL hard_err=$HARD nccl_warn=$NCCL ibv_info=$NCCL_INFO"

    if [ -n "$IT" ] && [ "$IT" != "$LAST_IT" ]; then LAST_IT=$IT; LAST_PROGRESS=$NOW; fi
    if [ "$NEW" -gt 0 ]; then
        grep -E "successfully saved checkpoint" "$LOG" | tail -"$NEW" | sed 's/.*saved checkpoint/    saved/' | cut -c1-105
        SEEN_SAVES=$SAVES
    fi
    if [ "$HARD" -gt 0 ]; then
        echo "ALERT: hard error in .$RID"
        grep -nE "oom_kill|Out Of Memory|Triton Error|RuntimeError|Traceback" "$LOG" | tail -4 | cut -c1-150
        exit 4
    fi
    if [ "$NCCL" -gt 0 ]; then
        echo "ALERT: $NCCL NET/IB transport error line(s) in .$RID (fabric fault; the pool restarts on this)"
        grep -E "NCCL WARN NET/IB|transport/net_ib/p2p\.cc" "$LOG" | tail -2 | cut -c1-140
        exit 5
    fi
    if [ -n "$IT" ] && [ $((NOW - LAST_PROGRESS)) -gt 1800 ]; then
        echo "ALERT: .$RID stalled at iteration $IT for $(((NOW-LAST_PROGRESS)/60)) min"
        tail -4 "$LOG" | cut -c1-140
        exit 6
    fi
    if [ -e "$D/rc" ]; then
        echo "ALERT: .$RID ended rc=$(cat "$D/rc") at iteration $IT (pool should restart the line)"
        tail -4 "$LOG" | cut -c1-140
        exit 7
    fi
    if [ "$EL" -ge "$MAX" ]; then echo "budget spent at .$RID it=$IT (still running)"; exit 0; fi
    sleep "$INTERVAL"
done
