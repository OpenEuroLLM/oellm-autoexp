#!/bin/bash
# Follow the v4 line across its 36900 handover: the experiment segment (.1314) must save 36900, exit, and
# the open-ended continuation (ambitious_v4_cont_main_continue) must then be submitted by the pool and
# start from runs/main/ckpt with mbs 1 and no --exit-interval.
#
#   bash scripts/korbi/recall_surgery/watch_handover_36900.sh [interval_s] [max_s]
#
# Read-only: it inspects pool files and logs, and never touches a Slurm job. Exits when the continuation
# is running, or when the wait budget runs out (then it says which half is missing).
set -uo pipefail

POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
STATE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/pool_state/pool_mn5-pool-p264
RUN=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/runs/main
LOG=$RUN/slurm-mn5-pool-p264.1314.log
SEG=$POOL/runs/mn5-pool-p264.1314
NAME=ambitious_v4_cont_main_continue
INTERVAL=${1:-120}
MAX=${2:-21600}   # 6 h
T0=$(date +%s)

iter() { grep -oE "iteration +[0-9]+/" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
tracker() { tr -d ' \n' < "$RUN/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null; }

echo "handover watch: segment .1314 -> $NAME  (interval ${INTERVAL}s, budget $((MAX/3600))h)"

while :; do
    NOW=$(date +%s); EL=$((NOW - T0))
    I=$(iter); T=$(tracker)
    SEGRC=$(cat "$SEG/rc" 2>/dev/null || echo ACTIVE)
    # one python process for the whole scan: the pool holds ~1400 run dirs, so a process per dir would
    # take minutes per iteration
    NEW=$(python3 - "$POOL" "$NAME" <<'PYNEW'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
best = 0
for d in glob.glob(os.path.join(pool, "runs", "mn5-pool-p264.*")):
    try:
        if json.load(open(os.path.join(d, "run.json"))).get("name") == name:
            best = max(best, int(d.rsplit(".", 1)[-1]))
    except Exception:
        continue
print(best or "")
PYNEW
)
    NEW=${NEW#mn5-pool-p264.}

    if [ -n "$NEW" ]; then
        ND=$POOL/runs/mn5-pool-p264.$NEW
        GEO=$(grep -oE "\-\-(micro-batch-size|global-batch-size|exit-interval) [0-9]+" "$ND/step.sh" 2>/dev/null | sort -u | tr '\n' ' ')
        echo "[$((EL/60))m] HANDOVER: continuation run .$NEW  tracker=$T  segment rc=$SEGRC"
        echo "      geometry: ${GEO:-<step.sh not written yet>}"
        if grep -q -- "--exit-interval" "$ND/step.sh" 2>/dev/null; then
            echo "      BAD: the continuation has an exit-interval"; exit 4
        fi
        if [ -e "$ND/rc" ]; then
            echo "      BAD: the continuation already ended: rc=$(cat "$ND/rc")"; exit 5
        fi
        echo "      OK: continuation is the live run of this line"
        exit 0
    fi

    # a stop marker or a non-zero rc on the segment with no successor means the handover is not happening
    if [ -e "$SEG/stop" ] && [ "$SEGRC" != ACTIVE ]; then
        echo "[$((EL/60))m] segment stopped (rc=$SEGRC, stop marker present) and no continuation run yet"
    fi
    if [ "$SEGRC" != ACTIVE ] && [ "$SEGRC" != 0 ]; then
        echo "[$((EL/60))m] segment ended rc=$SEGRC  tracker=$T  iteration=$I  (no continuation run yet)"
    else
        echo "[$((EL/60))m] segment $SEGRC  iteration=$I/236733  tracker=$T"
    fi

    if [ "$EL" -ge "$MAX" ]; then
        echo "budget spent after $((EL/60))m: segment=$SEGRC tracker=$T continuation=$([ -n "$NEW" ] && echo yes || echo no)"
        exit 2
    fi
    sleep "$INTERVAL"
done
