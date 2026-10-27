#!/bin/bash
# The long-context set (ds5lcresgqa_<size> + logn) on each GQA-rematch arm's FINAL checkpoint, as
# soon as that arm has it (latest_checkpointed_iteration 4000), enqueued into a pool (user
# 2026-10-27: small jobs go into the 33-node allocation, packed side by side). One enqueue per arm
# and eval (aux.only, no array group). Run detached on MN5 from the snapshot the runs execute in:
#   setsid nohup scripts/korbi/campaign/resgqa_pool_eval_chain.sh <pool dir> <snapshot> <size>:<arm>... \
#       > $R/monitor_logs/resgqa_pool_eval_chain.log 2>&1 &
# Other suites: TRAIN=screen5ngptldm EVAL=ds5lcngptldm (training suite / eval suite prefixes).
set -u
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
POOL=${1:?pool dir}
SNAP=${2:?snapshot dir}
shift 2
SPECS=("$@")
C=/gpfs/scratch/ehpc533/hybrid_nucleus/campaign_3b
TRAIN=${TRAIN:-screen5resgqa}
EVAL=${EVAL:-ds5lcresgqa}
log() { echo "$(date '+%F %T') $*"; }
declare -A done_
left=${#SPECS[@]}
log "long-context evals into $(basename "$POOL") from $SNAP: ${SPECS[*]}"
while [ "$left" -gt 0 ]; do
    for spec in "${SPECS[@]}"; do
        [ -n "${done_[$spec]:-}" ] && continue
        size=${spec%%:*}
        a=${spec#*:}
        it=$(cat $C/${TRAIN}_$size/$a/ckpt/latest_checkpointed_iteration.txt 2>/dev/null)
        [ "${it:-0}" = 4000 ] || continue
        cd "$SNAP"
        for s in ${EVAL}_$size ${EVAL}logn_$size; do
            scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --no-monitor \
                --monitor-state-dir $R/pool_state --pool "$POOL" \
                --config-name "experiments/korbi/campaign_3b/${s}_mn5" "aux.only=[$a]" \
                job.array_group=null --yes 2>&1 | grep -E "Enqueued|rror" | tail -2
            log "enqueued $s on $a (checkpoint $it)"
        done
        done_[$spec]=1
        left=$((left - 1))
    done
    [ "$left" -gt 0 ] && sleep 300
done
log "all arms enqueued"
