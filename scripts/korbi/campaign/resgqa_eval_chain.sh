#!/bin/bash
# The long-context set (ds5lcresgqa_<size> + logn) on each GQA-rematch arm's FINAL checkpoint, as
# soon as that arm has it (latest_checkpointed_iteration 4000). One launch per arm and eval, each in
# its own array group (a staggered aux.only launch into one group overwrites an earlier arm's task
# script). Run detached on MN5 from the snapshot the suite was launched from:
#   setsid nohup scripts/korbi/campaign/resgqa_eval_chain.sh <snapshot> <size> <arm>... \
#       > $R/monitor_logs/resgqa_<size>_eval_chain.log 2>&1 &
set -u
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
SNAP=${1:?snapshot dir}
SIZE=${2:?size}
shift 2
ARMS=("$@")
D=/gpfs/scratch/ehpc533/hybrid_nucleus/campaign_3b/screen5resgqa_$SIZE
log() { echo "$(date '+%F %T') $*"; }
declare -A done_
left=${#ARMS[@]}
log "per-arm long-context evals on the $SIZE step-4000 checkpoints: ${ARMS[*]}"
while [ "$left" -gt 0 ]; do
    for a in "${ARMS[@]}"; do
        [ -n "${done_[$a]:-}" ] && continue
        it=$(cat $D/$a/ckpt/latest_checkpointed_iteration.txt 2>/dev/null)
        [ "${it:-0}" = 4000 ] || continue
        cd "$SNAP"
        for s in ds5lcresgqa_$SIZE ds5lcresgqalogn_$SIZE; do
            scripts/korbi/mn5_autoexp_bg.sh "${s}_$a" scripts/run_autoexp.py \
                --config-name "experiments/korbi/campaign_3b/${s}_mn5" "aux.only=[$a]" \
                "job.array_group=${s}_$a" --yes
            log "launched $s on $a (checkpoint $it)"
        done
        done_[$a]=1
        left=$((left - 1))
    done
    [ "$left" -gt 0 ] && sleep 300
done
log "all arms launched"
