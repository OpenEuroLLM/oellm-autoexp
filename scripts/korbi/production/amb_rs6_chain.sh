#!/bin/bash
# amb_rs6: 4-node row-state resume validation on the rowmeta snapshot.
#
# Fix vs amb_rs5: train_iters 30 (not 40), so ref's step-20 rolling checkpoint is not
# overwritten by a step-40 non-persistent save; cont must load ref/ckpt_rolling iter 20.
# (amb_rs5's cont loaded iter 40 and the state comparison was invalid.)
#
# The 4-node job runs on a large pool host's free nodes, not on the dev queue. Default
# o134e: it had 133 free nodes (only a 1-node keepalive); the cns host004 has only 3
# usable spare nodes after as04r4b28 was excluded, so a 4-node job does not fit there.
#
#   nohup bash scripts/korbi/production/amb_rs6_chain.sh \
#       > $R/monitor_logs/amb_rs6_chain.log 2>&1 < /dev/null &
#   POOL=p129 SNAP=autoexp_rowmeta3 SUITE=amb_rs7 ...   (override the pool / snapshot)
set -u

R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
SNAP=${SNAP:-autoexp_rowmeta3}
POOL=${POOL:-o134e}
SUITE=${SUITE:-amb_rs6}
B=/gpfs/scratch/ehpc533/hybrid_nucleus/resume_tests/$SUITE
L=$R/monitor_logs
PD=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_$POOL

cd "$R/$SNAP" || { echo "[$SUITE] no snapshot $R/$SNAP"; exit 1; }
mkdir -p "$B" "$L"
echo "[$SUITE] $(date '+%F %T') snapshot=$SNAP pool=$POOL"

enqueue() {  # phase arm
    scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py \
        --no-monitor --monitor-state-dir "$R/pool_state" --pool "$PD" \
        --config-name experiments/korbi/production/hybrid_ambitious_resume_test_mn5_pool \
        --yes aux.phase="$1" aux.suite=$SUITE "aux.only=[$2]" \
        backend.megatron.exit_interval=null
}

wait_run() {  # name
    local name=$1 i rd rc
    for i in $(seq 1 240); do
        rd=$(grep -l "\"name\": \"$name\"" "$PD"/runs/*/run.json 2>/dev/null | head -1 | xargs -r dirname)
        if [ -n "$rd" ] && [ -e "$rd/rc" ]; then
            rc=$(cat "$rd/rc")
            echo "[$SUITE] $(date '+%F %T') $name $rd rc=$rc"
            case "$rc" in
                "0 completed"*) return 0 ;;
                *) echo "[$SUITE] $name ended: $rc"; return 1 ;;
            esac
        fi
        sleep 15
    done
    echo "[$SUITE] $name: timeout"
    return 1
}

echo "[$SUITE] $(date '+%F %T') enqueue ref"
enqueue save ref || exit 1
wait_run ${SUITE}_ref || exit 1

iter=$(cat "$B/ref/ckpt_rolling/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
echo "[$SUITE] $(date '+%F %T') ref rolling iter=$iter"
[ "$iter" = "20" ] || { echo "[$SUITE] ERROR: ref rolling is '$iter', expected 20"; exit 1; }

echo "[$SUITE] $(date '+%F %T') enqueue cont"
enqueue resume cont || exit 1
wait_run ${SUITE}_cont || exit 1

f=$(ls -t "$B/cont"/slurm-*.log 2>/dev/null | head -1)
echo "[$SUITE] cont log: $f"
grep -aE "Loading from a non-persistent|successfully loaded" "$f" 2>/dev/null | grep -a default0 | head -2
grep -aq "non-persistent iter 20" "$f" 2>/dev/null || {
    echo "[$SUITE] ERROR: cont did not load iter 20"
    exit 1
}

echo "[$SUITE] $(date '+%F %T') compare"
"$R/autoexp_env/bin/python" scripts/korbi/production/compare_resume_test.py "$B" 2>&1 \
    | tee "$L/${SUITE}_compare.log" | grep -vE "^      scalar" | tail -80
echo "[$SUITE] $(date '+%F %T') CHAIN_DONE"
