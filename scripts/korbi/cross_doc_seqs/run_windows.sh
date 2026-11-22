#!/bin/bash
# A chain of single-window global plans (docs/cross_doc_seqs/README.md), one after another on the same nodes: for
# every start in CDS_WINDOW_STARTS (":"-separated checkpoints) run_pipeline.sh stages global,checkplan,cleanglobal on
# [start, min(start + CDS_WINDOW_STEPS, CDS_RUN_END)), plan name ${CDS_PLAN_PREFIX}<start>, work dir on scratch
# (CDS_GLOBAL_ROOT/<name>, removed after a passed checkplan). A few chains side by side bound the GPFS load and the
# work dirs (~310 GB per 10k-step window). Every node runs the same loop (the stages' file barriers keep them in step);
# a failed window is logged and the chain goes on with the next one; the exit status counts the failed windows.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
: "${CDS_WINDOW_STARTS:?}" "${CDS_RUN_END:?}" "${CDS_GLOBAL_ROOT:?}"
STEPS=${CDS_WINDOW_STEPS:-10000}
PREFIX=${CDS_PLAN_PREFIX:-q3e128_g}
fails=0
for start in ${CDS_WINDOW_STARTS//:/ }; do
    end=$((start + STEPS)); [ "$end" -gt "$CDS_RUN_END" ] && end=$CDS_RUN_END
    name="$PREFIX$start"
    if [ -d "$CDS_OUT/plan/$name/ck$(printf %07d "$start")_$(printf %07d "$end")" ] \
        && grep -q "check_plan $name: 0 failures" "$CDS_OUT/logs/checkplan_$name.log" 2>/dev/null; then
        echo "[windows node ${SLURM_PROCID:-0}] $name: plan exists and passed, skipped"; continue
    fi
    echo "[windows node ${SLURM_PROCID:-0} $(date +%H:%M:%S)] $name: $start -> $end"
    CDS_FROM=$start CDS_STEPS=$((end - start)) CDS_WINDOW=$((end - start)) CDS_PLAN_NAME=$name \
        CDS_GLOBAL_WORK="$CDS_GLOBAL_ROOT/$name" CDS_STAGES=global,checkplan,cleanglobal \
        bash "$D/run_pipeline.sh" || { echo "[windows] $name FAILED"; fails=$((fails + 1)); }
done
echo "[windows node ${SLURM_PROCID:-0} $(date +%H:%M:%S)] chain done: $fails failed window(s)"
exit "$fails"
