#!/bin/bash
# End-of-window guard for the 84B JUPITER smoke test / prebuilds queued in jupiter_pool_small16
# (docs/recall_surgery_ambitious.md §19.1). The pool keeps max_hosts_running + keep_pending = 2 hosts while any job
# is queued; when host003 / host004 end (~00:29 / ~00:36) a still-queued job of ours would make it SUBMIT a new host
# (user rule: no new Slurm submissions for this work). So at DEADLINE every job named in JOBS is cancelled through
# pool_ctl (a pool job, never a Slurm job; a job that already finished is left as it is -- pool_ctl just reports it).
# Queued jobs that have no run yet do not show in `pool_ctl status`, hence the explicit job names (as enqueued).
# JUPITER login node:
#   JOBS="amb84_jsc_prebuild_p2_3eec12 ..." nohup bash scripts/korbi/recall_surgery/amb84_small16_guard.sh \
#       > /e/fscratch/e-sta-openeurollm/poeppel1/ambitious84/logs/amb84_small16_guard.log 2>&1 &
set -uo pipefail
OUR=${OUR_REPO:-/e/home/jusers/poeppel1/jupiter/work/Projects/oellm-autoexp-hybrid}
PY=${PY:-$OUR/.venv/bin/python}
P=${POOL:-/e/scratch/e-sta-openeurollm/poeppel1/pools/jupiter_pool_small16}
DEADLINE=${DEADLINE:-$(date -d 'today 23:30' +%s)}
JOBS=${JOBS:?job names as enqueued}
log() { echo "$(date '+%F %T') [amb84_guard] $*"; }
ctl() { (cd "$OUR" && PYTHONPATH=. timeout 120 "$PY" scripts/pool_ctl.py "$P" "$@"); }
log "deadline $(date -d "@$DEADLINE" '+%F %T'), jobs: $JOBS"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 300; done
for j in $JOBS; do log "deadline: cancel $j"; ctl cancel "$j" 2>&1 | tail -2; done
log "done"
