#!/bin/bash
# End-of-window guard for the 84B JUPITER smoke test / prebuilds queued in jupiter_pool_small16
# (docs/recall_surgery_ambitious.md §19.1). The pool keeps max_hosts_running + keep_pending = 2 hosts while any run
# is queued; when host003 / host004 end (~00:29 / ~00:36) a still-queued run of ours would make it SUBMIT a new host
# (user rule: no new Slurm submissions for this work). So at DEADLINE every amb84_jsc_* run still in the pool is
# cancelled (pool_ctl cancel: pool runs only, never a Slurm job). Exits early once none of ours is queued / running.
# JUPITER login node:
#   nohup bash scripts/korbi/recall_surgery/amb84_small16_guard.sh \
#       > /e/fscratch/e-sta-openeurollm/poeppel1/ambitious84/logs/amb84_small16_guard.log 2>&1 &
set -uo pipefail
OUR=${OUR_REPO:-/e/home/jusers/poeppel1/jupiter/work/Projects/oellm-autoexp-hybrid}
PY=${PY:-$OUR/.venv/bin/python}
P=${POOL:-/e/scratch/e-sta-openeurollm/poeppel1/pools/jupiter_pool_small16}
DEADLINE=${DEADLINE:-$(date -d 'today 23:30' +%s)}
NAMES=${NAMES:-amb84_jsc_prebuild_p2 amb84_jsc_prebuild_p1 amb84_jsc_smoke_a amb84_jsc_smoke_b}
POLL_S=${POLL_S:-300}
log() { echo "$(date '+%F %T') [amb84_guard] $*"; }
ctl() { (cd "$OUR" && PYTHONPATH=. timeout 120 "$PY" scripts/pool_ctl.py "$P" "$@"); }
ours_active() {  # run ids of ours the pool lists in a non-final state (queued / waiting / running)
  local s; s=$(ctl status 2>/dev/null) || { echo "?"; return 0; }
  for n in $NAMES; do
    echo "$s" | awk -v n="$n" '$1 ~ /^jupiter-pool-small16\./ && index($3, n) == 1 \
      && $2 !~ /^(CANCELLED|COMPLETED|FAILED|PREEMPTED|FINISHED)$/ {print $1}'
  done
}
log "deadline $(date -d "@$DEADLINE" '+%F %T'), names: $NAMES"
while :; do
  active=$(ours_active)
  if [ -z "$active" ]; then log "none of ours left in the pool -- done"; exit 0; fi
  if [ "$active" = "?" ]; then log "pool_ctl status failed; retry"; sleep 60; continue; fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    for r in $active; do log "deadline: cancelling run $r"; ctl cancel "$r" || log "cancel $r failed"; done
    exit 0
  fi
  sleep "$POLL_S"
done
