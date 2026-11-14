#!/bin/bash
# Guard of a small31 base run on the 32B pool's buffer nodes (docs/recall_surgery_ambitious.md §14; user 2026-11-14:
# use the 4 spare nodes of the 32B run's 516-node host, packed next to the 512-node 32B run). Runs on the JUPITER login
# node (nohup), every POLL_S seconds, and gives the nodes back at once when the 32B run might need them:
#   - no 32B run (NAME oellm_32b_dense_b1*) RUNNING in the pool, or one PENDING (restart / requeue / handover), or
#   - the guarded run's host reports fewer usable nodes than HOST_NODES (a node lost), or
#   - the guarded run's host is no longer busy with the 32B run (handover / release)
# then: `pool_ctl cancel` the guarded run (graceful: SIGUSR2 -> Megatron saves and exits), restore the pool's
# max_runs_per_host (OLD_MAX_RUNS), and re-enqueue the run in the small16 pool (resume from its own checkpoint dir).
# The same restore + re-enqueue when the guarded run ends on its own. Never touches the 32B run or any Slurm job.
#   GUARD_JOB=<job name in the 32B pool> ARM=b7 [GUARD_NAME=small31_b7] nohup bash buffer_guard.sh > <log> 2>&1 &
set -uo pipefail
P=${POOL:-/e/scratch/e-ext-2025e02-108/oellm_32b_dense_b1/pools/oellm32b_b1_relaxed_20261103}
CTL_REPO=${CTL_REPO:-/e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp}  # the 32B pool's own code
OUR=${OUR_REPO:-/e/home/jusers/poeppel1/jupiter/work/Projects/oellm-autoexp-hybrid}
PY=${PY:-/e/home/jusers/poeppel1/jupiter/work/Projects/oellm-autoexp-hybrid/.venv/bin/python}
SMALL16=${SMALL16:-/e/scratch/e-sta-openeurollm/poeppel1/pools/jupiter_pool_small16}
STATE=${STATE:-/e/scratch/e-sta-openeurollm/poeppel1/pool_state}
: "${GUARD_JOB:?job name of the guarded run in the 32B pool}" "${ARM:?b7 | r3 | r3p}"
GUARD_NAME=${GUARD_NAME:-small31_$ARM}  # the guarded run's NAME (meta.env) prefix
OLD_MAX_RUNS=${OLD_MAX_RUNS:-0}
HOST_NODES=${HOST_NODES:-516}
POLL_S=${POLL_S:-30}
log() { echo "$(date '+%F %T') [buffer_guard] $*"; }
ctl() { (cd "$CTL_REPO" && PYTHONPATH=. timeout 120 "$PY" scripts/pool_ctl.py "$P" "$@"); }
# runs of the pool that have not ended: "<rid> <name> <host> <status>"
runs() {
  for rd in "$P"/runs/*; do
    [ -f "$rd/rc" ] && continue
    [ -f "$rd/meta.env" ] || continue
    name=$(grep -h '^NAME=' "$rd/meta.env" | tail -1 | cut -d= -f2 | tr -d "'\"")
    host=$(grep -h '^host=' "$rd/started" 2>/dev/null | cut -d= -f2)
    st=PENDING; [ -n "$host" ] && st=RUNNING
    echo "$(basename "$rd") $name ${host:--} $st"
  done
}
give_back() {
  log "GIVE BACK: $1"
  ctl cancel "$GUARD_JOB" --wait 600 2>&1 | tail -2
  leave "after give-back"
}
# a guarded run that ended non-zero: stop at once (the pool would retry it; the cause needs a look first)
ended_rc() {
  for rd in "$P"/runs/*; do
    [ -f "$rd/rc" ] && grep -qh "^NAME=$GUARD_NAME" "$rd/meta.env" 2>/dev/null && [ "$(stat -c %Y "$rd/rc")" -ge "$T0" ] \
      && ! grep -qE "^0 " "$rd/rc" && { echo "$(basename "$rd"): $(cat "$rd/rc")"; return; }
  done
}
leave() {
  # the guarded job out of the pool first: a run that ended non-zero is retried by the pool (max_attempts), and a
  # queued retry must not wait there for the 32B run's nodes (cancel is idempotent)
  ctl cancel "$GUARD_JOB" --wait 300 2>&1 | tail -1
  ctl set max_runs_per_host="$OLD_MAX_RUNS" 2>&1 | tail -1
  log "max_runs_per_host restored to $OLD_MAX_RUNS ($1)"
  if [ "${REENQUEUE:-1}" = 1 ]; then
    (cd "$OUR" && PYTHONPATH=. timeout 600 "$PY" scripts/run_autoexp.py --no-monitor --monitor-state-dir "$STATE" \
      --pool "$SMALL16" --config-name experiments/korbi/recall_surgery/small31_jupiter --yes "aux.only=[$ARM]" 2>&1 \
      | grep -E "^Enqueued|rror" | cut -c1-200)
    log "$ARM re-enqueued in $SMALL16 (resumes from its checkpoint)"
  fi
  exit 0
}
T0=$(date +%s)
log "guarding $GUARD_JOB ($ARM) on $P; old max_runs_per_host $OLD_MAX_RUNS"
seen=0
while true; do
  bad=$(ended_rc); [ -n "$bad" ] && { REENQUEUE=0; leave "the guarded run failed ($bad); not re-enqueued"; }
  R=$(runs)
  mine=$(echo "$R" | awk -v n="$GUARD_NAME" 'index($2, n) == 1')
  big=$(echo "$R" | awk '$2 ~ /^oellm_32b_dense_b1/')
  if [ -z "$mine" ]; then
    if [ $seen = 1 ]; then leave "the guarded run ended"; fi
    # not yet visible (enqueue in flight): keep checking, but give the setting back if it never appears
    waited=$((${waited:-0} + POLL_S)); [ "$waited" -gt 1800 ] && leave "the guarded run never appeared"
  else
    seen=1
  fi
  if [ -z "$(echo "$big" | awk '$4 == "RUNNING"')" ]; then give_back "no 32B run RUNNING: ${big:-none}"; fi
  if [ -n "$(echo "$big" | awk '$4 == "PENDING"')" ]; then give_back "a 32B run is PENDING: $big"; fi
  myhost=$(echo "$mine" | awk '{print $3}' | head -1)
  if [ -n "$myhost" ] && [ "$myhost" != "-" ]; then
    hs=$(cat "$P/hosts/$myhost/state" 2>/dev/null)
    usable=$(echo "$hs" | awk '{print $NF}')
    bighost=$(echo "$big" | awk '$4 == "RUNNING" {print $3}' | head -1)
    [ "$bighost" != "$myhost" ] && give_back "the 32B run is on $bighost, ours on $myhost"
    [ -n "$usable" ] && [ "$usable" -lt "$HOST_NODES" ] 2>/dev/null && give_back "$myhost usable $usable < $HOST_NODES"
  fi
  sleep "$POLL_S"
done
