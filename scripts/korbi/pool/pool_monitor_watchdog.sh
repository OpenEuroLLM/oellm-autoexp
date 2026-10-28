#!/bin/bash
# Restart a production pool monitor whose heartbeat (<pool>/monitor.alive) is older than STALE_S
# while no monitor process for its session runs (2026-10-28: the p129 monitor was killed without a
# trace at 16:23). Processes are matched by their exact command line in /proc, never pgrep -f (it
# matches this script's own shell).
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
POOLS=/gpfs/scratch/ehpc533/hybrid_nucleus/pools
SNAP=${SNAP:-$R/autoexp_p132}
STALE_S=${STALE_S:-900}
log() { echo "$(date '+%F %T') $*"; }
running() { # running <session dir>: a monitor_autoexp.py process for it
  for c in /proc/[0-9]*/cmdline; do
    [ "${c#/proc/}" = "$$/cmdline" ] && continue
    tr '\0' ' ' <"$c" 2>/dev/null | grep -q "scripts/monitor_autoexp.py --session-dir $1 " && return 0
  done
  return 1
}
while true; do
  for name in p129 p132; do
    alive=$POOLS/mn5_pool_$name/monitor.alive
    session=$R/pool_state/pool_mn5-pool-$name
    age=$(( $(date +%s) - $(stat -c %Y "$alive" 2>/dev/null || echo 0) ))
    if [ "$age" -gt "$STALE_S" ] && ! running "$session"; then
      log "pool $name: heartbeat ${age}s old and no monitor: restarting from $SNAP"
      (cd "$SNAP" && PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus \
        scripts/korbi/mn5_autoexp_bg.sh "pool_${name}_mon" scripts/monitor_autoexp.py \
        --session-dir "$session" --poll-interval 30 --force)
    fi
  done
  sleep 300
done
