#!/bin/bash
# Off-cluster watchdog of the MN5 production pool monitors (user 2026-10-28), for a machine with ssh
# access to MareNostrum 5: every INTERVAL seconds one pass of pool_monitor_watchdog.sh on MN5 (restart
# a pool monitor that is dead: heartbeat stale and no process), and the MN5-side watchdog loop
# restarted if it is gone. Survives the login node killing processes (the p129 monitor, 2026-10-28).
#   nohup setsid scripts/korbi/pool/pool_monitor_watchdog_remote.sh >> ~/pool_watchdog_mn5.log 2>&1 &
#   stop it with: kill "$(cat ~/pool_watchdog_mn5.pid)"   (never pkill -f: it matches the calling shell)
set -u
echo $$ > "${PIDFILE:-$HOME/pool_watchdog_mn5.pid}"
HOST="${HOST:-marenostrum_new}"
INTERVAL="${INTERVAL:-300}"
W=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/monitor_logs/pool_monitor_watchdog.sh
log() { echo "$(date '+%F %T') $*"; }
while true; do
  out=$(timeout 240 ssh -o BatchMode=yes -o ConnectTimeout=30 "$HOST" "
    ONCE=1 $W 2>&1
    for c in /proc/[0-9]*/cmdline; do
      tr '\\0' ' ' < \$c 2>/dev/null | grep -q '^/bin/bash $W' && { echo WATCHDOG_ALIVE; break; }
    done
  " 2>&1 | grep -v '^load \|^CPLUS\|CUDA_HOME\|CUDNN_LIB')
  rc=$?
  if [ -z "$out" ] && [ "$rc" -ne 0 ]; then
    log "ssh to $HOST failed"
  else
    printf '%s\n' "$out" | grep -v '^WATCHDOG_ALIVE$' | sed "s/^/$(date '+%F %T') mn5: /" | grep -v ': $'
    if ! printf '%s\n' "$out" | grep -q '^WATCHDOG_ALIVE$'; then
      log "MN5-side watchdog not running: restarting it"
      timeout 120 ssh -o BatchMode=yes "$HOST" "setsid nohup $W >> ${W%.sh}.log 2>&1 < /dev/null &" >/dev/null 2>&1
    fi
  fi
  sleep "$INTERVAL"
done
