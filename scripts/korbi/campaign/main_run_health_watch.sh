#!/bin/bash
# Health watch of the ambitious main run (user 2026-11-06: "run a monitor for this checking regularly").
# Read-only: every INTERVAL seconds one ssh to MN5 reads the main-run singleton holder (pool + run id),
# the run's log (last iteration, step time, loss, grad norm, nan / skipped counts, log age), the run's
# rc file and the host's SLURM state. One status line per poll goes to $OUT. The script EXITS (= one
# notification for the agent that started it in the background) on anything to act on:
#   RESTART  the singleton moved to another run (requeue / restart / move)
#   ENDED    the run has an rc file
#   HOST     the host's SLURM job is not RUNNING
#   STALL    no new log line for STALL_S (saves take ~5 min, a restart's warm-up shows as RESTART)
#   NAN      nan / skipped iterations > 0
#   SLOW     step time > SLOW_MS in SLOW_N polls in a row
#   SSH      SSH_FAILS failed polls in a row
#   MONITOR  the run's POOL MONITOR is down (user: "for cases where the monitor fails"): its heartbeat
#            (<pool>/monitor.alive) is older than MON_STALE_S, or no monitor_autoexp.py process runs for
#            its session, or the MN5-side pool_monitor_watchdog.sh loop is gone. Without the monitor a
#            failed run is not requeued and a host is not looked after.
# It changes nothing; re-arm it after handling an event.
set -u
HOST=${HOST:-marenostrum_new}
INTERVAL=${INTERVAL:-600}
STALL_S=${STALL_S:-1800}
SLOW_MS=${SLOW_MS:-40000}
SLOW_N=${SLOW_N:-3}
SSH_FAILS=${SSH_FAILS:-3}
MON_STALE_S=${MON_STALE_S:-900}
SINGLETON=${SINGLETON:-ambitious_main_v4}
OUT=${OUT:-$HOME/main_run_health.log}
log() { echo "$(date '+%F %T') $*" | tee -a "$OUT"; }

probe() {
    timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=30 "$HOST" "SINGLETON=$SINGLETON bash -s" <<'EOF' 2>/dev/null | grep '^STATUS'
H=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/_singletons/$SINGLETON/holder.json
read -r pool rid < <(python3 -c "import json,sys; d=json.load(open('$H')); print(d['pool'], d['rid'])")
R=$pool/runs/$rid
rc=$(cat "$R/rc" 2>/dev/null | tr ' \n' '_ ')
L=$(sed -n 's/^LOG=//p' "$R/meta.env" | tr -d "'\"")
hid=$(cat "$R/assigned" 2>/dev/null)
jid=$(sed -n 's/^SLURM_JOB_ID=//p' "$pool/hosts/$hid/info.env" 2>/dev/null)
hstate=$(squeue -h -j "$jid" -o %T 2>/dev/null)
age=$(( $(date +%s) - $(stat -c %Y "$L" 2>/dev/null || echo 0) ))
last=$(grep -a '\[default3\].* iteration .*elapsed time per iteration' "$L" 2>/dev/null | tail -1)
it=$(sed -nE 's/.* iteration +([0-9]+)\/.*/\1/p' <<<"$last")
ms=$(sed -nE 's/.*elapsed time per iteration \(ms\): ([0-9.]+).*/\1/p' <<<"$last")
loss=$(sed -nE 's/.*lm loss: ([0-9.E+-]+|nan|inf).*/\1/p' <<<"$last")
gn=$(sed -nE 's/.*grad norm: ([0-9.a-z]+).*/\1/p' <<<"$last")
nan=$(sed -nE 's/.*number of nan iterations: +([0-9]+).*/\1/p' <<<"$last")
skip=$(sed -nE 's/.*number of skipped iterations: +([0-9]+).*/\1/p' <<<"$last")
mon_age=$(( $(date +%s) - $(stat -c %Y "$pool/monitor.alive" 2>/dev/null || echo 0) ))
sess=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/pool_state/pool_mn5-pool-${pool##*mn5_pool_}
mon=0; wd=0
for c in /proc/[0-9]*/cmdline; do
  l=$(tr '\0' ' ' <"$c" 2>/dev/null)
  case "$l" in
    *"scripts/monitor_autoexp.py --session-dir $sess "*) mon=1 ;;
    "/bin/bash /gpfs/projects/ehpc533/users/korbip/hybrid_v019/monitor_logs/pool_monitor_watchdog.sh"*) wd=1 ;;
  esac
done
echo "STATUS rid=$rid host=$hid job=$jid hstate=${hstate:-gone} rc=${rc:-none} age=$age it=${it:-none} ms=${ms:-0} loss=${loss:-none} gn=${gn:-none} nan=${nan:-0} skip=${skip:-0} mon_age=$mon_age mon=$mon wd=$wd"
EOF
}

field() { sed -nE "s/.* $1=([^ ]*).*/\1/p" <<<"$2"; }

first_rid=""; fails=0; slow=0
log "watch start (interval ${INTERVAL}s, stall ${STALL_S}s, slow ${SLOW_MS}ms x${SLOW_N})"
while true; do
    s=$(probe)
    if [ -z "$s" ]; then
        fails=$((fails + 1))
        log "probe failed ($fails/$SSH_FAILS)"
        [ "$fails" -ge "$SSH_FAILS" ] && { log "EVENT SSH: $fails failed polls in a row"; exit 10; }
        sleep "$INTERVAL"; continue
    fi
    fails=0
    log "$s"
    rid=$(field rid "$s")
    [ -z "$first_rid" ] && first_rid=$rid
    if [ "$rid" != "$first_rid" ]; then log "EVENT RESTART: singleton moved $first_rid -> $rid"; exit 11; fi
    if [ "$(field rc "$s")" != none ]; then log "EVENT ENDED: $rid rc=$(field rc "$s")"; exit 12; fi
    if [ "$(field hstate "$s")" != RUNNING ]; then log "EVENT HOST: job $(field job "$s") is $(field hstate "$s")"; exit 13; fi
    if [ "$(field age "$s")" -gt "$STALL_S" ]; then log "EVENT STALL: no log line for $(field age "$s")s"; exit 14; fi
    case "$(field loss "$s")" in nan|inf) log "EVENT NAN: loss $(field loss "$s")"; exit 15;; esac
    if [ "$(field nan "$s")" != 0 ] || [ "$(field skip "$s")" != 0 ]; then log "EVENT NAN: nan=$(field nan "$s") skipped=$(field skip "$s")"; exit 15; fi
    if [ "$(field mon_age "$s")" -gt "$MON_STALE_S" ] || [ "$(field mon "$s")" != 1 ] || [ "$(field wd "$s")" != 1 ]; then
        log "EVENT MONITOR: pool monitor heartbeat $(field mon_age "$s")s, process $(field mon "$s"), watchdog $(field wd "$s")"; exit 17
    fi
    ms=$(field ms "$s"); ms=${ms%.*}
    if [ "${ms:-0}" -gt "$SLOW_MS" ]; then slow=$((slow + 1)); else slow=0; fi
    if [ "$slow" -ge "$SLOW_N" ]; then log "EVENT SLOW: step ${ms} ms in $slow polls in a row"; exit 16; fi
    [ -n "${ONCE:-}" ] && exit 0
    sleep "$INTERVAL"
done
