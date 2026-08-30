#!/bin/bash
# Supervise cluster-side monitors from an always-on machine. Run OUTSIDE the cluster.
#
# The monitors themselves stay on the login node, next to the filesystem and the
# SLURM binaries -- a cluster can lose external connectivity while its jobs run
# perfectly, and a monitor living out here would be blind exactly then. This
# script only watches, and relaunches when the login node reboots or a monitor
# dies. It is optional: running without it costs nothing.
#
# Watch every live session in a monitor state folder (the usual way):
#   scripts/supervise_monitor.sh --ssh 'ssh jupiter' \
#       --repo '~/work/Projects/oellm-autoexp' \
#       --monitor-state-dir '~/work/Projects/oellm-autoexp/monitor_state' \
#       --notify-email you@example.org
#
# ...or pin it to one session:
#   scripts/supervise_monitor.sh --ssh 'ssh jupiter' --repo '...' --session 1787929769
#
# Sessions are discovered by monitor_probe.sh --all, which lists only those whose
# jobs are IN THE SLURM QUEUE right now. It will never resurrect a session whose
# jobs are gone, so it cannot resubmit abandoned work. See that script for why.
#
# Options:
#   --ssh <prefix>              how to reach the cluster; EMPTY means run locally (tests)
#   --repo <path>               repo root on the cluster
#   --monitor-state-dir <path>  watch every live session under here
#   --session <id>              ...or just this one, under <repo>/monitor_state
#   --session-dir <path>        ...or this session directory outright
#   --poll <s>                  seconds between probes                 (default 60)
#   --stale-after <s>           heartbeat age that counts as stale     (default 300)
#   --wedge-grace <s>           how long a WEDGED monitor is tolerated (default 600)
#   --restart-budget <n>        restarts per hour, per session         (default 5)
#   --notify-email <addr>       where to mail state changes
#   --notify-cmd <cmd>          override: invoked as `<cmd> "<subject>"`, body on stdin
#   --budget-dir <path>         where per-session restart budgets are kept
#   --dry-run                   report what it WOULD do; never launch, kill or write
#   --once                      run a single tick and exit (tests)
#
# ALWAYS `--dry-run --once` first against a real folder. A monitor started
# before this change writes no heartbeat, so it probes as DEAD and a live
# session would get a SECOND monitor -- see the migration note in the README.
set -u

SSH_PREFIX=""
REPO=""
SESSION=""
SESSION_DIR=""
STATE_ROOT=""
POLL=60
STALE_AFTER=300
WEDGE_GRACE=600
RESTART_BUDGET=5
NOTIFY_EMAIL=""
NOTIFY_CMD=""
BUDGET_DIR=""
DRY_RUN=0
ONCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --ssh) SSH_PREFIX="${2:-}"; shift 2 ;;
    --repo) REPO="${2:-}"; shift 2 ;;
    --session) SESSION="${2:-}"; shift 2 ;;
    --session-dir) SESSION_DIR="${2:-}"; shift 2 ;;
    --monitor-state-dir) STATE_ROOT="${2:-}"; shift 2 ;;
    --poll) POLL="${2:-}"; shift 2 ;;
    --stale-after) STALE_AFTER="${2:-}"; shift 2 ;;
    --wedge-grace) WEDGE_GRACE="${2:-}"; shift 2 ;;
    --restart-budget) RESTART_BUDGET="${2:-}"; shift 2 ;;
    --notify-email) NOTIFY_EMAIL="${2:-}"; shift 2 ;;
    --notify-cmd) NOTIFY_CMD="${2:-}"; shift 2 ;;
    --budget-dir) BUDGET_DIR="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --once) ONCE=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$REPO" ] || { echo "--repo is required" >&2; exit 2; }
if [ -z "$STATE_ROOT" ] && [ -z "$SESSION_DIR" ]; then
  [ -n "$SESSION" ] || {
    echo "one of --monitor-state-dir, --session or --session-dir is required" >&2; exit 2; }
  SESSION_DIR="$REPO/monitor_state/$SESSION"
fi

if [ -z "$BUDGET_DIR" ]; then
  for _d in "${XDG_RUNTIME_DIR:-}" "${TMPDIR:-}" /tmp; do
    if [ -n "$_d" ] && [ -d "$_d" ] && [ -w "$_d" ]; then BUDGET_DIR="$_d"; break; fi
  done
fi
[ -d "$BUDGET_DIR" ] && [ -w "$BUDGET_DIR" ] || {
  echo "budget dir not writable: $BUDGET_DIR" >&2; exit 2; }

NL='
'

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

notify() {
  local subject="$1"; shift
  local body="$*"
  log "[notify] $subject | $body"
  if [ -n "$NOTIFY_CMD" ]; then
    printf '%s\n' "$body" | $NOTIFY_CMD "$subject" || log "[notify] notify-cmd failed"
  elif [ -n "$NOTIFY_EMAIL" ]; then
    printf '%s\n' "$body" | mail -s "$subject" "$NOTIFY_EMAIL" || log "[notify] mail failed"
  fi
}

# One command string, whether it runs here or over ssh, so the two paths cannot
# drift. Cluster paths never contain spaces; do not add any.
remote() {
  if [ -n "$SSH_PREFIX" ]; then
    # shellcheck disable=SC2086
    $SSH_PREFIX "$*"
  else
    bash -c "$*"
  fi
}

dir_for() { # dir_for <session_id>
  if [ -n "$STATE_ROOT" ]; then printf '%s/%s' "$STATE_ROOT" "$1"; else printf '%s' "$SESSION_DIR"; fi
}

budget_for() { # budget_for <session_dir>
  printf '%s/autoexp-supervisor-%s.restarts' \
    "$BUDGET_DIR" "$(printf '%s' "$1" | cksum | cut -d' ' -f1)"
}

# ONE ssh per tick, whatever the number of sessions: --all does the discovery
# and the health check in a single pass on the cluster.
collect() {
  if [ -n "$STATE_ROOT" ]; then
    remote "STALE_AFTER=$STALE_AFTER bash '$REPO/scripts/monitor_probe.sh' --all '$STATE_ROOT'" \
      2>/dev/null
  else
    local line
    line="$(remote "STALE_AFTER=$STALE_AFTER bash '$REPO/scripts/monitor_probe.sh' --session-dir '$SESSION_DIR'" 2>/dev/null | tail -1)"
    [ -n "$line" ] && printf '%s %s\n' "$(basename "$SESSION_DIR")" "$line"
  fi
}

launch() { # launch <session_dir>
  if [ "$DRY_RUN" = "1" ]; then echo "[dry-run] would launch $1"; return; fi
  remote "bash '$REPO/scripts/monitor_launch.sh' --session-dir '$1'" 2>&1 | tail -2
}

terminate() { # terminate <pid>
  if [ "$DRY_RUN" = "1" ]; then log "[dry-run] would SIGTERM then SIGKILL pid $1"; return; fi
  remote "kill -TERM $1" >/dev/null 2>&1
  sleep 60
  remote "kill -0 $1 2>/dev/null && kill -KILL $1" >/dev/null 2>&1
}

kv() { printf '%s\n' "$1" | tr ' ' '\n' | sed -n "s/^$2=//p" | head -1; }

declare -A LAST_STATE=()
declare -A WEDGE_SINCE=()
unreachable=0

restart() { # restart <session_id> <session_dir> <detail>
  local sid="$1" dir="$2" detail="$3" now cutoff recent count out budget
  budget="$(budget_for "$dir")"
  now=$(date +%s)
  cutoff=$(( now - 3600 ))
  # The budget lives in a FILE, not a variable: systemd restarts this supervisor
  # with Restart=always, and an in-memory counter would reset exactly when a
  # crashloop is worst. It also makes --once usable, and keeps the budget
  # per-session when many sessions are watched at once.
  recent="$(awk -v c="$cutoff" '$1 ~ /^[0-9]+$/ && $1+0 > c {print $1}' "$budget" 2>/dev/null)"
  count="$(printf '%s' "$recent" | grep -c . || true)"

  if [ "$count" -ge "$RESTART_BUDGET" ]; then
    # Crashlooping. Stop fighting it and get a human involved: writing the stop
    # file makes the next probe report STOPPED, so we go quiet by construction
    # without needing a separate "given up" state.
    notify "autoexp[$sid]: monitor crashlooping, giving up" \
      "$count restarts in the last hour for $dir.${NL}Last probe: $detail${NL}Stop file written. Investigate, then: rm $dir/.monitor.stop"
    if [ "$DRY_RUN" = "1" ]; then
      log "[dry-run] would write $dir/.monitor.stop"
    else
      remote "printf '%s\n' 'crashloop: $count restarts/h, supervisor gave up' > '$dir/.monitor.stop'"
      : > "$budget"
    fi
    return
  fi

  if [ "$DRY_RUN" != "1" ]; then
    { [ -n "$recent" ] && printf '%s\n' "$recent"; printf '%s\n' "$now"; } > "$budget"
  fi
  out="$(launch "$dir")"
  log "[$sid] restarting #$(( count + 1 ))/$RESTART_BUDGET ($detail) -> $out"
  notify "autoexp[$sid]: monitor restarted" "$detail${NL}launch: $out"
}

handle() { # handle <session_id> <state> <detail...>
  local sid="$1" state="$2"; shift 2
  local detail="$*" dir prev now waited pid
  dir="$(dir_for "$sid")"
  prev="${LAST_STATE[$sid]:-}"

  case "$state" in
    OK)
      WEDGE_SINCE[$sid]=0
      [ "$prev" != "OK" ] && log "[$sid] OK  $detail"
      ;;

    STOPPED)
      WEDGE_SINCE[$sid]=0
      if [ "$prev" != "STOPPED" ]; then
        notify "autoexp[$sid]: monitor stopped" \
          "$detail -- standing down. Resume: rm $dir/.monitor.stop"
      fi
      ;;

    NOSESSION)
      [ "$prev" != "NOSESSION" ] && notify "autoexp[$sid]: session directory missing" "$detail"
      ;;

    DEAD)
      WEDGE_SINCE[$sid]=0
      restart "$sid" "$dir" "DEAD $detail"
      ;;

    WEDGED)
      now=$(date +%s)
      if [ "${WEDGE_SINCE[$sid]:-0}" -eq 0 ]; then
        WEDGE_SINCE[$sid]=$now
        log "[$sid] WEDGED  $detail (grace ${WEDGE_GRACE}s)"
      fi
      waited=$(( now - WEDGE_SINCE[$sid] ))
      if [ "$waited" -ge "$WEDGE_GRACE" ]; then
        pid="$(kv "$detail" pid)"
        log "[$sid] wedged for ${waited}s; terminating pid $pid"
        # SIGTERM first: the monitor turns it into a flag and finishes the poll
        # in progress, so a signal landing inside an sbatch does not leave a
        # submitted job whose id was never recorded. It deliberately does NOT
        # write a stop file, so this restart is not blocked by our own kill.
        terminate "$pid"
        WEDGE_SINCE[$sid]=0
        restart "$sid" "$dir" "WEDGED $detail (wedged ${waited}s)"
      fi
      ;;

    *)
      log "[$sid] unrecognised probe output: $state $detail"
      ;;
  esac
  LAST_STATE[$sid]="$state"
}

tick() {
  local out sid rest seen=" "
  out="$(collect)"
  if [ -z "$out" ] && [ -n "$SESSION_DIR" ]; then
    # Single-session mode: no output at all means the probe itself failed. The
    # monitor may be perfectly healthy behind a dead network link, so NEVER act.
    unreachable=$(( unreachable + 1 ))
    log "UNREACHABLE (x$unreachable)"
    [ "$unreachable" -eq 5 ] && notify "autoexp: cluster unreachable" \
      "5 consecutive probe failures for $SESSION_DIR. Not acting; the monitor may still be running."
    return
  fi
  if [ -n "$STATE_ROOT" ] && ! remote "test -d '$STATE_ROOT'" >/dev/null 2>&1; then
    # Discovery legitimately returns nothing when no session needs supervising,
    # so emptiness alone cannot mean "unreachable". Ask a question with a known
    # answer instead.
    unreachable=$(( unreachable + 1 ))
    log "UNREACHABLE (x$unreachable)"
    [ "$unreachable" -eq 5 ] && notify "autoexp: cluster unreachable" \
      "5 consecutive probe failures for $STATE_ROOT. Not acting; monitors may still be running."
    return
  fi
  if [ "$unreachable" -gt 0 ]; then
    log "reachable again after $unreachable failed probe(s)"
    unreachable=0
  fi

  while read -r sid rest; do
    [ -n "$sid" ] || continue
    seen="$seen$sid "
    # shellcheck disable=SC2086
    handle "$sid" $rest
  done <<< "$out"

  # A session that drops out of discovery no longer has anything in the queue:
  # it finished, or its jobs are gone. Say so once, then forget it.
  for sid in "${!LAST_STATE[@]}"; do
    case "$seen" in
      *" $sid "*) ;;
      *)
        notify "autoexp[$sid]: no longer needs supervision" \
          "No queued jobs remain for this session; dropping it."
        unset "LAST_STATE[$sid]" "WEDGE_SINCE[$sid]"
        ;;
    esac
  done
}

if [ -n "$STATE_ROOT" ]; then
  log "supervising every live session under $STATE_ROOT via ${SSH_PREFIX:-<local>}"
else
  log "supervising $SESSION_DIR via ${SSH_PREFIX:-<local>}"
fi
log "poll ${POLL}s, stale ${STALE_AFTER}s, wedge grace ${WEDGE_GRACE}s, budget ${RESTART_BUDGET}/h"
while true; do
  tick
  [ "$ONCE" -eq 1 ] && break
  sleep "$POLL"
done
