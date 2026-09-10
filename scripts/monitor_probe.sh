#!/bin/bash
# Report the health of monitor sessions, one line each. Run ON the cluster.
#
#   scripts/monitor_probe.sh <session_id> | --session-dir <dir>
#       -> <STATE> age=<s> pid=<pid> host=<host> tmux=<target>
#
#   scripts/monitor_probe.sh --all <monitor_state_dir>
#       -> <session_id> <STATE> age=... pid=... host=... tmux=...
#          one line per session that NEEDS supervising (see below); nothing else.
#
# Only the state field is significant; the rest is detail for the supervisor's
# emails.
#
#   STOPPED  .monitor.stop exists -- an intentional stop, keep hands off
#   OK       heartbeat is fresh
#   DEAD     no heartbeat, or a stale one whose process is gone
#   WEDGED   heartbeat is stale but the process is still alive
#
# The distinction matters because the responses differ: DEAD is relaunched
# immediately, WEDGED waits out a grace period before anything is killed.
set -u

STALE_AFTER="${STALE_AFTER:-300}"
# Only needed to resolve a bare <session_id>. Tolerate not knowing it, so the
# script can be dry-run by piping it in (`ssh host bash -s -- --all <dir>`)
# without having to sync anything first.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"

# --- which sessions deserve a monitor? ------------------------------------
#
# "Has an unfinished job record" is NOT the answer, and getting this wrong is
# dangerous. Measured on JUPITER 2026-08-28: of 506 session directories, 367
# still hold a record with final_state=null -- they are abandoned sessions whose
# monitor died before it could mark anything finished. Worse, 35 hold records
# with submitted=false, so resurrecting one would SUBMIT OLD WORK.
#
# The rule that works: a session needs a monitor iff one of its unfinished
# records names a SLURM job that is in the queue RIGHT NOW. That selected
# exactly 1 of the 506 (the live flagship), it can never resurrect a session
# whose jobs are gone, and it needs no age heuristic.
#
# Accepted limitation: if a monitor dies in the gap between chain stages, with
# nothing of its own in the queue, the session is not picked up. The alternative
# -- resurrecting sessions with unsubmitted work -- is far worse.
_live_job_ids() {
  # AUTOEXP_LIVE_JOB_IDS lets the local test harness stand in for squeue.
  if [ -n "${AUTOEXP_LIVE_JOB_IDS:-}" ]; then
    printf ' %s ' "$AUTOEXP_LIVE_JOB_IDS"
    return
  fi
  printf ' %s ' "$(squeue -u "${USER:-$(id -un)}" -h -o '%i' 2>/dev/null | tr '\n' ' ')"
}

_needs_monitor() { # _needs_monitor <session_dir> <" id1 id2 ">
  local dir="$1" live="$2" f rid
  for f in "$dir"/*.job.json; do
    [ -f "$f" ] || continue
    grep -q '"final_state": null' "$f" || continue
    rid=$(sed -n 's/.*"runtime_job_id": "\([^"]*\)".*/\1/p' "$f" | head -1)
    [ -n "$rid" ] || continue
    case "$live" in *" $rid "*) return 0 ;; esac
  done
  return 1
}

# --- health of one session -------------------------------------------------

probe_one() { # probe_one <session_dir> -> "<STATE> <detail>"
  local dir="$1" hb now mtime age pid host target detail same_host alive

  if [ ! -d "$dir" ]; then
    echo "NOSESSION dir=$dir"
    return
  fi
  if [ -f "$dir/.monitor.stop" ]; then
    echo "STOPPED reason=$(tr -d '\n' < "$dir/.monitor.stop")"
    return
  fi

  hb="$dir/.monitor.alive"
  if [ ! -f "$hb" ]; then
    echo "DEAD age=- pid=- host=- tmux=-"
    return
  fi

  # The timestamp IS the mtime, so liveness costs one stat and never depends on
  # the caller's clock -- both `date` and `stat` run here, on the cluster.
  now=$(date +%s)
  mtime=$(stat -c %Y "$hb" 2>/dev/null || echo 0)
  age=$(( now - mtime ))
  read -r pid host target _rest < "$hb" || true
  pid="${pid:--}"; host="${host:--}"; target="${target:--}"
  detail="age=$age pid=$pid host=$host tmux=$target"

  # Is the process still there? `kill -0` only answers for a monitor on THIS
  # node; a heartbeat naming another host is unknowable from here.
  same_host=0
  [ "$host" = "$(hostname)" ] && same_host=1
  alive=unknown
  if [ "$same_host" = "1" ]; then
    if kill -0 "$pid" 2>/dev/null; then alive=yes; else alive=no; fi
  fi

  # A crash leaves a heartbeat that is still FRESH, so age alone would hide a
  # dead monitor for a full stale window. Trust the pid first when we can see it.
  if [ "$alive" = "no" ]; then
    echo "DEAD $detail"
    return
  fi
  if [ "$age" -le "$STALE_AFTER" ]; then
    echo "OK $detail"
    return
  fi
  # Stale heartbeat. Alive-but-stuck and gone need different treatment. If the
  # monitor was on another login node we cannot check it, and report DEAD: the
  # monitor's own startup lease (ensure_no_live_monitor) reads the same
  # heartbeat off the shared filesystem and is the second guard against a
  # duplicate.
  if [ "$alive" = "yes" ]; then
    echo "WEDGED $detail"
  else
    echo "DEAD $detail"
  fi
}

# --- entry -----------------------------------------------------------------

case "${1:-}" in
  --all)
    STATE_ROOT="${2:-}"
    [ -d "$STATE_ROOT" ] || { echo "no such monitor state dir: $STATE_ROOT" >&2; exit 2; }
    LIVE="$(_live_job_ids)"
    for d in "$STATE_ROOT"/*/; do
      d="${d%/}"
      [ -d "$d" ] || continue
      [ "$(basename "$d")" = "manifests" ] && continue
      _needs_monitor "$d" "$LIVE" || continue
      printf '%s %s\n' "$(basename "$d")" "$(probe_one "$d")"
    done
    ;;
  --session-dir) probe_one "${2:-}" ;;
  "" ) echo "usage: $0 <session_id> | --session-dir <dir> | --all <monitor_state_dir>" >&2; exit 2 ;;
  * ) probe_one "${AUTOEXP_STATE_DIR:-$REPO/monitor_state}/$1" ;;
esac
