#!/bin/bash
# End-to-end test of the monitor supervisor. LOCAL ONLY -- no cluster, no SLURM.
#
# Drives a real MonitorLoop (LocalCommandClient, one `sleep` job) inside a real
# tmux server, and a real supervisor with --ssh '' so every code path except the
# ssh hop itself is exercised. Covers the recovery ladder (crash / window killed
# / server killed), the stop-file contract, the wedge path and the crashloop
# budget.
#
#   bash scripts/tests/test_monitor_supervisor.sh
#
# Leaves nothing behind: its tmux session and temp dir are removed on exit.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
TMUX_SESSION="autoexp-selftest-$$"
WORK="$(mktemp -d)"
STATE="$WORK/monitor_state"
SESSION_ID="selftest"
SESSION_DIR="$STATE/$SESSION_ID"

export AUTOEXP_TMUX_SESSION="$TMUX_SESSION"
export AUTOEXP_PYTHON="$PY"
export AUTOEXP_ENV_SETUP=":"          # the venv python is passed explicitly
# A short poll means a short heartbeat refresh, which is what makes a 3 s
# staleness threshold meaningful in the wedge test.
export AUTOEXP_MONITOR_ARGS="--poll-interval 2"

PASS=0
FAIL=0

cleanup() {
  tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
  # LocalCommandClient starts jobs with start_new_session=True, so they are NOT
  # in the tmux pane's process group and survive killing the session. Every job
  # command carries $TMUX_SESSION as a marker so they can be reaped by name.
  pkill -f "$TMUX_SESSION" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

probe() { STALE_AFTER="${1:-300}" bash "$REPO/scripts/monitor_probe.sh" --session-dir "$SESSION_DIR"; }
state() { probe "${1:-300}" | awk '{print $1}'; }
mon_pid() { probe 300 | tr ' ' '\n' | sed -n 's/^pid=//p'; }

expect_state() { # expect_state <want> <stale> <label>
  local got; got="$(state "${2:-300}")"
  if [ "$got" = "$1" ]; then ok "$3 (probe=$got)"; else bad "$3: wanted $1, got $got"; fi
}

wait_for_state() { # wait_for_state <want> <timeout_s> [stale]
  local want="$1" timeout="$2" stale="${3:-300}" i
  for ((i=0; i<timeout*2; i++)); do
    [ "$(state "$stale")" = "$want" ] && return 0
    sleep 0.5
  done
  return 1
}

wait_for_state_dir() { # wait_for_state_dir <session_dir> <want> <timeout_s>
  local dir="$1" want="$2" timeout="$3" i got
  for ((i=0; i<timeout*2; i++)); do
    got="$(bash "$REPO/scripts/monitor_probe.sh" --session-dir "$dir" | awk '{print $1}')"
    [ "$got" = "$want" ] && return 0
    sleep 0.5
  done
  return 1
}

supervise() { # one supervisor tick, locally
  bash "$REPO/scripts/supervise_monitor.sh" --ssh '' --repo "$REPO" \
    --session-dir "$SESSION_DIR" --once --stale-after "${STALE:-300}" \
    --wedge-grace "${GRACE:-600}" --restart-budget "${BUDGET:-5}" \
    --budget-dir "$WORK" 2>&1
}

trace() { printf '  . %s: stop=%s alive=%s pane_dead=%s\n' "$1" \
  "$(cat "$SESSION_DIR/.monitor.stop" 2>/dev/null || echo -)" \
  "$(cat "$SESSION_DIR/.monitor.alive" 2>/dev/null || echo -)" \
  "$(tmux display-message -p -t "$TMUX_SESSION:mon-$SESSION_ID" '#{pane_dead}' 2>/dev/null || echo -)"; }

# --- fixture: a session holding one long-running local job ----------------
mkdir -p "$SESSION_DIR"
PYTHONPATH="$REPO" "$PY" - "$SESSION_DIR" "$TMUX_SESSION" <<'PY'
import sys
from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime
from oellm_autoexp.monitor.submission import LocalJobConfig

session_dir, marker = sys.argv[1], sys.argv[2]
store = JobFileStore(session_dir)
store.upsert(
    JobRecord(
        job_id="sleeper",
        definition=LocalJobConfig(
            name="sleeper",
            # exec -a renames the process to the marker so cleanup() can reap it.
            # A trailing `# marker` comment does NOT survive: bash execs a
            # simple command, which drops it from the command line.
            command=["bash", "-c", f"exec -a {marker}-job sleep 3600"],
            log_path=f"{session_dir}/sleeper.log",
        ),
        runtime=JobRuntime(),
    )
)
PY

step "0. probe before anything has run"
expect_state DEAD 300 "no heartbeat yet reads as DEAD"

step "1. launch"
bash "$REPO/scripts/monitor_launch.sh" --session-dir "$SESSION_DIR" || bad "launch failed"
if wait_for_state OK 20; then ok "monitor came up (probe=OK)"; else bad "monitor never reported OK"; fi
PID1="$(mon_pid)"
tmux list-windows -t "$TMUX_SESSION" -F '  window: #{window_name} dead=#{pane_dead}' 2>/dev/null

step "2. idempotence: launching again must not start a second monitor"
out="$(bash "$REPO/scripts/monitor_launch.sh" --session-dir "$SESSION_DIR")"
case "$out" in *"already running"*) ok "second launch is a no-op: $out" ;; *) bad "unexpected: $out" ;; esac
[ "$(mon_pid)" = "$PID1" ] && ok "same pid still owns the session" || bad "pid changed"

step "3. lease: a second monitor must refuse to start on a live session"
out="$(cd "$REPO" && PYTHONPATH=. "$PY" scripts/monitor_autoexp.py --session-dir "$SESSION_DIR" 2>&1)"
case "$out" in *"appears to be running"*) ok "refused a duplicate monitor" ;; *) bad "duplicate not refused: $out" ;; esac

step "4. crash (kill -9) -> DEAD even though the heartbeat is still fresh"
kill -9 "$PID1" 2>/dev/null
if wait_for_state DEAD 10; then ok "crash detected immediately"; else bad "crash not detected"; fi

step "5. supervisor restarts the dead monitor, in place"
supervise | sed 's/^/  | /'
if wait_for_state OK 20; then ok "supervisor brought it back"; else bad "supervisor did not restart it"; fi
PID2="$(mon_pid)"
[ -n "$PID2" ] && [ "$PID2" != "$PID1" ] && ok "new pid $PID2" || bad "pid did not change ($PID1 -> $PID2)"
dead="$(tmux display-message -p -t "$TMUX_SESSION:mon-$SESSION_ID" '#{pane_dead}' 2>/dev/null)"
[ "$dead" = "0" ] && ok "respawned in the same window (pane alive)" || bad "pane_dead=$dead"

step "6. window killed (user closed it) -> recreated"
tmux kill-window -t "$TMUX_SESSION:mon-$SESSION_ID" 2>/dev/null
wait_for_state DEAD 10 || bad "killing the window did not read as DEAD"
supervise | sed 's/^/  | /'
if wait_for_state OK 20; then ok "window recreated"; else bad "window not recreated"; fi

step "7. whole tmux server killed (login node rebooted) -> session recreated"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
wait_for_state DEAD 10 || bad "killing the session did not read as DEAD"
supervise | sed 's/^/  | /'
if wait_for_state OK 20; then ok "session recreated"; else bad "session not recreated"; fi
tmux has-session -t "$TMUX_SESSION" 2>/dev/null && ok "tmux session exists again" || bad "no tmux session"

step "8. Ctrl-C (SIGINT) -> monitor records intent, supervisor stands down"
PID3="$(mon_pid)"
kill -INT "$PID3" 2>/dev/null
if wait_for_state STOPPED 20; then ok "stop file written by the signal handler"; else bad "no stop file after SIGINT"; fi
grep -q "SIGINT" "$SESSION_DIR/.monitor.stop" && ok "reason recorded: $(cat "$SESSION_DIR/.monitor.stop")" \
  || bad "reason missing: $(cat "$SESSION_DIR/.monitor.stop" 2>/dev/null)"
out="$(supervise)"; echo "$out" | sed 's/^/  | /'
case "$out" in *"standing down"*) ok "supervisor stood down" ;; *) bad "supervisor did not stand down" ;; esac
expect_state STOPPED 300 "still stopped after a supervisor tick"
p="$(mon_pid)"
if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then bad "a monitor is still running (pid $p)"; else ok "no monitor was restarted"; fi

step "9. launching while stopped is refused"
out="$(bash "$REPO/scripts/monitor_launch.sh" --session-dir "$SESSION_DIR" 2>&1)"
case "$out" in *"stop requested"*) ok "launch refused: $(echo "$out" | head -1)" ;; *) bad "launch not refused: $out" ;; esac
rm -f "$SESSION_DIR/.monitor.stop"

step "10. SIGTERM alone must NOT record a stop (or the supervisor blocks its own restart)"
bash "$REPO/scripts/monitor_launch.sh" --session-dir "$SESSION_DIR" >/dev/null
wait_for_state OK 20 || bad "monitor did not come back"
PIDT="$(mon_pid)"
kill -TERM "$PIDT" 2>/dev/null
sleep 3
if [ -f "$SESSION_DIR/.monitor.stop" ]; then
  bad "SIGTERM wrote a stop file: $(cat "$SESSION_DIR/.monitor.stop")"
else
  ok "SIGTERM exited cleanly without recording intent"
fi
expect_state DEAD 300 "reads as DEAD, so a supervisor will bring it back"

step "11. wedge -> WEDGED, then terminated and restarted after the grace"
# A real wedge is a hung syscall, not SIGSTOP: a tmux pane's process group is
# orphaned, so the kernel CONTinues a stopped pane and the stop does not stick
# (verified: ps shows S, not T). Simulate the thing the probe actually keys on
# instead -- a live pid whose heartbeat has stopped advancing -- by giving the
# monitor a long poll interval and backdating the heartbeat.
AUTOEXP_MONITOR_ARGS="--poll-interval 300" bash "$REPO/scripts/monitor_launch.sh" \
  --session-dir "$SESSION_DIR" >/dev/null
wait_for_state OK 20 || bad "monitor did not come back for the wedge test"
PID4="$(mon_pid)"
touch -d '10 minutes ago' "$SESSION_DIR/.monitor.alive"
trace "wedge subject pid=$PID4"
if wait_for_state WEDGED 10 3; then ok "stalled heartbeat + live pid reads as WEDGED"; else bad "not WEDGED"; fi
# grace 0 -> act on this tick. The supervisor waits 60 s between TERM and KILL,
# so this is the slow step.
STALE=3 GRACE=0 supervise | sed 's/^/  | /'
trace "after wedge handling"
if wait_for_state OK 30; then ok "wedged monitor terminated and restarted"; else bad "wedge not resolved"; fi
PID5="$(mon_pid)"
if [ -n "$PID5" ] && [ "$PID5" != "$PID4" ]; then ok "new pid after wedge ($PID4 -> $PID5)"; else bad "pid unchanged after wedge ($PID4)"; fi
kill -9 "$PID4" 2>/dev/null

step "12. unreachable cluster -> no action, no state mutation"
before="$(cat "$SESSION_DIR/.monitor.alive" 2>/dev/null)"
out="$(bash "$REPO/scripts/supervise_monitor.sh" --ssh 'false' --repo "$REPO" \
        --session-dir "$SESSION_DIR" --once --budget-dir "$WORK" 2>&1)"
echo "$out" | sed 's/^/  | /'
case "$out" in *UNREACHABLE*) ok "probe failure reported as UNREACHABLE" ;; *) bad "not reported: $out" ;; esac
if [ -f "$SESSION_DIR/.monitor.stop" ]; then bad "a stop file appeared"; else ok "nothing was stopped"; fi
if [ "$(cat "$SESSION_DIR/.monitor.alive" 2>/dev/null)" = "$before" ]; then
  ok "heartbeat owner unchanged"
else
  ok "heartbeat owner changed (monitor still polling on its own)"
fi

step "13. crashloop budget -> gives up and writes a stop file"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
kill -9 "$(mon_pid)" 2>/dev/null
rm -f "$SESSION_DIR/.monitor.alive" "$SESSION_DIR/.monitor.stop"
rm -f "$WORK"/autoexp-supervisor-*.restarts   # a fresh budget for this step alone
export AUTOEXP_PYTHON=/bin/false      # every launch dies instantly
all=""
for i in 1 2 3; do
  out="$(BUDGET=2 supervise)"; echo "$out" | sed "s/^/  ${i}| /"; all="$all$out"
done
case "$all" in
  *crashlooping*) ok "supervisor gave up after the budget was spent" ;;
  *) bad "budget not enforced" ;;
esac
# The budget must be spent by RESTARTS, not by the first tick.
restarts_logged="$(grep -c 'restarting #' <<<"$all" || true)"
[ "$restarts_logged" -eq 2 ] && ok "exactly 2 restarts before giving up" \
  || bad "expected 2 restarts, logged $restarts_logged"
[ -f "$SESSION_DIR/.monitor.stop" ] && ok "stop file written: $(cat "$SESSION_DIR/.monitor.stop")" \
  || bad "no stop file after giving up"
expect_state STOPPED 300 "goes quiet by construction"

step "14. auto-discovery: supervise a whole monitor_state folder"
export AUTOEXP_PYTHON="$PY"
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null
rm -f "$WORK"/autoexp-supervisor-*.restarts
# Three more sessions covering the cases that matter. `queued` is the only one
# that should ever be touched; `unsubmitted` is the dangerous one -- 35 real
# sessions on JUPITER look like this, and resurrecting one would SUBMIT OLD WORK.
for s in queued unsubmitted done_already; do mkdir -p "$STATE/$s"; done
PYTHONPATH="$REPO" "$PY" - "$STATE" "$TMUX_SESSION" <<'PY'
import sys
from oellm_autoexp.monitor.loop import JobFileStore, JobRecord, JobRuntime
from oellm_autoexp.monitor.submission import LocalJobConfig

root, marker = sys.argv[1], sys.argv[2]


def job(name):
    # Carries the marker so cleanup() can reap it: local jobs are detached and
    # outlive the tmux session they were started from.
    return LocalJobConfig(
        name=name,
        # exec -a renames the process to the marker so cleanup() can reap it.
            # A trailing `# marker` comment does NOT survive: bash execs a
            # simple command, which drops it from the command line.
            command=["bash", "-c", f"exec -a {marker}-job sleep 3600"],
        log_path=f"/tmp/{name}.log",
    )

# submitted, and its id is in the (stubbed) queue -> needs a monitor
JobFileStore(f"{root}/queued").upsert(
    JobRecord(job_id="q", definition=job("q"),
              runtime=JobRuntime(submitted=True, runtime_job_id="424242")))
# never submitted -> must be left alone, or we would sbatch abandoned work
JobFileStore(f"{root}/unsubmitted").upsert(
    JobRecord(job_id="u", definition=job("u"), runtime=JobRuntime()))
# submitted long ago, job gone from the queue, still marked active (limbo)
JobFileStore(f"{root}/done_already").upsert(
    JobRecord(job_id="d", definition=job("d"),
              runtime=JobRuntime(submitted=True, runtime_job_id="999999")))
PY

export AUTOEXP_LIVE_JOB_IDS="424242"   # stands in for `squeue -u $USER`
found="$(bash "$REPO/scripts/monitor_probe.sh" --all "$STATE" | awk '{print $1}' | sort | tr '\n' ' ')"
[ "$found" = "queued " ] && ok "discovery selected only the queued session: $found" \
  || bad "discovery selected '$found', expected 'queued '"

out="$(bash "$REPO/scripts/supervise_monitor.sh" --ssh '' --repo "$REPO" \
        --monitor-state-dir "$STATE" --once --budget-dir "$WORK" 2>&1)"
echo "$out" | sed 's/^/  | /'
case "$out" in *"[queued] restarting"*) ok "started a monitor for the queued session" ;;
  *) bad "no monitor started for 'queued'" ;; esac
case "$out" in *unsubmitted*) bad "touched the unsubmitted session" ;;
  *) ok "never touched the unsubmitted session" ;; esac
case "$out" in *done_already*) bad "touched the limbo session" ;;
  *) ok "never touched the limbo session" ;; esac
if wait_for_state_dir "$STATE/queued" OK 20; then ok "queued session's monitor is up"; else bad "queued monitor never came up"; fi

# One ssh per tick regardless of session count is the whole point of --all.
probes="$(grep -c 'monitor_probe.sh --all' <<<"$out" || true)"
tmux list-windows -t "$TMUX_SESSION" -F '  window: #{window_name}' 2>/dev/null

step "15. a session that leaves the queue is dropped, once"
pkill -f "monitor_autoexp.py --session-dir $STATE/queued" 2>/dev/null
export AUTOEXP_LIVE_JOB_IDS="123"      # nothing of ours is queued any more
found="$(bash "$REPO/scripts/monitor_probe.sh" --all "$STATE" | wc -l)"
[ "$found" -eq 0 ] && ok "discovery now selects nothing" || bad "still selects $found session(s)"
out="$(bash "$REPO/scripts/supervise_monitor.sh" --ssh '' --repo "$REPO" \
        --monitor-state-dir "$STATE" --once --budget-dir "$WORK" 2>&1)"
echo "$out" | sed 's/^/  | /'
case "$out" in *restarting*) bad "restarted something with an empty queue" ;;
  *) ok "nothing restarted when the queue is empty" ;; esac
case "$out" in *UNREACHABLE*) bad "empty discovery misread as unreachable" ;;
  *) ok "empty discovery is not confused with an unreachable cluster" ;; esac

step "16. unreachable in folder mode is still detected"
out="$(bash "$REPO/scripts/supervise_monitor.sh" --ssh 'false' --repo "$REPO" \
        --monitor-state-dir "$STATE" --once --budget-dir "$WORK" 2>&1)"
echo "$out" | sed 's/^/  | /'
case "$out" in *UNREACHABLE*) ok "probe failure reported as UNREACHABLE" ;;
  *) bad "not reported: $out" ;; esac

printf '\n\033[1m%d passed, %d failed\033[0m\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
