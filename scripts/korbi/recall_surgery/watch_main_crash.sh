#!/usr/bin/env bash
# Watch the v4b main run for every failure class it has produced, and preserve the evidence.
#
# Why: the run dies in several ways, each of which was missed by hand at least once:
#   * illegal access (MoonEP planning) reported only asynchronously by the NCCL watchdog,
#   * host-RAM OOM at a persistent save (pjob .2407, iteration 57000: 13 tasks OOM-killed),
#   * rendezvous / TCPStore failure (DistNetworkError, recvValueWithTimeout, keep-alive),
#   * MoonEP trap (tokens_per_expert disagrees with topk), NaN, step refused.
# The pool restarts the run within a minute and prunes the run directory, so the log line and
# the core dumps must be copied out first. A dead run whose `rc` was never written is a zombie:
# it looks RUNNING to the pool and is never restarted, so it must alert too.
#
# Usage (detached):  setsid nohup bash watch_main_crash.sh > /tmp/watch_main_crash.log 2>&1 &
# Options:           --interval 30   --budget 43200   --out <dir>   --stall-s 1800
#
# Exit codes: 0 budget spent, 2 no live run for > --stall-s, 3 output dir unusable,
#             4 failure signature in the log, 5 the run ended (rc present), 6 log silent (zombie).
set -uo pipefail

RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
# Both the old line and its fixed successor write this tree; after the handover the live run is
# ambitious_v4b_learned_fix_256, so match the common prefix rather than one exact name.
NAME_PREFIX=ambitious_v4b_learned_
INTERVAL=30
BUDGET=43200
STALL_S=1800
OUT=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/v4b_crash_evidence

# Every failure class observed in this tree. One line is enough to alert.
# A rendezvous keep-alive line is NOT here on purpose: it appears at teardown of every failed or
# cancelled run (pjob .2407, .2431), so alerting on it fires on symptoms rather than causes. The
# real store failure (DistNetworkError / recvValueWithTimeout) is here.
FAIL_RE='cudaErrorIllegalAddress|illegal memory access|CUDA error: an illegal|oom_kill events|Out of memory|Killed process|DistNetworkError|recvValueWithTimeout|disagrees with topk|Unexpected result nan|found NaN|Requested node configuration is not available|FileNotFoundError'

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --stall-s) STALL_S=$2; shift 2;;
    -h|--help) sed -n '2,18p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 3;;
  esac
done

mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 3; }
echo "[$(date +%H:%M:%S)] watching ${NAME_PREFIX}* for illegal access / OOM / rendezvous / trap / NaN / refused step; evidence -> $OUT"

START=$(date +%s)
last_rid=""
last_size=0
last_progress=$(date +%s)

resolve_rid() {
  # The run id changes on every restart: resolve the live one by name, skipping dirs with rc.
  /usr/bin/env python3 - "$POOL" "$NAME_PREFIX" <<'PY'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
best = None
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if not str(m.get("name") or "").startswith(name):
        continue
    rid = int(d.rsplit(".", 1)[-1])
    best = rid if best is None or rid > best else best
print(best if best is not None else "")
PY
}

# Copy the log and every artifact that explains the failure, then print the summary.
capture() {  # $1=rid  $2=log  $3=reason  $4=from_byte
  local rid=$1 log=$2 reason=$3 from=$4
  local ts ev
  ts=$(date +%Y%m%dT%H%M%S)
  ev="$OUT/fail_${ts}_rid${rid}"
  mkdir -p "$ev"
  cp -f "$log" "$ev/log_full.txt" 2>/dev/null
  {
    echo "run_id: .$rid"
    echo "reason: $reason"
    echo "log: $log"
    echo "detected: $(date -Is)"
    echo
    echo "== matched lines (3 lines of context) =="
    tail -c +"$from" "$log" | grep -nE -A3 -B3 "$FAIL_RE" | head -120
    echo
    echo "== first failing rank / node =="
    tail -c +"$from" "$log" | grep -oE "rank [0-9]+|as[0-9]{2}r[0-9]b[0-9]{2}" | head -10
    echo
    echo "== oom_kill / step errors =="
    grep -aE "oom_kill|error: Detected|srun: error" "$log" | tail -15
    echo
    echo "== core dumps written =="
    find "$RUN_DIR" -maxdepth 3 -name 'core*' -newermt '-2 hours' 2>/dev/null | head -10
  } > "$ev/summary.txt" 2>&1
  find "$RUN_DIR" -maxdepth 3 -name 'core*' -newermt '-2 hours' -exec cp -f {} "$ev/" \; 2>/dev/null
  echo "[$(date +%H:%M:%S)] ALERT $reason in .$rid -> $ev/summary.txt"
  head -25 "$ev/summary.txt"
}

# A pjob that ended between two polls is invisible to resolve_rid (it skips dirs with rc), so its
# end must be reported here or a failure would pass silently (pjob .2408 ended rc=140 unwatched).
report_end() {  # $1=rid
  local rid=$1 rc log
  [ -f "$POOL/runs/mn5-pool-p264.$rid/rc" ] || return 0
  rc=$(cat "$POOL/runs/mn5-pool-p264.$rid/rc" 2>/dev/null)
  log="$RUN_DIR/slurm-mn5-pool-p264.$rid.log"
  # The pool writes "<code> cancelled" / "<code> preempted" for a deliberate stop, and "<code> failed"
  # otherwise. A cancelled run (rc 0, or 137 after a strong kill) is not a failure.
  case "$rc" in
    *cancelled*|*preempted*) echo "[$(date +%H:%M:%S)] .$rid ended rc=$rc (deliberate stop)"; return 0;;
  esac
  if [ "$rc" = "0" ]; then
    echo "[$(date +%H:%M:%S)] .$rid ended rc=0 (stopped, not a failure)"
    return 0
  fi
  capture "$rid" "$log" "pjob ended rc=$rc" 1
  exit 5
}

while :; do
  now=$(date +%s)
  if [ $((now - START)) -ge "$BUDGET" ]; then
    echo "[$(date +%H:%M:%S)] budget spent"
    exit 0
  fi

  rid=$(resolve_rid)
  if [ -z "$rid" ]; then
    [ -n "$last_rid" ] && { report_end "$last_rid"; last_rid=""; }
    if [ $((now - last_progress)) -ge "$STALL_S" ]; then
      echo "[$(date +%H:%M:%S)] ALERT no live ${NAME_PREFIX}* run for $(( (now-last_progress)/60 )) min"
      exit 2
    fi
    sleep "$INTERVAL"
    continue
  fi

  if [ "$rid" != "$last_rid" ]; then
    if [ -n "$last_rid" ]; then
      echo "[$(date +%H:%M:%S)] RESTART .$last_rid -> .$rid"
      report_end "$last_rid"
    fi
    last_rid=$rid
    last_size=0
    last_progress=$now
  fi

  log="$RUN_DIR/slurm-mn5-pool-p264.$rid.log"
  if [ ! -f "$log" ]; then
    sleep "$INTERVAL"
    continue
  fi

  size=$(stat -c %s "$log" 2>/dev/null || echo 0)
  if [ "$size" -gt "$last_size" ]; then
    last_progress=$now
    # Scan only the new bytes, but keep one line of overlap for a split match.
    from=$(( last_size > 200 ? last_size - 200 : 1 ))
    hits=$(tail -c +"$from" "$log" | grep -E "$FAIL_RE" | head -3 || true)
    if [ -n "$hits" ]; then
      capture "$rid" "$log" "failure signature: $(printf '%s' "$hits" | head -1 | cut -c1-120)" "$from"
      exit 4
    fi
    last_size=$size
  fi

  # A run whose `rc` appeared has ended: the pool restarts it, but say so with the exit code and
  # the last error, because rc=0 (cancelled) and rc=1 (failed) mean very different things.
  if [ -f "$POOL/runs/mn5-pool-p264.$rid/rc" ]; then
    rc=$(cat "$POOL/runs/mn5-pool-p264.$rid/rc" 2>/dev/null)
    echo "[$(date +%H:%M:%S)] ALERT .$rid ended rc=$rc"
    tail -5 "$log" | cut -c1-160
    exit 5
  fi

  # A zombie: no rc, log silent past --stall-s. The pool still sees a busy host and never
  # restarts, so this must wake the operator (pjob .2407 was exactly this).
  if [ $((now - last_progress)) -ge "$STALL_S" ]; then
    capture "$rid" "$log" "log silent $(( (now-last_progress)/60 )) min, no rc (zombie)" "$(( size > 200 ? size - 200 : 1 ))"
    exit 6
  fi

  sleep "$INTERVAL"
done
