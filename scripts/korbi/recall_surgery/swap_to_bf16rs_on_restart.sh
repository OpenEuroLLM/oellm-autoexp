#!/usr/bin/env bash
# Swap the v4b main run onto the bf16 expert-gradient reduce-scatter config at the NEXT NATURAL
# RESTART -- never now.
#
# Operator decision (2026-11-25): the precision change is affordable while everything else is
# monitored, but the live run must not be interrupted for it. So this waits for the run to die on its
# own (a NaN, an OOM, any rc != 0 that is not a deliberate stop) and only then replaces the pool
# record, which costs nothing extra because the pool was going to restart the run anyway.
#
# The swap: cancel the RECORD, wait for no live run, wait for the cancel request to be consumed
# (requests/ holds only done/), then enqueue
#   config  experiments/korbi/recall_surgery/v4b_learned_fix_bf16rs_mn5
#   from    snapshot autoexp_v4bbf16   (the snapshot that carries the config and the bf16 flags)
# The barrier on the cancel request matters: on 2026-11-25 a swap enqueued while a cancel request was
# still pending, and the request killed the fresh run one minute in (pjob .2458, rc=143).
#
# Everything here is pool state. No SLURM job is submitted, cancelled, requeued or resized.
#
# Usage (detached):
#   setsid --fork nohup bash swap_to_bf16rs_on_restart.sh > $R/bf16rs_swap.log 2>&1 < /dev/null
#   bash swap_to_bf16rs_on_restart.sh --dry-run
#
# Exit codes: 0 swapped, 2 budget spent, 4 swap failed.
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
SNAP=$R/autoexp_v4bbf16
PY=$R/autoexp_env/bin/python
NAME=ambitious_v4b_learned_fix_256
CONFIG=experiments/korbi/recall_surgery/v4b_learned_fix_bf16rs_mn5
INTERVAL=30
BUDGET=43200
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 4;;
  esac
done

say() { echo "[$(date +%H:%M:%S)] $*"; }
pool_ctl() { ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$POOL" "$@" ); }

live_runs() {
  "$PY" - "$POOL" "$NAME" <<'PY'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
out = []
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if str(m.get("name") or "") == name:
        out.append(int(d.rsplit(".", 1)[-1]))
print(" ".join(str(r) for r in sorted(out, reverse=True)))
PY
}

newest_rid() {
  "$PY" - "$POOL" "$NAME" <<'PY'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
best = None
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if str(m.get("name") or "") != name:
        continue
    rid = int(d.rsplit(".", 1)[-1])
    best = rid if best is None or rid > best else best
print(best if best is not None else "")
PY
}

pending_requests() { ls -1 "$POOL"/requests/*.json 2>/dev/null | wc -l; }

say "waiting for a NATURAL restart of $NAME; then swap to $CONFIG from $(basename "$SNAP")"
START=$(date +%s)
seen_rid=""
seen_size=0

while :; do
  now=$(date +%s)
  if [ $((now - START)) -ge "$BUDGET" ]; then
    say "budget spent without a restart"
    exit 2
  fi

  rid=$(newest_rid)
  if [ -n "$rid" ] && [ "$rid" != "$seen_rid" ]; then
    [ -n "$seen_rid" ] && say "new pjob .$seen_rid -> .$rid"
    seen_rid=$rid
    seen_size=0
  fi

  if [ -n "$rid" ]; then
    log="$RUN_DIR/slurm-mn5-pool-p264.$rid.log"
    rcfile="$POOL/runs/mn5-pool-p264.$rid/rc"
    trigger=""

    # (a) a NaN or another failure signature in the log of the newest pjob
    if [ -f "$log" ]; then
      size=$(stat -c %s "$log" 2>/dev/null || echo 0)
      if [ "$size" -gt "$seen_size" ]; then
        from=$(( seen_size > 400 ? seen_size - 400 : 1 ))
        hit=$(tail -c +"$from" "$log" | grep -m1 -E "Unexpected result nan|oom_kill events|illegal memory access|disagrees with topk" || true)
        [ -n "$hit" ] && trigger="signature: $(printf '%s' "$hit" | cut -c1-110)"
        seen_size=$size
      fi
    fi

    # (b) the pjob ended on its own (a deliberate cancel/preempt is not a natural restart)
    if [ -z "$trigger" ] && [ -f "$rcfile" ]; then
      rc=$(cat "$rcfile" 2>/dev/null)
      case "$rc" in
        *cancelled*|*preempted*) ;;
        *) trigger="pjob ended rc=$rc";;
      esac
    fi

    if [ -n "$trigger" ]; then
      say "TRIGGER in .$rid: $trigger"
      if [ "$DRY" = 1 ]; then
        say "DRY RUN: would cancel the record of $NAME and enqueue $CONFIG from $(basename "$SNAP")"
        exit 0
      fi

      n=0
      while :; do
        n=$((n + 1))
        [ "$n" -gt 6 ] && { say "FAILED: record still live after 6 cancels: $(live_runs)"; exit 4; }
        say "cancel record of $NAME (attempt $n)"
        pool_ctl cancel "$NAME" 2>&1 | sed 's/^/    /'
        for _ in $(seq 1 18); do
          sleep 10
          [ -z "$(live_runs)" ] && break
        done
        [ -z "$(live_runs)" ] && { say "no live run of $NAME"; break; }
        say "  still live: $(live_runs)"
      done

      # barrier: the cancel request must be consumed before a new record is enqueued, or it kills it
      for _ in $(seq 1 24); do
        [ "$(pending_requests)" -eq 0 ] && { say "cancel request consumed"; break; }
        sleep 5
      done
      [ "$(pending_requests)" -eq 0 ] || { say "FAILED: requests still pending after 120 s"; exit 4; }

      say "enqueue $CONFIG"
      ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/run_autoexp.py --pool "$POOL" \
          --config-name "$CONFIG" --yes 2>&1 | tail -3 | sed 's/^/    /' )

      new=""
      for _ in $(seq 1 30); do
        sleep 10
        new=$(live_runs | awk '{print $1}')
        [ -n "$new" ] && break
      done
      if [ -z "$new" ]; then
        say "FAILED: enqueued but no live run appeared"
        exit 4
      fi
      step="$POOL/runs/mn5-pool-p264.$new/step.sh"
      say "new pjob .$new; bf16 grad-comm flags in its step script:"
      grep -oE -- "--megatron-fsdp-grad-comm-dtype[a-z-]*[= ]?[a-z0-9]*" "$step" 2>/dev/null | sed 's/^/    /'
      say "SWAPPED: .$new runs $CONFIG"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
