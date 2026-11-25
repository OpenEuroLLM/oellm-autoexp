#!/usr/bin/env bash
# Activate --nan-trace at the next NaN.
#
# Why: --nan-trace (megatron/training/nan_trace.py) takes effect from the next record enqueued from
# its config, and the live record's argv is fixed at enqueue time. Waiting for a NaN makes the swap
# free: the run has just died, so the ~16 min init happens anyway. Doing it now would spend a restart
# for nothing.
#
# Sequence at the NaN (the only order that leaves exactly one live record):
#   1. pool_ctl cancel <job>        -- cancel the RECORD, not the run (a cancelled run is restarted)
#   2. wait for no live run of that name (the monitor executes the cancel on its next poll)
#   3. run_autoexp --config-name <trace-enabled config> --yes
#   4. wait for a new live pjob, then print what it is running
#
# Never submits, cancels, requeues or resizes a SLURM job: everything here is pool state.
#
# Usage (detached):
#   setsid --fork nohup bash enable_nan_trace_on_nan.sh > $R/nan_trace_switch.log 2>&1 < /dev/null
#   bash enable_nan_trace_on_nan.sh --dry-run
#
# Exit codes: 0 swapped, 2 budget spent, 4 swap failed, 5 the run was cancelled by someone else.
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
SNAP=$R/autoexp_v4bfix2
PY=$R/autoexp_env/bin/python
NAME=ambitious_v4b_learned_fix_256
CONFIG=experiments/korbi/recall_surgery/v4b_learned_fix_lowmem_mn5
INTERVAL=30
BUDGET=43200
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 4;;
  esac
done

say() { echo "[$(date +%H:%M:%S)] $*"; }

pool_ctl() { ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$POOL" "$@" ); }

live_runs() {  # live pjobs of NAME, newest first
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

newest_rid() {  # newest pjob of NAME, live or ended (its log is what carries the NaN)
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

say "waiting for the next NaN of $NAME (config $CONFIG)"
START=$(date +%s)
seen_rid=""
seen_size=0

while :; do
  now=$(date +%s)
  if [ $((now - START)) -ge "$BUDGET" ]; then
    say "budget spent without a NaN"
    exit 2
  fi

  rid=$(newest_rid)
  if [ -n "$rid" ] && [ "$rid" != "$seen_rid" ]; then
    [ -n "$seen_rid" ] && say "new pjob .$seen_rid -> .$rid"
    seen_rid=$rid
    seen_size=0
  fi

  log="$RUN_DIR/slurm-mn5-pool-p264.$rid.log"
  if [ -n "$rid" ] && [ -f "$log" ]; then
    size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$size" -gt "$seen_size" ]; then
      from=$(( seen_size > 400 ? seen_size - 400 : 1 ))
      if tail -c +"$from" "$log" | grep -q "Unexpected result nan"; then
        hit=$(tail -c +"$from" "$log" | grep -m1 "Unexpected result nan" | cut -c1-150)
        say "TRIGGER NaN in .$rid: $hit"
        if [ "$DRY" = 1 ]; then
          say "DRY RUN: would cancel the record of $NAME and enqueue $CONFIG"
          exit 0
        fi

        # 1. cancel the RECORD, retried: the monitor executes it on its next poll
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

        # 2. enqueue the trace-enabled config under the same job name (no rename)
        say "enqueue $CONFIG"
        ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/run_autoexp.py --pool "$POOL" \
            --config-name "$CONFIG" --yes 2>&1 | tail -4 | sed 's/^/    /' )

        # 3. wait for the new pjob and report what it carries
        for _ in $(seq 1 30); do
          sleep 10
          new=$(live_runs | awk '{print $1}')
          [ -n "$new" ] && break
        done
        if [ -z "${new:-}" ]; then
          say "FAILED: enqueued but no live run appeared"
          exit 4
        fi
        step="$POOL/runs/mn5-pool-p264.$new/step.sh"
        say "new pjob .$new; --nan-trace in its step script: $(grep -c -- '--nan-trace' "$step" 2>/dev/null)"
        say "SWAPPED: .$new runs $CONFIG with --nan-trace"
        exit 0
      fi
      seen_size=$size
    fi
  fi
  sleep "$INTERVAL"
done
