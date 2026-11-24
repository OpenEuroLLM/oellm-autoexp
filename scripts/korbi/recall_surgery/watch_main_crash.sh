#!/usr/bin/env bash
# Watch the v4b main run for the cudaErrorIllegalAddress crash and preserve the evidence.
#
# Why: the crash is non-deterministic (~1 per 200-400 steps at 256 nodes), reported only
# asynchronously by the NCCL watchdog, and the pool restarts the run within a minute.
# The log line and the lightweight core dump are the only artifacts, so they must be
# copied out before the pool prunes the run directory.
#
# Usage (detached):  setsid nohup bash watch_main_crash.sh > /tmp/watch_main_crash.log 2>&1 &
# Options:           --interval 60   --budget 86400   --out <dir>
#
# Exit codes: 0 budget spent, 2 no live run for > --stall-s, 3 output dir unusable.
set -uo pipefail

RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
NAME=ambitious_v4b_learned_256
INTERVAL=60
BUDGET=86400
STALL_S=1800
OUT=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/v4b_crash_evidence

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --stall-s) STALL_S=$2; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 3;;
  esac
done

mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 3; }
echo "[$(date +%H:%M:%S)] watching $NAME for illegal-access crashes; evidence -> $OUT"

START=$(date +%s)
last_rid=""
last_size=0
last_progress=$(date +%s)
declare -A seen=()

resolve_rid() {
  # The run id changes on every restart: resolve the live one by name, skipping dirs with rc.
  /usr/bin/env python3 - "$POOL" "$NAME" <<'PY'
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
    if m.get("name") != name:
        continue
    rid = int(d.rsplit(".", 1)[-1])
    best = rid if best is None or rid > best else best
print(best if best is not None else "")
PY
}

while :; do
  now=$(date +%s)
  if [ $((now - START)) -ge "$BUDGET" ]; then
    echo "[$(date +%H:%M:%S)] budget spent"
    exit 0
  fi

  rid=$(resolve_rid)
  if [ -z "$rid" ]; then
    if [ $((now - last_progress)) -ge "$STALL_S" ]; then
      echo "[$(date +%H:%M:%S)] ALERT no live $NAME run for $(( (now-last_progress)/60 )) min"
      exit 2
    fi
    sleep "$INTERVAL"
    continue
  fi

  if [ "$rid" != "$last_rid" ]; then
    if [ -n "$last_rid" ]; then
      echo "[$(date +%H:%M:%S)] RESTART .$last_rid -> .$rid"
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
    hits=$(tail -c +$(( last_size > 200 ? last_size - 200 : 1 )) "$log" \
           | grep -nE "cudaErrorIllegalAddress|illegal memory access|CUDA error: an illegal" || true)
    if [ -n "$hits" ]; then
      ts=$(date +%Y%m%dT%H%M%S)
      ev="$OUT/crash_${ts}_rid${rid}"
      mkdir -p "$ev"
      cp -f "$log" "$ev/log_full.txt" 2>/dev/null
      {
        echo "run_id: .$rid"
        echo "log: $log"
        echo "detected: $(date -Is)"
        echo
        echo "== matched lines (with 3 lines of context, deduplicated) =="
        tail -c +$(( last_size > 200 ? last_size - 200 : 1 )) "$log" \
          | grep -nE -A3 -B3 "cudaErrorIllegalAddress|illegal memory access|CUDA error: an illegal" \
          | head -120
        echo
        echo "== first failing rank / node =="
        tail -c +$(( last_size > 200 ? last_size - 200 : 1 )) "$log" \
          | grep -oE "rank [0-9]+|as[0-9]{2}r[0-9]b[0-9]{2}" | head -10
        echo
        echo "== core dumps written =="
        find "$RUN_DIR" -maxdepth 3 -name 'core*' -newermt '-2 hours' 2>/dev/null | head -10
      } > "$ev/summary.txt" 2>&1
      # Copy the lightweight core dumps (they live under the run dir or TMPDIR).
      find "$RUN_DIR" -maxdepth 3 -name 'core*' -newermt '-2 hours' -exec cp -f {} "$ev/" \; 2>/dev/null
      echo "[$(date +%H:%M:%S)] ALERT illegal access in .$rid -> $ev/summary.txt"
      cat "$ev/summary.txt" | head -25
      exit 4
    fi
    last_size=$size
  fi

  sleep "$INTERVAL"
done
