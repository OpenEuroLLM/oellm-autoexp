#!/usr/bin/env bash
# Report the c136 64-node speed A/B (ab2_base_t1 vs ab2_fix_t1) when both arms finish.
#
# The arms were queued by the speed workstream before the session limit hit: 64 nodes each from
# autoexp_speedopt5, resume of ckpt_eval 55000, 80 steps, paired over steps 35-80. base = the main
# run's settings, fix = +create_all_gather_group and +FSDP_RECOMPUTE_PREFETCH=1 (Megatron
# 4fc29013d + 78f31a524 + ad638dd56, bit-identical at 2 nodes, never measured at 64).
#
# This script only reads. Usage (detached):
#   setsid nohup bash watch_ab2_speed.sh > $R/ab2_speed_watch.log 2>&1 &
set -uo pipefail

POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_c136
INTERVAL=120
BUDGET=21600

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

say() { echo "[$(date +%H:%M:%S)] $*"; }
START=$(date +%s)
say "watching ab2_base_t1 / ab2_fix_t1 in $(basename $POOL)"

latest_arm() {  # newest run dir id for an arm name (finished or not)
  /usr/bin/env python3 - "$POOL" "$1" <<'PY'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
best = None
for d in glob.glob(pool + "/runs/mn5-pool-c136.*"):
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

log_of() { grep -oE "^LOG=.*" "$POOL/runs/mn5-pool-c136.$1/meta.env" 2>/dev/null | cut -d= -f2-; }

step_stats() {  # median / min / max of the per-iteration ms from a Megatron log
  /usr/bin/env python3 - "$1" <<'PY'
import re, sys, statistics
ms = []
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", line)
    if m:
        ms.append(float(m.group(1)))
if not ms:
    print("no step lines")
else:
    # the first steps include warmup/compile; the paired window is the tail
    tail = ms[len(ms) // 3:] or ms
    print(f"n={len(ms)} median={statistics.median(tail):.0f} ms min={min(tail):.0f} max={max(tail):.0f}")
PY
}

while :; do
  if [ $(( $(date +%s) - START )) -ge "$BUDGET" ]; then say "budget spent"; exit 0; fi
  done_all=1
  for arm in ab2_base_t1 ab2_fix_t1; do
    rid=$(latest_arm "$arm")
    if [ -z "$rid" ]; then say "$arm: not enqueued yet"; done_all=0; continue; fi
    d=$POOL/runs/mn5-pool-c136.$rid
    if [ ! -f "$d/rc" ]; then
      st="queued"; [ -f "$d/started" ] && st="running"
      say "$arm .$rid $st $(cat $d/assigned 2>/dev/null)"
      done_all=0
    fi
  done
  if [ "$done_all" = 1 ]; then
    echo "===================== AB2 RESULT $(date -Is) ====================="
    for arm in ab2_base_t1 ab2_fix_t1; do
      rid=$(latest_arm "$arm"); d=$POOL/runs/mn5-pool-c136.$rid; log=$(log_of "$rid")
      echo "$arm .$rid rc=$(cat $d/rc 2>/dev/null) log=$(basename "${log:-none}")"
      [ -n "${log:-}" ] && [ -f "$log" ] && echo "  steps: $(step_stats "$log")"
      [ -n "${log:-}" ] && [ -f "$log" ] && \
        echo "  illegal-access lines: $(grep -cE 'cudaErrorIllegalAddress|illegal memory access' "$log")"
    done
    exit 0
  fi
  sleep "$INTERVAL"
done
