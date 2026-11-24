#!/usr/bin/env bash
# Report the throughput of the live v4b run once it has stepped a few times.
#
# One step trains gbs 1024 samples x 65536 tokens = 67,108,864 tokens on 256 nodes x 4 GPUs =
# 1024 GPUs, so tokens/s/GPU = 65536 / step_seconds (18.34 s -> 3573, the 3.6k regime; 20.7 s ->
# 3166). Prints a compact summary and exits 0; meant for a harness background task, whose
# completion notification carries the number.
#
# Usage: bash report_step_time.sh [--need 20] [--timeout-min 90]
set -uo pipefail

RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
NAME_PREFIX=ambitious_v4b_learned_

NEED=20
TIMEOUT_MIN=90
while [ $# -gt 0 ]; do
  case "$1" in
    --need) NEED=$2; shift 2;;
    --timeout-min) TIMEOUT_MIN=$2; shift 2;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

deadline=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
echo "[$(date +%H:%M:%S)] waiting for $NEED timed steps (timeout ${TIMEOUT_MIN} min)"

while :; do
  log=$(ls -1t "$RUN_DIR"/slurm-mn5-pool-p264.*.log 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    n=$(grep -cE "elapsed time per iteration \(ms\)" "$log" 2>/dev/null | head -1)
    n=${n:-0}
    if [ "$n" -ge "$NEED" ]; then
      /usr/bin/env python3 - "$log" <<'PY'
import re, statistics, sys
ms = [float(m.group(1)) / 1000.0
      for m in re.finditer(r"elapsed time per iteration \(ms\): ([0-9.]+)", open(sys.argv[1], errors="ignore").read())]
tail = ms[len(ms) // 4:] or ms          # drop warmup / compile
tail_sorted = sorted(tail)
med = statistics.median(tail_sorted)
p10 = tail_sorted[len(tail_sorted) // 10]
p90 = tail_sorted[9 * len(tail_sorted) // 10]
print("steps: %d   median %.2f s   p10 %.2f   p90 %.2f   max %.2f"
      % (len(ms), med, p10, p90, tail_sorted[-1]))
print("throughput: %.0f tok/s/GPU at the median (3.6k regime = 18.34 s, 3.2k = 20.7 s)" % (65536.0 / med))
PY
      grep -m1 -oE "iteration +[0-9]+/" "$log" | sed 's/^/first /'
      grep -E "iteration +[0-9]+/" "$log" | tail -1 | cut -c1-90 | sed 's/^/last  /'
      echo "log: $(basename "$log")"
      exit 0
    fi
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[$(date +%H:%M:%S)] timeout: only ${n:-0} timed step(s) after ${TIMEOUT_MIN} min"
    echo "log: $(basename "${log:-none}")"
    grep -m1 -iE "Loading from|loading checkpoint from|Error|Traceback" "${log:-/dev/null}" 2>/dev/null | cut -c1-140
    exit 0
  fi
  sleep 60
done
