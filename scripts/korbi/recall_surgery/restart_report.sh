#!/usr/bin/env bash
# After a planned restart: verify the excluded node is gone, time the startup, and report throughput.
#
# One restart, one report, then no more interventions. Exits 0 with a summary (the harness
# background task's notification carries it) or non-zero if the new run does not appear.
#
# Usage: bash restart_report.sh [--after 2186] [--need 20] [--timeout-min 120]
set -uo pipefail

RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
NODE=as02r1b08

AFTER=0
NEED=20
TIMEOUT_MIN=120
while [ $# -gt 0 ]; do
  case "$1" in
    --after) AFTER=$2; shift 2;;
    --need) NEED=$2; shift 2;;
    --timeout-min) TIMEOUT_MIN=$2; shift 2;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

newest_rid() {
  /usr/bin/env python3 - "$POOL" <<'PY'
import glob, json, os, sys
best = None
for d in glob.glob(sys.argv[1] + "/runs/mn5-pool-p264.*"):
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if not str(m.get("name") or "").startswith("ambitious_v4b_learned_"):
        continue
    rid = int(d.rsplit(".", 1)[-1])
    if best is None or rid > best:
        best = rid
print(best if best is not None else 0)
PY
}

deadline=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
echo "[$(date +%H:%M:%S)] waiting for a run newer than .$AFTER"

rid=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  r=$(newest_rid)
  if [ "$r" -gt "$AFTER" ] && [ -f "$POOL/runs/mn5-pool-p264.$r/started" ]; then rid=$r; break; fi
  sleep 20
done
if [ -z "$rid" ]; then echo "ALERT no new run started within ${TIMEOUT_MIN} min"; exit 5; fi

d=$POOL/runs/mn5-pool-p264.$rid
echo "[$(date +%H:%M:%S)] new run .$rid started $(stat -c %y "$d/started" | cut -d. -f1)"
if grep -q "$NODE" "$d/meta.env" 2>/dev/null; then
  echo "  $NODE: STILL IN THE RUN's EXCLUDE LIST -> check the render"
else
  echo "  $NODE: not in the run's EXCLUDE (dropped as intended)"
fi
echo "  build: $(grep -oE "pylibs_moonep_[a-z0-9]+" "$d/step.sh" 2>/dev/null | head -1)  recipe: sparse=$(grep -c "page-sparse-backend" "$d/step.sh") packing=$(grep -c "packing-plan-path" "$d/step.sh") bundle=$(grep -c "dataloader-index-bundle" "$d/step.sh")"

log=$RUN_DIR/slurm-mn5-pool-p264.$rid.log
while [ "$(date +%s)" -lt "$deadline" ]; do
  n=$(grep -cE "elapsed time per iteration \(ms\)" "$log" 2>/dev/null | head -1); n=${n:-0}
  [ "$n" -ge "$NEED" ] && break
  if [ -f "$d/rc" ]; then echo "ALERT .$rid ended rc=$(cat "$d/rc") before $NEED steps"; exit 6; fi
  sleep 60
done

/usr/bin/env python3 - "$log" "$d/started" <<'PY'
import os, re, statistics, sys, datetime

log, started = sys.argv[1], sys.argv[2]
first_t = first_it = last_t = last_it = None
for line in open(log, errors="ignore"):
    m = re.search(r"\[(\d{4}-\d\d-\d\d) (\d\d):(\d\d):(\d\d)\.\d+\] iteration +(\d+)/", line)
    if not m:
        continue
    t = datetime.datetime.strptime(
        "%s %s:%s:%s" % (m.group(1), m.group(2), m.group(3), m.group(4)), "%Y-%m-%d %H:%M:%S")
    if first_t is None:
        first_t, first_it = t, int(m.group(5))
    last_t, last_it = t, int(m.group(5))

steps = [float(m.group(1)) / 1000.0
         for m in re.finditer(r"elapsed time per iteration \(ms\): ([0-9.]+)", open(log, errors="ignore").read())]
tail = sorted(steps[len(steps) // 4:] or steps)
med = statistics.median(tail)
startup = (first_t.timestamp() - os.path.getmtime(started)) if first_t else float("nan")
wall = (last_t - first_t).total_seconds() / max(last_it - first_it, 1)
print("startup: %.1f min (step start -> first iteration)" % (startup / 60.0))
print("steps: %d (iter %d -> %d)   per-iteration median %.2f s" % (len(steps), first_it, last_it, med))
print("wall rate: %.2f s/step -> %.0f tok/s/GPU effective (65536 / wall)" % (wall, 65536.0 / wall))
PY

exit 0
