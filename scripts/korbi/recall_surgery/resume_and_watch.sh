#!/usr/bin/env bash
# Make sure the v4b main line is running the fixed, full recipe, then watch it.
#
# Meant to run as one harness background task (its exit code is the alert):
#   1. if nothing of ambitious_v4b_learned_* is live and no fix submission is pending, submit it;
#   2. hand over to watch_line.sh, which alerts on failure, restart churn or a stall.
#
# Usage: bash resume_and_watch.sh [--budget 43200] [--interval 30]
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
STATE=$R/pool_state/pool_mn5-pool-p264
SNAP=$R/autoexp_v4bfix2
PY=$R/autoexp_env/bin/python
WATCH=$SNAP/scripts/korbi/recall_surgery/watch_line.sh

BUDGET=43200
INTERVAL=30
WATCH_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --budget) BUDGET=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --auto-exclude) WATCH_ARGS+=(--auto-exclude); shift;;
    *) echo "unknown arg: $1" >&2; exit 9;;
  esac
done

live() {
  /usr/bin/env python3 - "$POOL" <<'PY'
import glob, json, os, sys
out = []
for d in glob.glob(sys.argv[1] + "/runs/mn5-pool-p264.*"):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if str(m.get("name") or "").startswith("ambitious_v4b_learned_"):
        out.append("%s:%s" % (d.rsplit(".", 1)[-1], m.get("name")))
print(" ".join(out))
PY
}

pending() {  # submitted but not yet placed fix submissions
  /usr/bin/env python3 - "$STATE" <<'PY'
import glob, json, os, sys
n = 0
for p in glob.glob(sys.argv[1] + "/*learned_fix_256*.job.json"):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    rt = d.get("runtime") or {}
    if rt.get("submitted") and rt.get("final_state") is None:
        n += 1
print(n)
PY
}

echo "[$(date +%H:%M:%S)] live: $(live)"
if [ -z "$(live)" ] && [ "$(pending)" = "0" ]; then
  echo "[$(date +%H:%M:%S)] nothing live and nothing pending: submitting the fixed, full-recipe run"
  ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/run_autoexp.py \
      --config-name experiments/korbi/recall_surgery/v4b_learned_fix_mn5 \
      --no-monitor --monitor-state-dir "$R/pool_state" --pool "$POOL" -y 2>&1 \
    | grep -E "Enqueued|ERROR|Traceback" | head -5 )
else
  echo "[$(date +%H:%M:%S)] leaving the pool alone: live='$(live)' pending=$(pending)"
fi

exec bash "$WATCH" --interval "$INTERVAL" --budget "$BUDGET" "${WATCH_ARGS[@]}"
