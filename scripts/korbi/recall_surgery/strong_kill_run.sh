#!/usr/bin/env bash
# Strong stop: end a pool run's SLURM step immediately, with no grace period.
#
# Policy (operator, 2026-11-24): a run that has just (re)started holds no new checkpoint, so the
# pool's graceful stop is pure waiting -- STOP_GRACE_S / PREEMPT_GRACE_S (120 / 900 s) exist to let
# a run reach its next save, and a run with zero iterations cannot. For those, kill the step.
#
# What it does: finds the step whose name is the run id (`squeue -s -j <sjob>`) and sends it
# SIGKILL. It never touches the SLURM job, only a step of an allocation we already hold, which is
# the same object the supervisor signals for a normal stop -- only without the grace.
#
# The pool then sees `rc` appear (137 cancelled / failed) and restarts or holds the run according to
# its record. To stop the job for good, cancel the RECORD as well:
#   pool_ctl cancel <job-name>            (a cancelled run alone is restarted on the next poll)
#
# Usage:
#   bash strong_kill_run.sh                       # newest live run of ambitious_v4b_learned_*
#   bash strong_kill_run.sh <rid>                 # e.g. 2432 or mn5-pool-p264.2432
#   bash strong_kill_run.sh --name <prefix>       # newest live run whose name starts with prefix
#   bash strong_kill_run.sh --dry-run
#
# Exit codes: 0 killed, 2 no live run, 3 no step found, 4 SLURM refused the kill.
set -uo pipefail

POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
NAME_PREFIX=ambitious_v4b_learned_
RID=""
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME_PREFIX=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *) RID=${1##*.}; shift;;
  esac
done

newest_live() {  # $1=name prefix
  /usr/bin/env python3 - "$POOL" "$1" <<'PY'
import glob, json, os, sys
pool, prefix = sys.argv[1], sys.argv[2]
best = None
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if not str(m.get("name") or "").startswith(prefix):
        continue
    rid = int(d.rsplit(".", 1)[-1])
    best = rid if best is None or rid > best else best
print(best if best is not None else "")
PY
}

[ -n "$RID" ] || RID=$(newest_live "$NAME_PREFIX")
[ -n "$RID" ] || { echo "no live run for prefix $NAME_PREFIX"; exit 2; }

SJOB=$(cat "$POOL/pool.json" 2>/dev/null | /usr/bin/env python3 -c 'import json,sys; print(json.load(sys.stdin).get("slurm_job_id") or "")' 2>/dev/null)
[ -n "$SJOB" ] || SJOB=$(squeue -h -u "$USER" -o "%i %j" 2>/dev/null | awk '$2 ~ /mn5_pool_p264/ {print $1; exit}')
[ -n "$SJOB" ] || { echo "cannot resolve the SLURM job of the pool"; exit 3; }

STEP=$(squeue -h -s -j "$SJOB" -o "%i %j" 2>/dev/null | awk -v n="mn5-pool-p264.$RID" '$2==n {print $1; exit}')
[ -n "$STEP" ] || { echo "no step for run .$RID in job $SJOB (already gone?)"; exit 3; }

ITERS=$(grep -cE "iteration +[0-9]+/" "$RUN_DIR/slurm-mn5-pool-p264.$RID.log" 2>/dev/null || echo 0)
echo "run .$RID  step $STEP  iterations so far: ${ITERS:-0}  (no new checkpoint if 0)"

if [ "$DRY" = 1 ]; then
  echo "DRY RUN: would scancel --signal=KILL $STEP"
  exit 0
fi

scancel --signal=KILL "$STEP" 2>&1 | head -2
for _ in $(seq 1 12); do
  sleep 5
  [ -f "$POOL/runs/mn5-pool-p264.$RID/rc" ] && { echo "ended: $(cat "$POOL/runs/mn5-pool-p264.$RID/rc")"; exit 0; }
  squeue -h -s -j "$SJOB" -o "%i" 2>/dev/null | grep -qx "$STEP" || { echo "step $STEP gone (rc not written yet)"; exit 0; }
done
echo "step $STEP still present after 60 s" >&2
exit 4
