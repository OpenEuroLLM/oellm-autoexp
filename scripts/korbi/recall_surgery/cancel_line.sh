#!/usr/bin/env bash
# Finally cancel a pool line: its job records AND every restart they spawn.
#
# Why this exists: `pool_ctl cancel <name>` cancels ONE record, and a line can have several live
# records under the same definition name (the pool renames duplicates _q2, _q3, ... and each restart
# of a record can produce another). Cancelling a *run* cancels nothing durable -- the record simply
# starts it again on the next poll. So the only reliable stop is: cancel every live record whose
# definition name matches, by record file stem, in rounds, until none is left and no run of the line
# is unfinished.
#
# Usage: bash cancel_line.sh <pool-dir> <definition-name> [--rounds 8] [--dry-run]
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
PY=$R/autoexp_env/bin/python
CTL_DIR=$R/autoexp_v4bfix2

POOL=${1:-}; NAME=${2:-}
[ -n "$POOL" ] && [ -n "$NAME" ] || { echo "usage: $0 <pool-dir> <definition-name> [--rounds N] [--dry-run]" >&2; exit 2; }
shift 2
ROUNDS=8
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds) ROUNDS=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

STATE=$(ls -d "$R"/pool_state/pool_* 2>/dev/null | while read -r d; do
  [ -f "$d/$(basename "$POOL")".* ] 2>/dev/null; done)
# the session dir is named after the pool, not the path
SESSION="$R/pool_state/pool_$(basename "$POOL" | tr '_' '-')"
[ -d "$SESSION" ] || SESSION="$R/pool_state/pool_$(basename "$POOL")"

say() { echo "[$(date +%H:%M:%S)] $*"; }
ctl() { ( cd "$CTL_DIR" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$POOL" "$@" 2>&1 | tail -2 | sed 's/^/    /' ); }

live_records() {  # record file stems whose definition name is $NAME and which are not finished
  "$PY" - "$SESSION" "$NAME" <<'PY'
import glob, json, os, sys
session, name = sys.argv[1], sys.argv[2]
for p in sorted(glob.glob(os.path.join(session, "*.job.json"))):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if ((d.get("definition") or {}).get("name")) != name:
        continue
    if (d.get("runtime") or {}).get("final_state") is None:
        print(os.path.basename(p)[:-len(".job.json")])
PY
}

live_runs() {
  "$PY" - "$POOL" "$NAME" <<'PY'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
for d in sorted(glob.glob(os.path.join(pool, "runs", "*"))):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if m.get("name") == name:
        print(os.path.basename(d))
PY
}

say "cancelling line '$NAME' in $(basename "$POOL")  (session $SESSION)"
for round in $(seq 1 "$ROUNDS"); do
  recs=$(live_records)
  runs=$(live_runs)
  [ -z "$recs" ] && [ -z "$runs" ] && { say "clean: no live record and no live run"; exit 0; }
  say "round $round: records=[$(echo $recs | tr '\n' ' ')] runs=[$(echo $runs | tr '\n' ' ')]"
  for rec in $recs; do
    if [ "$DRY" = 1 ]; then echo "    would cancel record $rec"; else ctl cancel "$rec"; fi
  done
  for run in $runs; do
    if [ "$DRY" = 1 ]; then echo "    would cancel run $run"; else ctl cancel "$run"; fi
  done
  [ "$DRY" = 1 ] && exit 0
  sleep 35   # one monitor poll, so the cancels land before the next round
done

recs=$(live_records); runs=$(live_runs)
if [ -z "$recs" ] && [ -z "$runs" ]; then say "clean"; exit 0; fi
say "STILL LIVE after $ROUNDS rounds: records=[$(echo $recs | tr '\n' ' ')] runs=[$(echo $runs | tr '\n' ' ')]"
exit 5
