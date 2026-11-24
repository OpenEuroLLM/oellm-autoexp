#!/usr/bin/env bash
# Nuclear reset of one pool, then exactly one main run. Run as a single background task.
#
# 1. cancel every live RUN on the pool (evals included);
# 2. cancel every live RECORD (this is what stops the monitor from re-submitting -- a cancelled run
#    alone is simply restarted on the next poll);
# 3. verify the pool is empty of live work and the host is idle;
# 4. submit exactly ONE main run and wait for its `started` marker.
#
# The monitor keeps running: with no live record there is nothing for it to restart, so it only has
# to place the one submission.
#
# Usage: bash reset_pool_and_start.sh [--pool <dir>] [--name <definition>] [--rounds 5]
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
NAME=ambitious_v4b_learned_fix_256
PY=$R/autoexp_env/bin/python
SNAP=$R/autoexp_v4bfix2
SESSION=$R/pool_state/pool_mn5-pool-p264
RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
ROUNDS=5
while [ $# -gt 0 ]; do
  case "$1" in
    --pool) P=$2; shift 2;;
    --name) NAME=$2; shift 2;;
    --rounds) ROUNDS=$2; shift 2;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -d "$P" ] || { echo "no pool at $P"; exit 2; }
SESSION="$R/pool_state/pool_$(basename "$P" | tr '_' '-')"
[ -d "$SESSION" ] || SESSION="$R/pool_state/pool_$(basename "$P")"

say() { echo "[$(date +%H:%M:%S)] $*"; }
ctl() { ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$P" "$@" 2>&1 | tail -1 | sed 's/^/    /' ); }

all_live_runs() {
  "$PY" - "$P" <<'PY'
import glob, json, os, sys
for d in sorted(glob.glob(os.path.join(sys.argv[1], "runs", "*"))):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    print(os.path.basename(d))
PY
}
all_live_records() {
  "$PY" - "$SESSION" <<'PY'
import glob, json, os, sys
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.job.json"))):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if (d.get("runtime") or {}).get("final_state") is None:
        print(os.path.basename(p)[: -len(".job.json")])
PY
}
main_rid() {
  "$PY" - "$P" "$NAME" <<'PY'
import glob, json, os, sys
best = None
for d in glob.glob(os.path.join(sys.argv[1], "runs", "*")):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if m.get("name") == sys.argv[2]:
        r = int(d.rsplit(".", 1)[-1])
        best = r if best is None or r > best else best
print(best if best is not None else "")
PY
}

say "resetting $(basename "$P") — every run and every record"
for round in $(seq 1 "$ROUNDS"); do
  runs=$(all_live_runs); recs=$(all_live_records)
  if [ -z "$runs" ] && [ -z "$recs" ]; then say "round $round: pool is clean"; break; fi
  say "round $round: runs=[$(echo $runs | tr '\n' ' ')] records=[$(echo $recs | tr '\n' ' ')]"
  for r in $runs; do ctl cancel "$r"; done
  for r in $recs; do ctl cancel "$r"; done
  sleep 35
done

left_runs=$(all_live_runs); left_recs=$(all_live_records)
if [ -n "$left_runs" ] || [ -n "$left_recs" ]; then
  say "STILL LIVE: runs=[$(echo $left_runs | tr '\n' ' ')] records=[$(echo $left_recs | tr '\n' ' ')]"
  exit 5
fi
say "pool clean: no live run, no live record"
say "host state: $(cat "$P/hosts/host010/state" 2>/dev/null)"

say "submitting exactly one main run"
( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/run_autoexp.py \
    --config-name experiments/korbi/recall_surgery/v4b_learned_fix_mn5 \
    --no-monitor --monitor-state-dir "$R/pool_state" --pool "$P" -y 2>&1 \
  | grep -E "Enqueued|ERROR|Traceback" | head -3 | sed 's/^/    /' )

for i in $(seq 1 60); do
  rid=$(main_rid)
  if [ -n "$rid" ] && [ -f "$P/runs/mn5-pool-p264.$rid/started" ]; then
    say "MAIN RUN .$rid STARTED"
    say "  excludes: $(grep -oE '^EXCLUDE=.*' "$P/runs/mn5-pool-p264.$rid/meta.env" | cut -d= -f2 | tr ',' '\n' | grep -c .) entries"
    say "  host: $(cat "$P/hosts/host010/state")"
    say "  trackers: persistent $(cat "$RUN_DIR/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null) rolling $(cat "$RUN_DIR/ckpt_rolling/latest_checkpointed_iteration.txt" 2>/dev/null)"
    L=$RUN_DIR/slurm-mn5-pool-p264.$rid.log
    for j in $(seq 1 60); do
      it=$(grep -m1 -E "iteration +[0-9]+/" "$L" 2>/dev/null | cut -c45-80)
      [ -n "$it" ] && { say "  first iteration: $it"; exit 0; }
      sleep 30
    done
    exit 0
  fi
  say "not started yet (rid=${rid:-none}); host=$(cat "$P/hosts/host010/state" 2>/dev/null | cut -c1-44)"
  sleep 30
done
say "ALERT main run did not start in 30 min"
exit 5
