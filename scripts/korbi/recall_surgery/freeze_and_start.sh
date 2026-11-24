#!/usr/bin/env bash
# Clear host010 of evals, hold the host so nothing refills, then release it to the main run.
#
# The failure mode: a main run that is *assigned* is never re-evaluated, so if it was assigned when
# fewer than 256 usable nodes were free it waits forever, while evals keep taking the rest. Freezing
# the host stops all new assignments (including the main run), so the order is:
#   freeze -> preempt every running eval -> wait for the host to empty -> unfreeze -> the main run
#   is the first thing the next assignment pass offers (priority 150 against 55/60) and it fits.
#
# Usage: bash freeze_and_start.sh [--host host010] [--rounds 12]
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
HOST=host010
H=$P/hosts/$HOST
PY=$R/autoexp_env/bin/python
SNAP=$R/autoexp_v4bfix2
RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main

ROUNDS=12
while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST=$2; H=$P/hosts/$HOST; shift 2;;
    --rounds) ROUNDS=$2; shift 2;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

say() { echo "[$(date +%H:%M:%S)] $*"; }
ctl() { ( cd "$SNAP" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$P" "$@" 2>&1 | tail -2 | sed 's/^/    /' ); }

live_runs() {  # every unfinished run on this host, from inuse
  ls -1 "$H/inuse" 2>/dev/null
}
main_rid() {
  "$PY" - "$P" <<'PY'
import glob, json, os, sys
best = None
for d in glob.glob(sys.argv[1] + "/runs/mn5-pool-p264.*"):
    if os.path.exists(os.path.join(d, "rc")):
        continue
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    n = str(m.get("name") or "")
    if n.startswith("ambitious_v4b_learned_") and not n.startswith("ev_"):
        r = int(d.rsplit(".", 1)[-1])
        best = r if best is None or r > best else best
print(best if best is not None else "")
PY
}

say "freezing $HOST so no new run is placed while the evals are cleared"
ctl freeze "$HOST"

for round in $(seq 1 "$ROUNDS"); do
  runs=$(live_runs)
  [ -z "$runs" ] && { say "host is empty"; break; }
  say "round $round: preempting $(echo $runs | tr '\n' ' ')"
  for r in $runs; do ctl preempt "$r"; done
  sleep 30
done

say "host inuse after clearing: '$(live_runs | tr '\n' ' ')'"
ctl unfreeze "$HOST"
say "host released to the assignment pass"

rid=$(main_rid)
say "main run candidate: ${rid:-none}"
for i in $(seq 1 40); do
  rid=$(main_rid)
  if [ -n "$rid" ] && [ -f "$P/runs/mn5-pool-p264.$rid/started" ]; then
    say "MAIN RUN .$rid STARTED"
    say "  host state: $(cat $H/state)"
    say "  inuse: $(live_runs | tr '\n' ' ')"
    say "  excludes inside job: $(grep -oE '\-\-exclude=[^ ]*' "$P/runs/mn5-pool-p264.$rid/step.sh" | head -1 | cut -d= -f2 | tr ',' '\n' | grep -c .)"
    L=$RUN_DIR/slurm-mn5-pool-p264.$rid.log
    for j in $(seq 1 50); do
      it=$(grep -m1 -E "iteration +[0-9]+/" "$L" 2>/dev/null | cut -c45-80)
      [ -n "$it" ] && { say "  first iteration: $it"; break; }
      sleep 30
    done
    exit 0
  fi
  sleep 20
done
say "ALERT main run did not start after the host was released"
say "  inuse: $(live_runs | tr '\n' ' ')"
exit 5
