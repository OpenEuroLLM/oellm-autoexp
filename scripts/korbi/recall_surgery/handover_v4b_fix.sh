#!/usr/bin/env bash
# Hand the v4b main line over to the fixed successor (MoonEP e9e8238 + corrected restart flags).
#
# Trigger: the old line's cudaErrorIllegalAddress (the crash the fix removes) or the old line ending.
# The successor (ambitious_v4b_learned_fix_256) waits for <R>/v4b_fix_switch/armed and for no live
# run of the old name, so this script is the only thing that can start it.
#
# Sequence at the trigger:
#   1. pool_ctl cancel ambitious_v4b_learned_256   -- stop the job and its automatic restarts
#   2. verify no unfinished old-name run (retry the cancel; the monitor may have restarted it)
#   3. touch armed                                  -- the successor's start condition turns true
#   4. wait for the successor, then verify it really runs the fixed build
#
# Usage:
#   setsid nohup bash handover_v4b_fix.sh > $R/v4b_fix_switch/handover.log 2>&1 &
#   bash handover_v4b_fix.sh --dry-run        # print what it would do
#   bash handover_v4b_fix.sh --force          # hand over now (no crash needed)
#
# Exit codes: 0 handover done and successor verified, 2 budget spent, 5 handover failed.
set -uo pipefail

R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
STATE=$R/v4b_fix_switch
ARMED=$STATE/armed
OLD_NAME=ambitious_v4b_learned_256
NEW_NAME=ambitious_v4b_learned_fix_256
OLD_RUN_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/v4b_learned/main
CTL_DIR=$R/autoexp_v4bfix1
PY=$R/autoexp_env/bin/python

INTERVAL=20
BUDGET=86400
MAX_CANCELS=8
DRY=0
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --max-cancels) MAX_CANCELS=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    --force) FORCE=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 5;;
  esac
done

mkdir -p "$STATE" || exit 5
say() { echo "[$(date +%H:%M:%S)] $*"; }

pool_ctl() {
  # pool_ctl needs the repo on PYTHONPATH (compoconf comes from the venv, not the system python).
  ( cd "$CTL_DIR" && PYTHONPATH=. "$PY" scripts/pool_ctl.py "$POOL" "$@" )
}

live_runs() {  # ids of unfinished runs whose name is exactly $1
  "$PY" - "$POOL" "$1" <<'PY'
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
    if m.get("name") == name:
        out.append(d.rsplit(".", 1)[-1])
print(" ".join(sorted(out)))
PY
}

recent_rids() {  # newest 3 ids of $1, finished or not: a crash is written to the run that died,
                # and the pool can restart the line faster than one poll interval
  "$PY" - "$POOL" "$1" <<'PY'
import glob, json, os, sys
pool, name = sys.argv[1], sys.argv[2]
out = []
for d in glob.glob(pool + "/runs/mn5-pool-p264.*"):
    try:
        m = json.load(open(os.path.join(d, "run.json")))
    except Exception:
        continue
    if m.get("name") == name:
        out.append(int(d.rsplit(".", 1)[-1]))
print(" ".join(str(i) for i in sorted(out)[-3:]))
PY
}

crashed() {  # 0 if the run's log carries the illegal-access signature
  local rid=$1
  local log=$OLD_RUN_DIR/slurm-mn5-pool-p264.$rid.log
  [ -f "$log" ] || return 1
  grep -qE "cudaErrorIllegalAddress|illegal memory access" "$log"
}

if [ "$DRY" = 1 ]; then
  say "DRY RUN"
  say "  old live runs : $(live_runs $OLD_NAME)"
  say "  recent old    : $(recent_rids $OLD_NAME)"
  say "  armed marker  : $([ -f $ARMED ] && echo present || echo absent)"
  say "  would run     : pool_ctl cancel $OLD_NAME ; touch $ARMED ; wait for $NEW_NAME"
  exit 0
fi

say "handover watcher: $OLD_NAME -> $NEW_NAME (armed=$([ -f $ARMED ] && echo yes || echo no))"
START=$(date +%s)

# ---------- phase 1: wait for the trigger ----------
trigger=""
while [ -z "$trigger" ]; do
  if [ $(( $(date +%s) - START )) -ge "$BUDGET" ]; then
    say "budget spent without a trigger"
    exit 2
  fi
  if [ -f "$ARMED" ]; then
    trigger="already armed"
    break
  fi
  if [ "$FORCE" = 1 ]; then
    trigger="forced"
    break
  fi
  # Check the newest runs whether or not they are still live: the pool restarts the line within a
  # minute of a crash, so a live-only scan would miss the very event we hand over on.
  for rid in $(recent_rids $OLD_NAME); do
    if crashed "$rid"; then
      trigger="illegal access in .$rid"
      say "TRIGGER $trigger"
      break
    fi
  done
  if [ -z "$trigger" ]; then
    # A run that ended without the signature still means the line is down; take over rather
    # than let the monitor restart the unfixed build again.
    if [ -z "$(live_runs $OLD_NAME)" ]; then
      trigger="old line ended"
      say "TRIGGER $trigger"
    fi
  fi
  [ -z "$trigger" ] && sleep "$INTERVAL"
done

# ---------- phase 2: stop the old line, arm the successor ----------
if [ ! -f "$ARMED" ]; then
  n=0
  while :; do
    n=$((n + 1))
    if [ "$n" -gt "$MAX_CANCELS" ]; then
      say "FAILED: old line still live after $MAX_CANCELS cancels: $(live_runs $OLD_NAME)"
      exit 5
    fi
    say "cancel $OLD_NAME (attempt $n)"
    pool_ctl cancel "$OLD_NAME" 2>&1 | sed 's/^/    /'
    sleep 15
    if [ -z "$(live_runs $OLD_NAME)" ]; then
      say "old line stopped"
      break
    fi
    say "  still live: $(live_runs $OLD_NAME)"
  done
  touch "$ARMED"
  say "armed $ARMED -- the successor may start"
fi

# ---------- phase 3: verify the successor ----------
deadline=$(( $(date +%s) + 1800 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  rid=$(live_runs $NEW_NAME)
  if [ -n "$rid" ]; then
    rid=$(echo "$rid" | awk '{print $NF}')
    d=$POOL/runs/mn5-pool-p264.$rid
    if [ -f "$d/started" ]; then
      moonep=$(grep -oE "pylibs_moonep_[a-z0-9]+" "$d/step.sh" 2>/dev/null | head -1)
      host=$(grep -oE "^host[0-9]+" "$d/assigned" 2>/dev/null | head -1)
      log=$OLD_RUN_DIR/slurm-mn5-pool-p264.$rid.log
      it=$(grep -m1 -oE "iteration +[0-9]+/" "$log" 2>/dev/null | head -1)
      say "successor .$rid started on ${host:-?} moonep=$moonep ${it:-no iteration yet}"
      if [ "$moonep" != "pylibs_moonep_e9e8238" ]; then
        say "FAILED: successor renders $moonep, expected pylibs_moonep_e9e8238"
        exit 5
      fi
      say "HANDOVER DONE"
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done

say "FAILED: successor did not start within 30 min (armed=$ARMED)"
exit 5
