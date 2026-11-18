#!/bin/bash
# v4c guard on host007: keep the arm runnable without touching v4_cont or v4b.
#
# Two failure modes this handles, both seen on 2026-11-18:
#
#  1. "Requested node configuration is not available": the run's node list holds a node that
#     Slurm has since removed from the allocation (as07r5b08, as04r3b13, as07r5b14). The fix is
#     to put those nodes into host007's exclude_nodes, which usable_nodes() subtracts.
#  2. the arm is neither running nor queued (every attempt failed): re-enqueue it.
#
# It never cancels, preempts or edits a SLURM job, and it only ever looks at host007.
set -u
P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
TREE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_v4c
PY=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_env/bin/python
JOB=48685344
NAME=ambitious_v4c_ramp_256
EXCL="$P/hosts/host007/exclude_nodes"
say() { echo "[$(date +%H:%M:%S)] $*"; }

removed_nodes() {
  scontrol show hostnames "$(squeue -h -j "$JOB" -o %N | head -n1)" 2>/dev/null | sort -u > /tmp/guard_live.txt
  grep -oE "^NODELIST=.*" "$P/hosts/host007/info.env" 2>/dev/null | cut -d= -f2- \
    | tr ',' '\n' | grep -v '^$' | sort -u > /tmp/guard_known.txt
  comm -23 /tmp/guard_known.txt /tmp/guard_live.txt
}

exclude_removed() {
  local n changed=0
  for n in $(removed_nodes); do
    grep -qx "$n" "$EXCL" 2>/dev/null && continue
    echo "$n" >> "$EXCL"
    say "excluded removed-from-allocation node $n"
    changed=1
  done
  return $((1 - changed))
}

arm_state() {  # running | queued | absent
  local d n
  for d in "$P"/runs/mn5-pool-p264.*/; do
    n=$(grep -E "^NAME=" "$d/meta.env" 2>/dev/null | cut -d= -f2)
    [ "${n#ambitious_v4c}" = "$n" ] && continue
    [ -e "$d/rc" ] && continue
    echo running; return
  done
  echo absent
}

last_rc() {  # rc of the newest v4c run that has one
  local d n
  for d in $(ls -td "$P"/runs/mn5-pool-p264.* 2>/dev/null); do
    n=$(grep -E "^NAME=" "$d/meta.env" 2>/dev/null | cut -d= -f2)
    [ "${n#ambitious_v4c}" = "$n" ] && continue
    [ -e "$d/rc" ] && { cat "$d/rc"; return; }
  done
}

requeue() {
  say "no active v4c run: re-enqueueing $NAME"
  ( cd "$TREE" && PYTHONPATH=. "$PY" scripts/run_autoexp.py --no-monitor --pool "$P" \
      --config-name experiments/korbi/recall_surgery/v4c_ramp_mn5 2>&1 | tail -1 )
}

say "guard up: host007, $NAME"
while true; do
  # a node-config failure in a recent v4c attempt means the list is stale again
  stale=0
  for d in $(ls -td "$P"/runs/mn5-pool-p264.* 2>/dev/null | head -8); do
    n=$(grep -E "^NAME=" "$d/meta.env" 2>/dev/null | cut -d= -f2)
    [ "${n#ambitious_v4c}" = "$n" ] && continue
    [ -e "$d/rc" ] || continue
    L=$(grep -E "^LOG=" "$d/meta.env" 2>/dev/null | cut -d= -f2-)
    [ -n "$L" ] || continue
    grep -aq "Requested node configuration" "$L" 2>/dev/null && stale=1
    grep -aq "misaligned address" "$L" 2>/dev/null && say "MISALIGNED at $(basename "$d") (should be impossible after 429a61ba)"
  done
  [ "$stale" = 1 ] && exclude_removed

  case "$(arm_state)" in
    running) : ;;
    absent)
      rc=$(last_rc)
      case "$rc" in
        *completed*) : ;;                     # the arm finished: never resurrect it
        *cancelled*) : ;;                     # a human stopped it on purpose
        *failed*|*insufficient*|"")
          exclude_removed; requeue ;;
        *) say "last v4c rc='$rc': leaving it alone" ;;
      esac
      ;;
  esac
  sleep 300
done
