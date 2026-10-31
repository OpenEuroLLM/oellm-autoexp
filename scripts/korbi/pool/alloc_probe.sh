#!/bin/bash
# Allocation / main-run probe for the controlling agent (user 2026-10-31): one line per fact, STABLE
# between calls unless something changed, so a caller diffing two calls sees only events.
#   JOB  <id> <name> <state> <nodes> <account>       every own SLURM job with >= 32 nodes
#   MAIN <job name> <pool> <run> <status> [host]      the newest active run of each main run
#   STALL <job name> <pool> <run> log idle >15min     a RUNNING main run whose log stopped
#   MONDEAD <pool>                                    pool monitor heartbeat older than 10 min
#   HELDIDLE <pool> <host> <nodes>                    a running host with no active run (costs)
#   PREEMPT/FAIL <job name> <pool> <run> <rc>         a main run that ended badly (last 2 h)
# Run on MareNostrum 5: bash alloc_probe.sh
P=/gpfs/scratch/ehpc533/hybrid_nucleus/pools
POOLS="p33 p129 p132 p264 o134e o134c o260c"
MAINS="q35c_main_gqa ambitious_v2_128n"
now=$(date +%s)

squeue -u "$USER" -h -o "%i %j %T %D %a" 2>/dev/null | awk '$4 >= 32 {print "JOB", $0}' | sort

for p in $POOLS; do
  d=$P/mn5_pool_$p
  [ -d "$d" ] || continue
  alive=$(stat -c %Y "$d/monitor.alive" 2>/dev/null || echo 0)
  [ $((now - alive)) -gt 600 ] && echo "MONDEAD $p"
  for h in "$d"/hosts/host*; do
    st=$(cat "$h/state" 2>/dev/null)
    [ -e "$h/exited" ] && continue
    case "$st" in
      idle*)
        ts=$(echo "$st" | awk '{print $2}')
        [ -n "$ts" ] && [ $((now - ${ts%.*})) -gt 600 ] &&
          echo "HELDIDLE $p $(basename "$h") $(grep -o 'NUM_NODES=[0-9]*' "$h/info.env" 2>/dev/null | cut -d= -f2)"
        ;;
    esac
  done
done

for m in $MAINS; do
  best=""
  for p in $POOLS; do
    d=$P/mn5_pool_$p/runs
    [ -d "$d" ] || continue
    for r in $(grep -l "\"name\": \"$m\"" "$d"/*/run.json 2>/dev/null | xargs -r -n1 dirname); do
      rid=$(basename "$r")
      if [ -e "$r/started" ] && [ ! -e "$r/rc" ]; then
        host=$(grep -o 'host=[^ ]*' "$r/started" 2>/dev/null | head -1)
        echo "MAIN $m $p $rid RUNNING $host"
        log=$(grep -o '"log": "[^"]*"' "$r/run.json" | cut -d'"' -f4)
        lm=$(stat -c %Y "$log" 2>/dev/null || echo "$now")
        [ $((now - lm)) -gt 900 ] && echo "STALL $m $p $rid log idle >15min"
        best=1
      elif [ -e "$r/rc" ]; then
        rc=$(cat "$r/rc")
        t=$(stat -c %Y "$r/rc")
        case "$rc" in
          *completed* | *cancelled*) ;;
          *) [ $((now - t)) -lt 7200 ] && echo "FAIL $m $p $rid ${rc// /_}" ;;
        esac
      fi
    done
  done
  [ -z "$best" ] && echo "MAIN $m NOT_RUNNING"
done
