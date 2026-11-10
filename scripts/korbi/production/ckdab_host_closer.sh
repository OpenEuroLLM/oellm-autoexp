#!/bin/bash
# Close the pool host the ckda_branch production arms ran on, once they are done (user 2026-11-10: a 264-node
# host taken for the four remaining arms must not be kept afterwards). Runs on the workstation, polls MN5:
#
#   nohup setsid scripts/korbi/production/ckdab_host_closer.sh >> ~/ckdab_host_closer.log 2>&1 &
#
# Every INTERVAL s it reads `pool_ctl status` of p264. For each arm in ARMS it takes the newest pool run
# (a node-check retry is a new run). A host other than KEEP (the main run's / the standing experiment host)
# is closed with `pool_ctl exit-host` once every arm placed on it has ended (COMPLETED / FAILED / CANCELLED)
# and no arm still waits for a placement -- the supervisor preempts what else runs there (the pool requeues
# it) and releases the allocation. A host that holds another session's run of >= BIG nodes is NOT closed (a
# warning instead). Exits when every arm has ended (or after MAX_H hours).
set -u
SSH=${SSH_TARGET:-marenostrum_new}
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
SNAP=${SNAP:-autoexp_poolshrink1}
ARMS=${ARMS:-"ckda_branch_bgqa ckda_branch_d ckda_branch_cgqa ckda_branch_drop"}
KEEP=${KEEP:-"host001 host002 host003"}
BIG=${BIG:-16}
INTERVAL=${INTERVAL:-300}
MAX_H=${MAX_H:-30}
DRY=${DRY:-0}
start=$(date +%s)
declare -A closed=()

status() {
  ssh -o ConnectTimeout=30 "$SSH" "cd $R/$SNAP && timeout 200 $R/autoexp_env/bin/python scripts/pool_ctl.py $POOL status" \
    2>/dev/null | grep -E "^\s+mn5-pool-p264\.[0-9]+ "
}

while :; do
  now=$(date +%s)
  [ $(( (now - start) / 3600 )) -ge "$MAX_H" ] && { echo "$(date -Is) giving up after $MAX_H h"; exit 1; }
  st=$(status)
  if [ -z "$st" ]; then echo "$(date -Is) no status (ssh?)"; sleep "$INTERVAL"; continue; fi
  pending=0; declare -A live=() used=()
  for arm in $ARMS; do
    # newest run of this arm: the highest run id among lines whose job name is the arm
    # (a sweep launch appends its index: ckda_branch_d_4; the *_cal_full tests do not match)
    line=$(printf '%s\n' "$st" | awk -v a="$arm" '$3 ~ ("^" a "(_[0-9]+)?$")' | sort -t. -k2 -n | tail -1)
    if [ -z "$line" ]; then echo "$(date -Is) $arm: no run yet"; pending=1; continue; fi
    state=$(awk '{print $2}' <<<"$line")
    host=$(awk '{for (i = 1; i <= NF; i++) if ($i == "host") print $(i + 1)}' <<<"$line")
    echo "$(date -Is) $arm: $state on ${host:--}"
    if [ -z "$host" ] || [ "$host" = "-" ]; then
      # never placed: waiting, unless it ended unplaced (a cancelled arm does not hold hosts open)
      case $state in COMPLETED|FAILED|CANCELLED) ;; *) pending=1 ;; esac
      continue
    fi
    grep -qw "$host" <<<"$KEEP" && continue
    used[$host]=1
    case $state in COMPLETED|FAILED|CANCELLED) ;; *) live[$host]=1 ;; esac
  done
  # a host is closed when every arm placed on it has ended and no arm still waits for a placement
  # (a waiting arm may need it); arms on KEEP hosts do not hold it open
  if [ "$pending" = 0 ]; then
    for h in "${!used[@]}"; do
      [ -n "${live[$h]:-}" ] && continue
      [ -n "${closed[$h]:-}" ] && continue
      big=$(ssh -o ConnectTimeout=30 "$SSH" "P=$POOL/hosts/$h; for r in \$(ls \$P/inuse 2>/dev/null | grep p264); do \
        n=\$(tr , '\n' < \$P/inuse/\$r | grep -c .); nm=\$(grep -h -m1 -oE '\"name\": *\"[^\"]+\"' $POOL/runs/\$r/*.json 2>/dev/null | head -1 | cut -d'\"' -f4); \
        [ \"\$n\" -ge $BIG ] && echo \"\$r \$nm \$n\"; done" 2>/dev/null | grep -v "ckda_branch_")
      if [ -n "$big" ]; then
        echo "$(date -Is) NOT closing $h: other large runs on it: $big"
        closed[$h]=skipped
        continue
      fi
      echo "$(date -Is) the arms on $h are done: closing it (pool_ctl exit-host)"
      [ "$DRY" = 1 ] || ssh -o ConnectTimeout=30 "$SSH" "cd $R/$SNAP && $R/autoexp_env/bin/python scripts/pool_ctl.py $POOL exit-host $h"
      closed[$h]=1
    done
    # every arm ended (wherever it ran): done
    n_live=0; for h in "${!live[@]}"; do n_live=$((n_live + 1)); done
    if [ "$n_live" = 0 ]; then echo "$(date -Is) all arms ended"; exit 0; fi
  fi
  sleep "$INTERVAL"
done
