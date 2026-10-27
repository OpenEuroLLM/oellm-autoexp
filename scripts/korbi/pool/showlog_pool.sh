#!/bin/bash
# showlog_pool.sh -- like ~/showlog.sh, but for a pool host it shows the log of the RUN the host
# is executing now (the job's own training log), not the host's supervisor log.
#
#   showlog_pool.sh <n>          the n-th job of `squeue -u $USER` (0 = first, negative: from the end)
#   showlog_pool.sh j<jobid>     a SLURM job id (a pool host, an adopted job, or any other job)
#   showlog_pool.sh <run id>     a pool run, e.g. mn5-pool-p129.0012
#   ... [host]                   the host's supervisor log instead of the run's
#   ... [err]                    a plain (non-pool) job's StdErr
#   ... [path]                   only print the log path (no less)
#
# An idle host shows its most recent run. A job that is not a pool host shows its own StdOut, as
# showlog.sh does. Pools are found under $POOLS_ROOT (default: the hybrid_nucleus pools dir).

POOLS_ROOT=${POOLS_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/pools}
n=$1
shift
mode="" only_path=""
for opt in "$@"; do # options in any order: host | err, and path
  case $opt in
    path) only_path=1 ;;
    host | err) mode=$opt ;;
    *) echo "unknown option $opt (host, err, path)" >&2; exit 1 ;;
  esac
done

view() { # view <file>
  echo "$1"
  [ -n "$only_path" ] && return
  less +F "$1"
}

run_log() { # run_log <pool dir> <rid> -> path of the run's log
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['log'])" "$1/runs/$2/run.json" 2>/dev/null
}

if [ -z "$n" ]; then
  sed -n '2,13p' "$0"
  exit 1
fi

# a pool run id given directly
if [[ $n == *.[0-9][0-9][0-9][0-9] ]]; then
  for p in "$POOLS_ROOT"/*/; do
    if [ -d "$p/runs/$n" ]; then
      echo "pool $(basename "$p"), run $n"
      view "$(run_log "${p%/}" "$n")"
      exit 0
    fi
  done
  echo "no pool run $n under $POOLS_ROOT" >&2
  exit 1
fi

if [[ ${n:0:1} == "j" ]]; then
  slurmid=${n:1}
else
  x=$(($(squeue -u "$USER" | wc -l) - 1))
  [[ $n -lt 0 ]] && n=$((x + n))
  slurmid=$(squeue -u "$USER" | tail -n +$((n + 2)) | head -n 1 | awk '{print $1}')
fi
echo "job $slurmid"

# the pool host (or adopted job) with this SLURM job id
hostdir=$(grep -lx "SLURM_JOB_ID=$slurmid" "$POOLS_ROOT"/*/hosts/*/info.env 2>/dev/null | head -n 1)
if [ -n "$hostdir" ]; then
  hostdir=$(dirname "$hostdir")
  pool=$(dirname "$(dirname "$hostdir")")
  hid=$(basename "$hostdir")
  if [ "$mode" = host ]; then
    echo "pool $(basename "$pool"), host $hid: supervisor log"
    view "$pool/slurm-$slurmid.log"
    exit 0
  fi
  # state: "busy <rid> <ts> <usable>" | "idle <ts> <usable>"
  read -r what rid _ < "$hostdir/state"
  if [ "$what" != busy ]; then
    # idle: the host's most recent run (ledger: <ts> <hid> <jobid> <rid> start ...)
    rid=$(awk -F'\t' -v h="$hid" -v j="$slurmid" '$2==h && $3==j && $5 ~ /^start/ {r=$4} END {print r}' "$pool/ledger.tsv")
    echo "pool $(basename "$pool"), host $hid is ${what:-?}; its last run: ${rid:-none}"
    [ -z "$rid" ] && exit 1
  else
    echo "pool $(basename "$pool"), host $hid, running $rid ($(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$pool/runs/$rid/run.json" 2>/dev/null))"
  fi
  view "$(run_log "$pool" "$rid")"
  exit 0
fi

# not a pool host: the job's own log, as showlog.sh
key=StdOut
[ "$mode" = err ] && key=StdErr
view "$(scontrol show jobid="$slurmid" | grep "$key=" | sed "s/.*$key=//" | tr -d ' ')"
