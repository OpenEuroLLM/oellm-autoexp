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
#   ... [<k>|<name part>]        a host running several packed runs: the k-th (1 = first) or the
#                                one whose job name contains <name part> (default: the first)
#
# An idle host shows its most recent run. A job that is not a pool host shows its own StdOut, as
# showlog.sh does. Pools are found under $POOLS_ROOT (default: the hybrid_nucleus pools dir).

POOLS_ROOT=${POOLS_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/pools}
n=$1
shift
mode="" only_path="" pick=""
for opt in "$@"; do # options in any order: host | err, path, and a packed-run selector
  case $opt in
    path) only_path=1 ;;
    host | err) mode=$opt ;;
    *) pick=$opt ;;
  esac
done

run_name() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$1/runs/$2/run.json" 2>/dev/null; }

pick_run() { # pick_run <pool dir> <rid,rid,...> -> one rid (by $pick: index or name part)
  local i=0 r chosen="" rids
  IFS=, read -r -a rids <<<"$2"
  if [ "${#rids[@]}" -gt 1 ]; then
    echo "packed runs on this host:" >&2
    for r in "${rids[@]}"; do i=$((i + 1)); echo "  $i. $r  $(run_name "$1" "$r")" >&2; done
  fi
  if [ -z "$pick" ]; then
    chosen=${rids[0]}
  elif [[ $pick =~ ^[0-9]+$ ]]; then
    chosen=${rids[$((pick - 1))]:-}
  else
    for r in "${rids[@]}"; do
      if [[ "$(run_name "$1" "$r")" == *"$pick"* ]]; then chosen=$r; break; fi
    done
  fi
  [ -n "$chosen" ] || { echo "no packed run matches '$pick'" >&2; return 1; }
  echo "$chosen"
}

view() { # view <file>
  echo "$1"
  [ -L "$1" ] && echo "  -> $(readlink "$1")"
  [ -n "$only_path" ] && return
  less +F "$1"
}

run_log() { # run_log <pool dir> <rid> -> the run's log, as the pool's named symlink
  # <pool>/slurm-<host jobid>.<step index>-<job name>.log (the name says which job it is)
  local started job step link
  started="$1/runs/$2/started"
  job=$(sed -n 's/^job=//p' "$started" 2>/dev/null)
  step=$(sed -n 's/^step_index=//p' "$started" 2>/dev/null)
  if [ -n "$job" ] && [ -n "$step" ]; then
    link=$(ls "$1"/slurm-"$job"."$step"-*.log 2>/dev/null | head -n 1)
    if [ -n "$link" ]; then
      echo "$link"
      return
    fi
  fi
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['log'])" "$1/runs/$2/run.json" 2>/dev/null
}

schedule() { # schedule <pool dir>: hosts, the queue in start order, the last finished runs
  python3 - "$1" <<'PY'
import glob, json, os, sys, time
pool = sys.argv[1]
def env(path):
    out = {}
    try:
        for line in open(path):
            k, _, v = line.strip().partition("=")
            out[k] = v.strip("'\"")
    except OSError:
        pass
    return out
def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""
runs = []
for rd in sorted(glob.glob(f"{pool}/runs/*/")):
    try:
        meta = json.load(open(rd + "run.json"))
    except (OSError, ValueError):
        continue
    started, rc = env(rd + "started"), read(rd + "rc")
    link = ""
    if started.get("job") and started.get("step_index"):
        hits = glob.glob(f"{pool}/slurm-{started['job']}.{started['step_index']}-*.log")
        link = os.path.basename(hits[0]) if hits else ""
    state = ("done " + rc) if rc else ("RUNNING" if started else ("assigned" if read(rd + "assigned") else "queued"))
    runs.append(dict(rid=os.path.basename(rd.rstrip("/")), name=meta.get("name", "?"), nodes=meta.get("nodes", "?"),
                     prio=int(meta.get("priority", 0) or 0), enq=float(meta.get("enqueued_ts", 0) or 0),
                     state=state, host=started.get("host") or read(rd + "assigned"), link=link,
                     end=os.path.getmtime(rd + "rc") if rc else 0))
print(f"--- pool {os.path.basename(pool)}")
for hd in sorted(glob.glob(f"{pool}/hosts/*/")):
    hid, info = os.path.basename(hd.rstrip("/")), env(hd + "info.env")
    if os.path.exists(hd + "exited") or not info:
        continue
    st = read(hd + "state").split()
    what = (f"running {st[1]}" if st and st[0] == "busy" else (st[0] if st else "?"))
    adopted = f" (adopted {info['ADOPTED']})" if info.get("ADOPTED") else ""
    print(f"  host {hid}  job {info.get('SLURM_JOB_ID', '?')}{adopted}  {info.get('NUM_NODES', '?')} nodes  {what}")
live = [r for r in runs if r["state"] in ("RUNNING", "assigned")]
queued = sorted((r for r in runs if r["state"] == "queued"), key=lambda r: (-r["prio"], r["enq"]))
done = sorted((r for r in runs if r["state"].startswith("done")), key=lambda r: r["end"])[-5:]
for title, rows in (("now", live), ("next (start order)", queued), ("last finished", done)):
    if not rows:
        continue
    print(f"  {title}:")
    for i, r in enumerate(rows, 1):
        prefix = f"{i:2d}." if title.startswith("next") else "   "
        where = f" on {r['host']}" if r["host"] else ""
        print(f"   {prefix} {r['name']:<32} {str(r['nodes']):>4}n prio {r['prio']:<3} {r['state']:<18}"
              f"{where}  {r['link'] or r['rid']}")
PY
}

if [ -z "$n" ]; then
  sed -n '2,15p' "$0"
  exit 1
fi

# a pool run id given directly
if [[ $n == *.[0-9][0-9][0-9][0-9] ]]; then
  for p in "$POOLS_ROOT"/*/; do
    if [ -d "$p/runs/$n" ]; then
      schedule "${p%/}"
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
# (one allocation can be a host of two pools: an attached one (pool_ctl attach) and its idle
# original; take the busy one, else one that has not exited)
hostdir="" fallback=""
for f in $(grep -lx "SLURM_JOB_ID=$slurmid" "$POOLS_ROOT"/*/hosts/*/info.env 2>/dev/null); do
  d=$(dirname "$f")
  if read -r st _ <"$d/state" 2>/dev/null && [ "$st" = busy ]; then hostdir=$d; break; fi
  [ -e "$d/exited" ] || [ -n "$fallback" ] || fallback=$d
done
hostdir=${hostdir:-$fallback}
if [ -n "$hostdir" ]; then
  pool=$(dirname "$(dirname "$hostdir")")
  hid=$(basename "$hostdir")
  schedule "$pool"
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
    rid=$(pick_run "$pool" "$rid") || exit 1
    echo "pool $(basename "$pool"), host $hid, running $rid ($(run_name "$pool" "$rid"))"
  fi
  view "$(run_log "$pool" "$rid")"
  exit 0
fi

# not a pool host: the job's own log, as showlog.sh
key=StdOut
[ "$mode" = err ] && key=StdErr
view "$(scontrol show jobid="$slurmid" | grep "$key=" | sed "s/.*$key=//" | tr -d ' ')"
