#!/bin/bash
# Every FABRIC_PROBE_INTERVAL_S (900) submit scripts/pool/fabric_probe.sbatch twice, each only
# while the previous one of its kind is not queued or running, avoiding the excluded nodes:
#   fabric_probe        placed by SLURM (often all in one network cell: the same-cell reference)
#   fabric_probe_cross  pinned to one IDLE node in each of 2-4 different cells (starts at once;
#                       cross-cell; skipped while fewer than 2 cells have an idle node)
# Run on a login node, e.g.
#   systemd-run --user --unit=fabric-probe -p WorkingDirectory=$PWD bash scripts/pool/fabric_probe_loop.sh
# and stop it with `systemctl --user stop fabric-probe`. Results: $FABRIC_PROBE_RESULTS.
set -u
export FABRIC_PROBE_REPO=${FABRIC_PROBE_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
export FABRIC_PROBE_SIF=${FABRIC_PROBE_SIF:-/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif}
export FABRIC_PROBE_RESULTS=${FABRIC_PROBE_RESULTS:?a results file, e.g. <dir>/fabric_probe.tsv}
ACCOUNT=${FABRIC_PROBE_ACCOUNT:-e-sta-openeurollm}
INTERVAL=${FABRIC_PROBE_INTERVAL_S:-900}
EXCLUDE_FILE=${FABRIC_PROBE_EXCLUDE_FILE:-/e/project1/e-sta-openeurollm/production_training/jupiter_exclude_nodes.txt}
LOGDIR=$(dirname "$FABRIC_PROBE_RESULTS")
mkdir -p "$LOGDIR"

submit() {  # name, extra sbatch args...
  local name=$1
  shift
  sbatch --account="$ACCOUNT" --job-name="$name" --output="$LOGDIR/$name-%j.log" --export=ALL \
    "$@" "$FABRIC_PROBE_REPO/scripts/pool/fabric_probe.sbatch" 2>&1 | sed "s/^/$(date +%T) $name: /"
}

# one idle, not excluded node in each of up to 4 random network cells ((rack-1)//5), comma-separated.
# State exactly "idle": `sinfo -t idle` also lists drained idle nodes, which never start a job.
cross_nodes() {
  local excluded=$1
  sinfo -p booster -h -t idle -N -o "%N %T" | awk '$2 == "idle" {print $1}' | sort -u |
    grep -v -x -F -f <(tr , '\n' <<<"$excluded") |
    awk -F- '{c=int(($2-1)/5); if (!(c in pick)) pick[c]=$0} END{for (c in pick) print pick[c]}' |
    shuf -n 4 | paste -sd,
}

while true; do
  exclude=$(grep -v '^#' "$EXCLUDE_FILE" 2>/dev/null | tr ', ' '\n\n' | grep . | paste -sd,)
  if [ -z "$(squeue -u "$USER" -h -n fabric_probe -o %i 2>/dev/null)" ]; then
    submit fabric_probe ${exclude:+--exclude="$exclude"}
  fi
  # a pinned probe still pending a round later lost a node to another job: pick again
  stale=$(squeue -u "$USER" -h -n fabric_probe_cross -t PENDING -o %i 2>/dev/null)
  [ -n "$stale" ] && scancel $stale && echo "$(date +%T) fabric_probe_cross: $stale still pending, re-picking"
  [ -n "$stale" ] && sleep 5
  if [ -z "$(squeue -u "$USER" -h -n fabric_probe_cross -o %i 2>/dev/null)" ]; then
    nodes=$(cross_nodes "$exclude")
    n=$(tr , '\n' <<<"$nodes" | grep -c .)
    if [ "$n" -ge 2 ]; then
      submit fabric_probe_cross --nodes="$n" --nodelist="$nodes"
    else
      echo "$(date +%T) fabric_probe_cross: fewer than 2 cells with an idle node, skipped"
    fi
  fi
  sleep "$INTERVAL"
done
