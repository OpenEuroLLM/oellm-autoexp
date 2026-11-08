#!/bin/bash
# Submit scripts/pool/fabric_probe.sbatch every FABRIC_PROBE_INTERVAL_S (900) while none of ours
# is queued or running, avoiding the nodes the pool excludes. Run on a login node, e.g.
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

while true; do
  if [ -z "$(squeue -u "$USER" -h -n fabric_probe -o %i 2>/dev/null)" ]; then
    exclude=$(grep -v '^#' "$EXCLUDE_FILE" 2>/dev/null | tr ', ' '\n\n' | grep . | paste -sd,)
    sbatch --account="$ACCOUNT" ${exclude:+--exclude="$exclude"} --output="$LOGDIR/fabric_probe-%j.log" \
      --export=ALL "$FABRIC_PROBE_REPO/scripts/pool/fabric_probe.sbatch" 2>&1 | sed "s/^/$(date +%T) /"
  fi
  sleep "$INTERVAL"
done
