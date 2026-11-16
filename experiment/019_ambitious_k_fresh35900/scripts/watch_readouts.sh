#!/bin/bash
# Run on the MN5 login node. Export jobs publish readiness only after parity.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
root=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116
exports=/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports
mkdir -p "$root/readout_watch"
exec 9> "$root/readout_watch/lock"
flock -n 9 || exit 0
echo $$ > "$root/readout_watch/pid"
while [ ! -e "$root/readout_watch/stop" ]; do
  remaining=0
  for step in 35900 35999 36400 36900; do
    marker=$root/readout_watch/launched_$step
    [ ! -e "$marker" ] || continue
    remaining=$((remaining + 1))
    name=ambitious_K_fresh35900_fp8_$step
    [ -f "$exports/$name/.downstream_ready" ] || continue
    if scripts/korbi/mn5_autoexp.sh scripts/evals/launch.py config/evals/ambitious_k_recall_mn5.yaml \
      --step "$step" --export "$name" > "$root/readout_watch/launch_$step.log" 2>&1; then
      date -Is > "$marker"
      echo "$(date -Is) launched recall readout $step"
    else
      echo "$(date -Is) launch $step failed; see launch_$step.log"
    fi
  done
  [ "$remaining" -gt 0 ] || exit 0
  sleep 30
done
