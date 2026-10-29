#!/bin/bash
# Run oellm-autoexp on the MareNostrum 5 login node (ehpc533), whose system Python is 3.9:
# with the packed autoexp environment $R/autoexp_env (built by
# scripts/korbi/build_autoexp_env.sh), directly against the host's Slurm. Run from the
# synced checkout (./sync_to_marenostrum.sh -> $R/autoexp):
#
#   cd $R/autoexp && scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py \
#       --config-name experiments/korbi/hybrid_speed/speed_130M_mn5 --yes
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
# Runs go to scratch: configs place them under $OUTPUT_DIR/<experiment group>/... -- the shared
# MareNostrum log root /gpfs/scratch/ehpc533/hybrid_nucleus (runs before 2026-10-24: under
# /gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp). Override with HYBRID_OUTPUT_DIR.
export OUTPUT_DIR=${HYBRID_OUTPUT_DIR:-/gpfs/scratch/ehpc533/hybrid_nucleus}
cd "$(dirname "$0")/../.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
# The login node caps CPU time per process (alogin2: ulimit -t 300, SIGKILL at the limit; the
# p129 pool monitor died three times, 2026-10-28/29). monitor_autoexp.py leaves cleanly near its
# limit with status 75 (EXIT_CPU_BUDGET), and a fresh process takes over here. A TERM / INT for
# this wrapper (kill <pid file>) goes to the child, and ends the loop.
child=""
stop=""
trap 'stop=1; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null' TERM
trap 'stop=1; [ -n "$child" ] && kill -INT "$child" 2>/dev/null' INT
while true; do
  "$R/autoexp_env/bin/python" "$@" &
  child=$!
  rc=0
  wait "$child" || rc=$?
  # a trapped signal interrupts wait: wait again for the child's own exit status
  while kill -0 "$child" 2>/dev/null; do wait "$child" || rc=$?; done
  child=""
  if [ "$rc" -eq 75 ] && [ -z "$stop" ]; then
    echo "[mn5_autoexp] $(date '+%F %T') CPU budget reached: restarting $*" >&2
    continue
  fi
  exit "$rc"
done
