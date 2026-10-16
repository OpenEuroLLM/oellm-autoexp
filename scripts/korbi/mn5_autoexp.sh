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
# Runs go to scratch (configs place them under $OUTPUT_DIR; unset, they land in /gpfs/projects).
export OUTPUT_DIR=${OUTPUT_DIR:-/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp}
cd "$(dirname "$0")/../.."
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
exec "$R/autoexp_env/bin/python" "$@"
