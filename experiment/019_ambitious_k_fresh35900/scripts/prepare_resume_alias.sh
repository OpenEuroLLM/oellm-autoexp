#!/bin/bash
# Rolling saves are disabled. Let the loader's existing rolling-source fallback
# follow this run's published persistent tracker on any subsequent retry.
set -euo pipefail
run=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/K_main
mkdir -p "$run/ckpt"
if [ -L "$run/ckpt_rolling" ]; then
  [ "$(readlink "$run/ckpt_rolling")" = ckpt ]
else
  if [ -d "$run/ckpt_rolling" ]; then rmdir "$run/ckpt_rolling"; fi
  ln -s ckpt "$run/ckpt_rolling"
fi
echo "[K-resume] rolling-source alias follows the run's own completed persistent checkpoints"
