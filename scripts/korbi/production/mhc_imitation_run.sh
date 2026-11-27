#!/bin/bash
# The fit of an mhc_coeff_dump directory, then the LDM with fixed addresses fitted to it, in one container call
# (the pool step wraps its command in bash -c '...': a chained '&& python3' there runs on the host).
#   bash mhc_imitation_run.sh <dump dir> [keys, default 8,16,32] [adam steps, default 3000]
set -euo pipefail
cd "$(dirname "$0")/../../.."
DUMP=$1
KEYS=${2:-8,16,32}
STEPS=${3:-3000}
python3 scripts/korbi/production/mhc_imitation_fit.py --dump "$DUMP" --out "$DUMP/fit.json" --device cuda --gram "$DUMP/gram.npz"
python3 scripts/korbi/production/ldm_imitation_init.py --gram "$DUMP/gram.npz" --keys "$KEYS" --steps "$STEPS" --out "$DUMP/ldm_init.json"
