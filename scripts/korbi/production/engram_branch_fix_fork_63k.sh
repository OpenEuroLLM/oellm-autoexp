#!/bin/bash
# Translate the main run's 63,000 checkpoint into the fork checkpoint, then the fork trial loads it.
#
#   1. export the 63,000 checkpoint (HF export) and read layer 52's mean query RMS with the diag;
#      layer 2 is fixed at 0.3 (gate-opening), layer 52 keeps its mean.
#   2. add_engram_branch_gates.py --query-mix --logit-scale 2:0.3,52:<mean52>
#      -> <FORK>/ckpt/iter_0063000
#
# The main tree is only read. Run after the main run's 63,000 persistent save exists.
set -euo pipefail
cd "$(dirname "$0")/../../.."

MAIN_CKPT=${MAIN_CKPT:?set MAIN_CKPT to the main run's ckpt dir}
FORK=${FORK:?set FORK to the fork checkpoint root}
IT=${IT:-63000}
MEAN52=${MEAN52:?set MEAN52 to layer 52's mean query RMS from the 63,000 diag}
READ_SCALE="2:0.3,52:${MEAN52}"

src_it="$MAIN_CKPT/iter_$(printf %07d "$IT")"
[ -f "$src_it/.metadata" ] || { echo "no checkpoint at $src_it"; exit 1; }
dst="$FORK/ckpt/iter_$(printf %07d "$IT")"
mkdir -p "$FORK/ckpt"

echo "== bg_learned -> $dst (--query-mix --logit-scale $READ_SCALE)"
PYTHONPATH="scripts/korbi/production${PYTHONPATH:+:$PYTHONPATH}" \
    python3 scripts/korbi/production/add_engram_branch_gates.py \
    --src "$src_it" --dst "$dst" --query-mix --logit-scale "$READ_SCALE"

echo "$IT" > "$FORK/ckpt/latest_checkpointed_iteration.txt"
echo "fork checkpoint ready: $dst"
