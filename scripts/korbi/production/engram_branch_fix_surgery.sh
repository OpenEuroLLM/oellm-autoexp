#!/bin/bash
# Engram gate fix surgery on v4b's checkpoint, one 1-node pool job:
#   fix        single gate + gate_terms.logit_scale, no per-stream copies (--no-branch)
#   bg_read    per-stream gates, query read + logit_scale
#   bg_stream  per-stream gates, query stream + logit_scale
# The logit_scale init is the query's mean |h~| from the offline diagnosis of the same export:
#   read   0.016 (layer 2) / 0.47 (layer 52)
#   stream ~0.033 (the streams are nearly identical)
# so step 0 is function-preserving. Output: <OUT_ROOT>/<v>/ckpt/iter_<IT>.
set -euo pipefail
cd "$(dirname "$0")/../../.."
SRC=${SRC:?set SRC to the v4b ckpt directory}
IT=${IT:-53000}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT}
READ_SCALE=${READ_SCALE:-2:0.016,52:0.47}
STREAM_SCALE=${STREAM_SCALE:-2:0.033,52:0.033}
it=iter_$(printf %07d "$IT")
src_it="$SRC/$it"
[ -f "$src_it/.metadata" ] || { echo "no checkpoint at $src_it"; exit 1; }
export PYTHONPATH="scripts/korbi/production${PYTHONPATH:+:$PYTHONPATH}"
for v in fix bg_read bg_stream; do
    dst="$OUT_ROOT/$v/ckpt/$it"
    if [ -f "$OUT_ROOT/$v/ckpt/latest_checkpointed_iteration.txt" ]; then
        echo "$v exists: skipped"
        continue
    fi
    extra=()
    case "$v" in
        fix) extra=(--no-branch --logit-scale "$READ_SCALE") ;;
        bg_read) extra=(--logit-scale "$READ_SCALE") ;;
        bg_stream) extra=(--logit-scale "$STREAM_SCALE") ;;
    esac
    echo "== $v -> $dst (${extra[*]})"
    python3 scripts/korbi/production/add_engram_branch_gates.py --src "$src_it" --dst "$dst" "${extra[@]}"
done
