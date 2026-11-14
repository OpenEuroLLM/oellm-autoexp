#!/bin/bash
# Phase-2 early insertion of the small-scale testbed (docs/recall_surgery_ambitious.md §14): a b7 (7:1) checkpoint ->
# a new arm's start checkpoint with random-init softmax attention at more CKDA positions
# (scripts/korbi/production/replace_ckda_with_attention.py: rows at the reference attention's median row norms, fresh
# MuOwn / Adam states, the position's own input norm and mHC; the new hybrid_layer_pattern in the saved args).
#   <ROOT>/<ARM>/ckpt/iter_<IT>  (hard links of b7's + the new tensors) and its tracker (= IT, after the read-back)
# Arms (sublayer indices of the 50-sublayer small31 pattern; mixer m -> sublayer):
#   E   3:1 as r3:  new attention at mixers 4 / 12 / 20  = sublayers 9 / 25 / 42, references 0 / 17 / 34
#   Ep  paired as r3p: new attention at mixers 1 / 9 / 17 = sublayers 3 / 19 / 36, references 0 / 17 / 34
# The first CKDA mixers (complex eigenvalues) stay CKDA in both. Login node (hard links + a few MB of new tensors).
#   ARM=E IT=1907 small31_insert.sh
set -euo pipefail
: "${ARM:?E | Ep}" "${IT:?b7 iteration}"
ROOT=${ROOT:-/e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/small31}
SRC_ARM=${SRC_ARM:-b7}
SIF=${SIF:-/e/project1/e-sta-openeurollm/poeppel1/container_cache/MegatronTraining-JUPITER_pt2512_aarch64_202602231451.sif}
B7="*-NL-L-L-L-L-L-L-*-L-L-L-L-L-L-L-N*-L-L-L-L-L-L-L-"
case "$ARM" in
  E|L)   LAYERS=9,25,42; PATTERN="*-NL-L-L-*-L-L-L-*-L-L-L-*-L-L-L-N*-L-L-L-*-L-L-L-" ;;
  Ep|Lp) LAYERS=3,19,36; PATTERN="*-N*-L-L-L-L-L-L-*-*-L-L-L-L-L-L-N*-*-L-L-L-L-L-L-" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
REFS=0,17,34
it=$(printf 'iter_%07d' "$IT")
SRC=$ROOT/$SRC_ARM/ckpt/$it
[ -f "$SRC/.metadata" ] || SRC=$ROOT/$SRC_ARM/ckpt_rolling/$it
[ -f "$SRC/.metadata" ] || { echo "no $it of $SRC_ARM" >&2; exit 1; }
OUT=$ROOT/$ARM/ckpt
if [ -f "$OUT/latest_checkpointed_iteration.txt" ]; then
  echo "$ARM: $OUT already has a tracker ($(cat "$OUT/latest_checkpointed_iteration.txt")) -- left alone"; exit 0
fi
[ -e "$OUT/$it" ] && { echo "$ARM: removing the partial attempt $OUT/$it"; rm -rf "$OUT/$it"; }
mkdir -p "$OUT"
cd "$(dirname "$0")/../../.."
apptainer exec --bind /e --env PYTHONPATH=.:submodules/Megatron-LM:flash-linear-attention "$SIF" \
  python scripts/korbi/production/replace_ckda_with_attention.py apply \
  --src "$SRC" --dst "$OUT/$it" --layers "$LAYERS" --refs "$REFS" --pattern "$PATTERN" --seed 1234
echo "$IT" > "$OUT/latest_checkpointed_iteration.txt"
echo "$ARM: $OUT/$it <- $SRC (sublayers $LAYERS -> attention; pattern $PATTERN)"
