#!/bin/bash
# Phase-2 insertion of the small-scale testbed (docs/recall_surgery_ambitious.md §14): a b7 (7:1) checkpoint -> a new
# arm's start checkpoint with random-init softmax attention at more CKDA positions, FUNCTION-PRESERVING (2026-11-14:
# the hard insertion -- replace_ckda_with_attention.py, kept as HARD_E / HARD_Ep -- blew up after one update, loss
# 2.9 -> 34 -> 63): a random attention ckda_branch on each of those CKDA sublayers (add_ckda_branch.py --init random:
# rows at the reference attention's median row norms, fresh MuOwn / Adam states), run as a crossfade
# (ckda_branch_crossfade: (alpha) ckda + (1 - alpha) branch, alpha 1 -> 0 over the ramp): step 0 is exactly b7, the
# branch's gradients and weight grow with (1 - alpha) -- the new parameters' warmup; past the ramp the sublayer is the
# attention alone (exported as plain attention).
#   <ROOT>/<ARM>/ckpt/iter_<IT>  (hard links of b7's + the new tensors) and its tracker (= IT, after the read-back)
# Arms (sublayer indices of the 50-sublayer small31 pattern; mixer m -> sublayer):
#   E   3:1 as r3:  new attention at mixers 4 / 12 / 20  = sublayers 9 / 25 / 42, references 0 / 17 / 34
#   Ep  paired as r3p: new attention at mixers 1 / 9 / 17 = sublayers 3 / 19 / 36, references 0 / 17 / 34
#   L / Lp  the same random insertion LATE (IT = b7's late checkpoint)
#   C   b7's checkpoint as it is (hard links): the late control
#   M / Mp  E's / Ep's trained new attention (DONOR_IT of E / Ep) transplanted into b7@IT as crossfade ckda_branch
#           attention branches at the same sublayers (scripts/korbi/production/transplant_attention.py)
# The first CKDA mixers (complex eigenvalues) stay CKDA in both. Login node (hard links + a few MB of new tensors).
#   ARM=E IT=1907 small31_insert.sh;  ARM=M IT=38146 DONOR_IT=9535 small31_insert.sh;  ARM=C IT=38146 small31_insert.sh
set -euo pipefail
: "${ARM:?E | Ep}" "${IT:?b7 iteration}"
ROOT=${ROOT:-/e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/small31}
SRC_ARM=${SRC_ARM:-b7}
SIF=${SIF:-/e/project1/e-sta-openeurollm/poeppel1/container_cache/MegatronTraining-JUPITER_pt2512_aarch64_202602231451.sif}
B7="*-NL-L-L-L-L-L-L-*-L-L-L-L-L-L-L-N*-L-L-L-L-L-L-L-"
case "$ARM" in
  E|L|M)   LAYERS=9,25,42; PATTERN="*-NL-L-L-*-L-L-L-*-L-L-L-*-L-L-L-N*-L-L-L-*-L-L-L-"; DONOR=E ;;
  Ep|Lp|Mp) LAYERS=3,19,36; PATTERN="*-N*-L-L-L-L-L-L-*-*-L-L-L-L-L-L-N*-*-L-L-L-L-L-L-"; DONOR=Ep ;;
  C) LAYERS=; PATTERN= ;;
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
X=(apptainer exec --bind /e --env PYTHONPATH=.:submodules/Megatron-LM:flash-linear-attention "$SIF")
case "$ARM" in
  C) cp -al "$SRC" "$OUT/$it" ;;
  M|Mp)
    : "${DONOR_IT:?the donor E / Ep checkpoint iteration}"
    din=$(printf 'iter_%07d' "$DONOR_IT")
    [ -f "$ROOT/$DONOR/ckpt/$din/.metadata" ] || { echo "no $din of $DONOR" >&2; exit 1; }
    "${X[@]}" python scripts/korbi/production/transplant_attention.py apply --base "$SRC" \
      --donor "$ROOT/$DONOR/ckpt/$din" --dst "$OUT/$it" --layers "$LAYERS" ;;
  HARD_E|HARD_Ep) "${X[@]}" python scripts/korbi/production/replace_ckda_with_attention.py apply \
      --src "$SRC" --dst "$OUT/$it" --layers "$LAYERS" --refs "$REFS" --pattern "$PATTERN" --seed 1234 ;;
  *) # function-preserving: a random attention BRANCH on each CKDA sublayer (crossfade from the CKDA output, branch
     # weight 0 at the insertion; add_ckda_branch.py --init random, rows at the reference attention's norms)
     "${X[@]}" env CKDA_BRANCH_ATTN_LAYERS=$REFS python scripts/korbi/production/add_ckda_branch.py apply \
       --src "$SRC" --dst "$OUT/$it" --branch attention --init random --groups 2 --layers "$LAYERS" \
       --kv-sources "$REFS" --seed 1234 ;;
esac
echo "$IT" > "$OUT/latest_checkpointed_iteration.txt"
echo "$ARM: $OUT/$it <- $SRC (sublayers ${LAYERS:-none}; pattern ${PATTERN:-unchanged})"
