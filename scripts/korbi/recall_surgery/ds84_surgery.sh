#!/bin/bash
# Checkpoint steps of the 84B dropout-stitch session (docs/recall_surgery_ambitious.md §19; the small-scale arms D1 / D2 /
# DK1 / DK2 of §14.8 on the 84B model, from the 23300 broad-session geometry). Each name writes an arm's OWN start
# checkpoint <ROOT>/<arm>/ckpt/iter_<N> (+ its tracker, only after the step verified), so the training run of
# config/experiments/korbi/recall_surgery/dropout_stitch84_mn5.yaml runs with load = save = <ROOT>/<arm>/ckpt:
#   d6     -> D1    the main run's ckpt_eval/iter_0023300 + ADDITIVE attention branches on complex_kda 9 / 25 / 41 / 58 /
#                   74 / 90 (the 3:1 positions = small31's E layout), each a COPY of the nearest preceding attention layer
#                   0 / 17 / 33 / 50 / 66 / 82 (add_ckda_branch.py --init attn: linear_qkv / linear_proj / q,k norm and
#                   their optimizer states, input-norm gain folded), gamma 0 (exact at insertion), no calib -- as small31 D1
#   dk6    -> DK1   the same positions with ADDITIVE kvshare branches (add_ckda_branch.py --branch kvshare
#                   --kvshare-init copy: q / gate / o / q norm copied from the source layer, input norm folded; keys /
#                   values = the source layer's at run time), gamma 0 -- as small31 K / DK1
#   ctrl2  -> ctrl2 the broad session's ctrl at CTRL_IT (v4_broad23k/ctrl/ckpt_eval/iter_<CTRL_IT>, hard links): the paired
#                   control of the stitch leg (re-runs ctrl's CTRL_IT -> +400 on today's code)
#   d2     -> D2    D1's trained branches (+ gamma + their optimizer states; D1@CTRL_IT) transplanted into ctrl@CTRL_IT
#                   (scripts/korbi/production/transplant_attention.py, branch donor: the donor's input-norm gain folded
#                   into branch.linear_qkv against ctrl's), the rest of the model = ctrl's -- as small31 D2
#   dk2    -> DK2   DK1's kvshare branches into ctrl@CTRL_IT, the same way -- as small31 DK2
#   d6h    -> D1h   D1's start (hard links of D1/ckpt/iter_<IT>, after d6), trained at rest dropout 0.5 -- as small31 D1h
#   d2h    -> D2h   D1h's branches into ctrl@CTRL_IT -- as small31 D2h
# Deterministic (no randomness in --init attn / --kvshare-init copy / the transplant). A target whose tracker holds the
# expected iteration is skipped; another iteration is an error; a partial attempt (no tracker) is removed and redone.
# 1-node pool jobs of config/experiments/korbi/recall_surgery/ds84_surgery_mn5.yaml (training image, synced snapshot).
#   ds84_surgery.sh d6|dk6|ctrl2|d2|dk2|d6h|d2h [...]        (DRY=1: print the commands only)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROD=$HERE/../production
SIF=${SIF:-/gpfs/projects/ehpc533/users/bsc088649/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
ROOT=${ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/dropout_stitch84}
SRC=${SRC:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_eval}
CTRL=${CTRL:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_broad23k/ctrl}
IT=${IT:-23300}            # the branch-insertion start (the main run's frozen 23300, as every broad-session arm)
CTRL_IT=${CTRL_IT:-24400}  # the stitch step: D1 / DK1 train IT -> CTRL_IT, ctrl@CTRL_IT exists (broad session +1100)
LAYERS=${LAYERS:-9,25,41,58,74,90}
SOURCES=${SOURCES:-0,17,33,50,66,82}
DRY=${DRY:-0}
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then PY=(python3); else PY=(singularity exec --nv "$SIF" python3); fi
[ "$DRY" = 1 ] && PY=(echo python3)
case $ROOT in */main_mhc4*|*/v4_broad23k*) echo "refusing to write below $ROOT (a source tree)" >&2; exit 2 ;; esac
[ $# -gt 0 ] || { echo "usage: $0 d6|dk6|ctrl2|d2|dk2|d6h|d2h ..." >&2; exit 2; }
iter() { printf 'iter_%07d' "$1"; }

# done_at <ckpt dir> <it>: 0 if its tracker holds it, 1 if absent, exit 3 on another iteration
done_at() {
  local t=$1/latest_checkpointed_iteration.txt
  [ -f "$t" ] || return 1
  local v; v=$(tr -d '[:space:]' < "$t")
  [ "$v" = "$2" ] && return 0
  echo "$t holds $v, not $2: the arm has trained on (or another session's output) -- left alone" >&2
  exit 3
}

# the newest save of a run dir at iteration <it> (persistent, rolling or eval copy)
save_of() {
  local d
  for d in "$1/ckpt" "$1/ckpt_rolling" "$1/ckpt_eval"; do
    [ -f "$d/$(iter "$2")/.metadata" ] && { echo "$d/$(iter "$2")"; return 0; }
  done
  return 1
}

for name in "$@"; do
  case $name in
    d6) arm=D1; at=$IT ;;
    dk6) arm=DK1; at=$IT ;;
    ctrl2) arm=ctrl2; at=$CTRL_IT ;;
    d2) arm=D2; at=$CTRL_IT; donor_arm=D1 ;;
    dk2) arm=DK2; at=$CTRL_IT; donor_arm=DK1 ;;
    d6h) arm=D1h; at=$IT ;;
    d2h) arm=D2h; at=$CTRL_IT; donor_arm=D1h ;;
    *) echo "unknown step $name" >&2; exit 2 ;;
  esac
  dst=$ROOT/$arm/ckpt
  out=$dst/$(iter "$at")
  if done_at "$dst" "$at"; then
    echo "$name: $out exists (tracker $at) -- skipped"; continue
  fi
  [ "$DRY" = 1 ] || { mkdir -p "$dst"; rm -rf "${out:?}"; }  # a partial earlier attempt (no tracker yet)
  case $name in
    d6|dk6)
      src=$SRC/$(iter "$IT")
      [ "$DRY" = 1 ] || [ -f "$src/.metadata" ] || { echo "$name: no $src/.metadata" >&2; exit 1; }
      if [ $name = d6 ]; then
        args=(--branch attention --groups 2 --init attn --layers "$LAYERS" --kv-sources "$SOURCES" --seed 1234)
      else
        args=(--branch kvshare --groups 2 --kvshare-init copy --layers "$LAYERS" --kv-sources "$SOURCES" --seed 1234)
      fi
      echo "$name: $src -> $out (${args[*]})"
      CUDA_VISIBLE_DEVICES= "${PY[@]}" "$PROD/add_ckda_branch.py" apply --src "$src" --dst "$out" "${args[@]}"
      CUDA_VISIBLE_DEVICES= "${PY[@]}" "$PROD/add_ckda_branch.py" verify --dst "$out"
      ;;
    ctrl2)
      src=$(save_of "$CTRL" "$CTRL_IT") || { [ "$DRY" = 1 ] && src=$CTRL/ckpt_eval/$(iter "$CTRL_IT"); } || {
        echo "ctrl2: no ctrl save at $CTRL_IT under $CTRL" >&2; exit 1; }
      echo "ctrl2: $src -> $out (hard links)"
      if [ "$DRY" = 1 ]; then echo "+ cp -al $src $out"; else
        cp -al "$src" "$out"
        [ "$(find "$src" -type f | wc -l)" = "$(find "$out" -type f | wc -l)" ] || { echo "ctrl2: file count" >&2; exit 1; }
      fi
      ;;
    d6h)
      src=$ROOT/D1/ckpt/$(iter "$IT")
      [ "$DRY" = 1 ] || [ -f "$src/.metadata" ] || { echo "d6h: no $src/.metadata (run d6 first)" >&2; exit 1; }
      echo "d6h: $src -> $out (hard links)"
      if [ "$DRY" = 1 ]; then echo "+ cp -al $src $out"; else
        cp -al "$src" "$out"
        [ "$(find "$src" -type f | wc -l)" = "$(find "$out" -type f | wc -l)" ] || { echo "d6h: file count" >&2; exit 1; }
      fi
      ;;
    d2|dk2|d2h)
      base=$(save_of "$CTRL" "$CTRL_IT") || { [ "$DRY" = 1 ] && base=$CTRL/ckpt_eval/$(iter "$CTRL_IT"); } || {
        echo "$name: no ctrl save at $CTRL_IT under $CTRL" >&2; exit 1; }
      donor=$(save_of "$ROOT/$donor_arm" "$CTRL_IT") || {
        [ "$DRY" = 1 ] && donor=$ROOT/$donor_arm/ckpt/$(iter "$CTRL_IT") || {
          echo "$name: $donor_arm has no save at $CTRL_IT yet" >&2; exit 1; }; }
      echo "$name: $donor_arm@$CTRL_IT ($donor) branches -> ctrl@$CTRL_IT ($base) -> $out"
      CUDA_VISIBLE_DEVICES= "${PY[@]}" "$PROD/transplant_attention.py" apply --base "$base" --donor "$donor" \
        --dst "$out" --layers "$LAYERS"
      ;;
  esac
  if [ "$DRY" = 1 ]; then echo "+ echo $at > $dst/latest_checkpointed_iteration.txt"; else
    echo "$at" > "$dst/latest_checkpointed_iteration.txt"
    printf '{"arm": "%s", "start": %s, "step": "%s", "seeded": "%s"}\n' "$arm" "$at" "$name" "$(date -Is)" \
      > "$dst/seeded_from.json"
  fi
  echo "$name: done -> $out"
done
