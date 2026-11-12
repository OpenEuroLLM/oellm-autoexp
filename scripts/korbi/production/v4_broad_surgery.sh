#!/bin/bash
# Checkpoint surgeries of the broad session (docs/recall_surgery_ambitious.md §10): the main run's start checkpoint
# ckpt_eval/iter_<IT> (read only; the runbook hard-links the pre-empted main run's final save there) ->
#   <ROOT>/src/<name>/ckpt/iter_<IT>  (the source's shards hard-linked + the new / rewritten tensors and the manifest)
#   <ROOT>/src/<name>/ckpt/latest_checkpointed_iteration.txt  (= IT, written only after the surgery verified)
# names:
#   kv8, kv16  the 6 attention layers 2 -> 8 / 16 KV groups (duplicate_kv_groups.py, moments split; §4.3), then
#              verify (re-derive, compare hashes) and gpucheck (the real layers through Megatron's attention, 1 GPU)
#   post4      additive attention branches on CKDA 19 / 35 / 52 / 68, COPIES of attention 17 / 33 / 50 / 66 (input-norm
#              gain folded, all optimizer states), gamma 0, calib 0, 2 KV groups (add_ckda_branch.py --init attn; §4.4)
#   kv8post4   post4 on top of kv8's output: the branches copy the 8-group stack layers verbatim (+ norm fold), so
#              the branches have 8 KV groups like the stack (branch_add export and the one global
#              ckda_branch_num_query_groups need branch groups == stack groups). Needs kv8 first (its tracker).
#   conv       the crossfade conversion: attention branches on CKDA 74 / 90, copies of attention 66 / 82, gamma 0,
#              calib 0, 2 KV groups; the arm runs ckda_branch_crossfade (weight (1 - alpha)(1 + gamma) calib: exact
#              at alpha 1, the branch alone at alpha 0)
# Every step is deterministic (no randomness in --init attn / the KV copy): a rerun reproduces every tensor. An
# output whose tracker holds IT is not redone (its checks still run); a tracker with another iteration is an error
# (another session's ROOT); a partial attempt (no tracker) is removed and redone. Run from a synced snapshot as the
# 1-node pool jobs of config/experiments/korbi/production/v4_broad_surgery_mn5.yaml.
#   IT=<iteration> v4_broad_surgery.sh <name> [...]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SIF=${SIF:-/gpfs/projects/ehpc533/users/bsc088649/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
SRC=${SRC:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_eval}
ROOT=${ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_broad}
IT=${IT:?IT = the start iteration (the main run checkpoint in SRC/iter_<IT>)}
GPUCHECK=${GPUCHECK:-1}
VERIFY=${VERIFY:-1}
DRY=${DRY:-0}  # 1: print the commands only
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then PY=(python3); else PY=(singularity exec --nv "$SIF" python3); fi
[ "$DRY" = 1 ] && PY=(echo python3)
ITER=iter_$(printf %07d "$IT")
case $ROOT in */main_mhc4*) echo "refusing to write below the main run ($ROOT)" >&2; exit 2 ;; esac
[ $# -gt 0 ] || { echo "usage: IT=<it> $0 kv8|kv16|post4|kv8post4|conv ..." >&2; exit 2; }

# done_at <ckpt dir>: 0 if its tracker holds IT, 1 if absent, exit 3 on another iteration
done_at() {
  local t=$1/latest_checkpointed_iteration.txt
  [ -f "$t" ] || return 1
  local v; v=$(tr -d '[:space:]' < "$t")
  [ "$v" = "$IT" ] && return 0
  echo "$t holds $v, not $IT: another session's output in $ROOT (use a new ROOT)" >&2
  exit 3
}

for name in "$@"; do
  src=$SRC
  case $name in
    kv8|kv16) tool=kv; groups=${name#kv} ;;
    post4) tool=branch; args=(--branch attention --groups 2 --layers 19,35,52,68 --init attn --kv-sources 17,33,50,66 --calibrate) ;;
    kv8post4)
      tool=branch; src=$ROOT/src/kv8/ckpt
      [ "$DRY" = 1 ] || done_at "$src" || { echo "kv8post4: kv8 is not done ($src/latest_checkpointed_iteration.txt)" >&2; exit 1; }
      args=(--branch attention --groups 8 --layers 19,35,52,68 --init attn --kv-sources 17,33,50,66 --calibrate) ;;
    conv) tool=branch; args=(--branch attention --groups 2 --layers 74,90 --init attn --kv-sources 66,82 --calibrate) ;;
    *) echo "unknown surgery $name" >&2; exit 2 ;;
  esac
  [ "$DRY" = 1 ] || [ -f "$src/$ITER/.metadata" ] || { echo "$name: no $src/$ITER/.metadata" >&2; exit 1; }
  dst=$ROOT/src/$name/ckpt
  if done_at "$dst"; then
    echo "$name: $dst/$ITER exists (tracker $IT), surgery skipped"
  else
    [ "$DRY" = 1 ] || { mkdir -p "$dst"; rm -rf "${dst:?}/$ITER"; }  # a partial earlier attempt (no tracker yet)
    echo "$name: $src/$ITER -> $dst/$ITER"
    if [ $tool = kv ]; then
      CUDA_VISIBLE_DEVICES= "${PY[@]}" "$HERE/duplicate_kv_groups.py" apply \
        --src "$src/$ITER" --dst "$dst/$ITER" --groups "$groups" --moments split
      [ "$VERIFY" = 1 ] && CUDA_VISIBLE_DEVICES= "${PY[@]}" "$HERE/duplicate_kv_groups.py" verify --dst "$dst/$ITER"
    else
      CUDA_VISIBLE_DEVICES= "${PY[@]}" "$HERE/add_ckda_branch.py" apply --src "$src/$ITER" --dst "$dst/$ITER" "${args[@]}"
      [ "$VERIFY" = 1 ] && CUDA_VISIBLE_DEVICES= "${PY[@]}" "$HERE/add_ckda_branch.py" verify --dst "$dst/$ITER"
    fi
    [ "$DRY" = 1 ] || echo "$IT" > "$dst/latest_checkpointed_iteration.txt"
  fi
  if [ $tool = kv ] && [ "$GPUCHECK" = 1 ]; then
    "${PY[@]}" "$HERE/duplicate_kv_groups.py" gpucheck --src "$src/$ITER" --dst "$dst/$ITER" --fused-qk
  fi
done
