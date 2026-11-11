#!/bin/bash
# KV-group duplication surgery (docs/recall_surgery_ambitious.md §4.3): the main run's 6 attention layers
# 2 -> G KV groups by copying K/V heads (duplicate_kv_groups.py apply; function-preserving), one checkpoint
# per G: <ROOT>/kv<G>_src/ckpt/iter_<IT> = the source's shards hard-linked + the rewritten linear_qkv
# tensors / optimizer states / args, its latest_checkpointed_iteration.txt and kv_dup_surgery.json. Then
# `verify` (re-derive from the manifest, compare hashes) and `gpucheck` (the real layers through Megatron's
# attention on one GPU, source vs surgery, fp32 + bf16). A G whose checkpoint exists is not redone (checks
# still run). The source is only read. Run from a synced snapshot, as the 1-node pool job
# config/experiments/korbi/production/kv_dup_surgery_mn5.yaml.
#   kv_dup_surgery.sh [G ...]          (default: 8 16)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SIF=${SIF:-/gpfs/projects/ehpc533/users/bsc088649/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
SRC=${SRC:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph_hd256/ckpt_eval}
ROOT=${ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/attn_kv}
IT=${IT:-23400}
MOMENTS=${MOMENTS:-split}
GPUCHECK=${GPUCHECK:-1}
if [ -n "${SINGULARITY_CONTAINER:-}${APPTAINER_CONTAINER:-}" ]; then PY=(python3); else PY=(singularity exec --nv "$SIF" python3); fi
GROUPS_LIST=("$@")
[ ${#GROUPS_LIST[@]} -gt 0 ] || GROUPS_LIST=(8 16)
ITER=iter_$(printf %07d "$IT")
[ -d "$SRC/$ITER" ] || { echo "no $SRC/$ITER" >&2; exit 1; }
case $ROOT in */main_mhc4*) echo "refusing to write below the main run ($ROOT)" >&2; exit 2 ;; esac

for g in "${GROUPS_LIST[@]}"; do
  dst=$ROOT/kv${g}_src/ckpt
  if [ -f "$dst/latest_checkpointed_iteration.txt" ]; then
    echo "kv$g: $dst exists, surgery skipped"
  else
    mkdir -p "$dst"
    rm -rf "${dst:?}/$ITER"  # a partial earlier attempt (no latest file yet)
    echo "kv$g: $SRC/$ITER -> $dst/$ITER (moments $MOMENTS)"
    CUDA_VISIBLE_DEVICES= "${PY[@]}" "$HERE/duplicate_kv_groups.py" apply \
      --src "$SRC/$ITER" --dst "$dst/$ITER" --groups "$g" --moments "$MOMENTS"
    echo "$IT" > "$dst/latest_checkpointed_iteration.txt"
  fi
  CUDA_VISIBLE_DEVICES= "${PY[@]}" "$HERE/duplicate_kv_groups.py" verify --dst "$dst/$ITER"
  if [ "$GPUCHECK" = 1 ]; then
    "${PY[@]}" "$HERE/duplicate_kv_groups.py" gpucheck --src "$SRC/$ITER" --dst "$dst/$ITER" --fused-qk
  fi
done
