#!/bin/bash
# Recall-surgery checkpoint surgery for one 1.7B testbed arm, inside the training image (1-node pool job,
# config/experiments/korbi/recall_surgery/ladder17b_surgery_mn5.yaml):
#   1. the patched code dir (scripts/korbi/recall_surgery/native_patch/make_code_dir.sh), built once -- code dirs
#      are immutable, an existing one is used as is
#   2. surgery.py: a MODEL-ONLY copy of the native stable-end checkpoint with the arm's new tensors (o_proj zero:
#      the model is the original one at step 0; intervention "" = the plain copy, the ctrl_fresh baseline)
# All arms in one job, one after another (they share the code dir, built once).
# Environment: CODE_DIR, ROOT (recall_surgery root), SEED, ARMS = ","-separated "name:model:intervention:init"
#   (shell-safe separators: the value is passed unquoted through the step's srun/singularity command line), e.g.
#   "pure_ctrl_fresh:skda-shipped-lowrank::random,pure_add_every4:skda-shipped-lowrank:add_attn@every4:random"
#   source: $ROOT/ladder17b_native/<model>; output: $ROOT/ladder17b_surgery/<name>
set -euo pipefail
: "${CODE_DIR:?}" "${ROOT:?}" "${ARMS:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
NATIVE_MEGATRON=${NATIVE_MEGATRON:-/gpfs/scratch/ehpc533/hybrid_nucleus/recall_surgery/native_megatron_ladder}
if [ ! -f "$CODE_DIR/lm_scaling/megatron_ext/interventions.py" ]; then
  bash "$HERE/native_patch/make_code_dir.sh" "$CODE_DIR"
fi
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # native checkpoints pickle numpy objects (args, rng)
export PYTHONPATH="$HERE/notitan:$NATIVE_MEGATRON:$CODE_DIR:$CODE_DIR/lm_scaling:$CODE_DIR"
status=0
IFS=',' read -ra items <<< "$ARMS"
for item in "${items[@]}"; do
  IFS=':' read -r name model intervention init <<< "$item"
  [ -n "$name" ] || continue
  src=$ROOT/ladder17b_native/$model; dst=$ROOT/ladder17b_surgery/$name
  if [ -f "$dst/latest_checkpointed_iteration.txt" ]; then echo "[ladder17b_surgery] $name: $dst exists, skipped"; continue; fi
  echo "[ladder17b_surgery] $name: arm=$model intervention='$intervention' init=${init:-random} $src -> $dst"
  SKDA_ARM=$model SKDA_INTERVENTION=$intervention python "$CODE_DIR/lm_scaling/megatron_ext/surgery.py" \
    --src "$src" --dst "$dst" --arm "$model" --intervention "$intervention" --init "${init:-random}" --seed "${SEED:-1234}" \
    || { echo "[ladder17b_surgery] FAIL: $name"; status=1; continue; }
  test -f "$dst/latest_checkpointed_iteration.txt" || { echo "[ladder17b_surgery] FAIL: $name wrote no checkpoint"; status=1; }
done
echo "[ladder17b_surgery] exit $status"
exit $status
