#!/bin/bash
# Megatron torch_dist checkpoint -> HF safetensors export, as used for the published 32B results.
#
#   convert_export.sh NAME CHECKPOINT_DIR [ARCH_YAML]
#
# Called by convert.sbatch (one manifest entry per array task) and by oellm-autoexp `convert`
# stages (config/experiments/korbi/chain_flag_130M_jupiter.yaml). ARCH_YAML is a yaml with
# backend.megatron.* (a model profile's *.arch.yaml, or a training run's resolved current.yaml);
# default $ARCH_YAML. Writes $EXPORT_ROOT/NAME and, last, $EXPORT_ROOT/NAME/.convert_done, which is
# what downstream stages wait for.
#
# Environment: WORK_ROOT (holds tokenizer/, tokenizer_src/, bridge/ from `eval_checkpoints.sh
# prepare`), EXPORT_ROOT, BRIDGE_SIF, MEGATRON_REPO, HF_MODEL, DERIVE_HF_ARCH, CONTAINER_BIND.
set -euo pipefail
NAME="${1:?usage: convert_export.sh NAME CHECKPOINT_DIR [ARCH_YAML]}" CKPT="${2:?checkpoint dir}"
ARCH="${3:-$ARCH_YAML}"
HERE=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
for need in "$WORK_ROOT/tokenizer/tokenizer.json" "$WORK_ROOT/bridge/src" "$CKPT" "$ARCH"; do
    [ -e "$need" ] || { echo "error: missing $need (eval_checkpoints.sh prepare / wrong path)" >&2; exit 1; }
done
echo "convert $NAME <- $CKPT (arch: $ARCH)"

apptainer exec --nv --bind "$CONTAINER_BIND" --env PYTHONNOUSERSITE=1,HF_HUB_OFFLINE=1 \
    --bind "$MEGATRON_REPO":/workspace/oellm-autoexp \
    --bind "$WORK_ROOT/bridge":/workspace/oellm-autoexp/submodules/Megatron-Bridge \
    "$BRIDGE_SIF" \
    python "$HERE/convert.py" --megatron-path "$CKPT" --hf-path "$EXPORT_ROOT/$NAME" \
        --hf-model "$HF_MODEL" --tokenizer "$WORK_ROOT/tokenizer" \
        --bridge-root /workspace/oellm-autoexp/submodules/Megatron-Bridge \
        --derive-hf-arch "$DERIVE_HF_ARCH" --megatron-config "$ARCH"

# The HF eval container (transformers 4.53) needs the original-format tokenizer files and an explicit dtype.
cp "$WORK_ROOT"/tokenizer_src/* "$EXPORT_ROOT/$NAME/"
rm -f "$EXPORT_ROOT/$NAME/tokenizer.json"
python3 -c "import json,sys; p=sys.argv[1]; c=json.load(open(p)); c['torch_dtype']='bfloat16'; json.dump(c,open(p,'w'),indent=2)" \
    "$EXPORT_ROOT/$NAME/config.json"
# A Qwen3 chat template would make the evals apply a chat protocol the base model never saw.
[ ! -f "$EXPORT_ROOT/$NAME/chat_template.jinja" ] || \
    mv "$EXPORT_ROOT/$NAME/chat_template.jinja" "$EXPORT_ROOT/$NAME/chat_template.jinja.$DERIVE_HF_ARCH"
date -Is > "$EXPORT_ROOT/$NAME/.convert_done"
echo "done: $EXPORT_ROOT/$NAME"
