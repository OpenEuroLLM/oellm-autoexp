#!/bin/bash
# Megatron torch_dist checkpoint -> HF safetensors, one checkpoint per array task.
# Submitted by eval_checkpoints.sh (which passes the Slurm options and MANIFEST).
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=convert
set -euo pipefail
ENTRY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
NAME=${ENTRY%%:*} CKPT=${ENTRY#*:}
echo "convert $NAME <- $CKPT"

apptainer exec --nv --bind "$CONTAINER_BIND" --env PYTHONNOUSERSITE=1,HF_HUB_OFFLINE=1 \
    --bind "$MEGATRON_REPO":/workspace/oellm-autoexp \
    --bind "$WORK_ROOT/bridge":/workspace/oellm-autoexp/submodules/Megatron-Bridge \
    "$BRIDGE_SIF" \
    python "$SCRIPT_DIR/convert.py" --megatron-path "$CKPT" --hf-path "$EXPORT_ROOT/$NAME" \
        --hf-model "$HF_MODEL" --tokenizer "$WORK_ROOT/tokenizer" \
        --bridge-root /workspace/oellm-autoexp/submodules/Megatron-Bridge \
        --derive-hf-arch "$DERIVE_HF_ARCH" --megatron-config "$ARCH_YAML"

# The HF eval container (transformers 4.53) needs the original-format tokenizer files and an explicit dtype.
cp "$WORK_ROOT"/tokenizer_src/* "$EXPORT_ROOT/$NAME/"
rm -f "$EXPORT_ROOT/$NAME/tokenizer.json"
python3 -c "import json,sys; p=sys.argv[1]; c=json.load(open(p)); c['torch_dtype']='bfloat16'; json.dump(c,open(p,'w'),indent=2)" \
    "$EXPORT_ROOT/$NAME/config.json"
# A Qwen3 chat template would make the evals apply a chat protocol the base model never saw.
[ ! -f "$EXPORT_ROOT/$NAME/chat_template.jinja" ] || \
    mv "$EXPORT_ROOT/$NAME/chat_template.jinja" "$EXPORT_ROOT/$NAME/chat_template.jinja.$DERIVE_HF_ARCH"
echo "done: $EXPORT_ROOT/$NAME"
