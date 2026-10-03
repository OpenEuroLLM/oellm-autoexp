#!/bin/bash
# Megatron torch_dist checkpoint -> HF safetensors, one checkpoint per array task.
# Submitted by eval_checkpoints.sh (which passes the Slurm options and MANIFEST).
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --job-name=convert
set -euo pipefail
ENTRY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
"$SCRIPT_DIR/convert_export.sh" "${ENTRY%%:*}" "${ENTRY#*:}"
