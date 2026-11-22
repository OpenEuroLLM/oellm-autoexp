#!/bin/bash
# Run a module of this folder inside the Clariden container on a debug node.
# The login nodes have no torch; the training container (nemo-26.08) has torch, transformers and FLA.
#
#   oellm_autoexp/hf_export/engram_scaling_clariden/run_in_container.sh <module> [args...]
#   e.g. ... run_in_container.sh inspect_megatron_ckpt --ckpt-dir <dir> --config <yaml>
#
# Env: PARTITION (default debug), TIME (default 00:20:00), GPUS (default 0),
#      HF_HOME (default /iopsstor/scratch/cscs/$USER/hf_home, where the tokenizer is cached),
#      HF_HUB_OFFLINE (default 1),
#      LAUNCHER (default python; "torchrun --standalone --nproc-per-node 1" for megatron_parity).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MODULE=$1
shift

GPU_ARGS=()
if [ "${GPUS:-0}" != "0" ]; then
  GPU_ARGS=(--gpus-per-node="$GPUS")
fi

srun -A a0248 -p "${PARTITION:-debug}" -N1 -n1 --time="${TIME:-00:20:00}" "${GPU_ARGS[@]}" \
  --environment="$ROOT/container/alps/nemo-26.08.toml" \
  bash -c 'cd "$0" && export PATH=/opt/venv/bin:$PATH PYTHONPATH=.:submodules/Megatron-LM && \
    export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/hf_home} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} && \
    $1 -m "oellm_autoexp.hf_export.engram_scaling_clariden.$2" "${@:3}"' \
  "$ROOT" "${LAUNCHER:-python}" "$MODULE" "$@"
