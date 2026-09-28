#!/bin/bash
# 1:1 parity check: Megatron checkpoint vs the converted HF model.
#
#   bash scripts/korbi/check_hf_megatron_parity.sh <train_run_dir> [hf_dir] [seq_len] [batch]
#
# Runs the SAME token ids through both paths and reports the max absolute logit
# difference and the KL between the two next-token distributions:
#
#   1. Megatron side — `dump_megatron_logits.py` inside the training container,
#      building the model from the run's own `job.sbatch` flags and loading its
#      `torch_dist` checkpoint.
#   2. HF side — `validate_hf_export.py --reference`, which loads the converted
#      dir with `trust_remote_code` and diffs against that dump.
#
# Both halves run in one GPU allocation, so it is a single srun.
#
# Token ids come from the pretokenized training mix (real in-distribution ids,
# not random) so the comparison exercises the model where it is actually used.
set -euo pipefail

RUN_DIR="${1:?usage: check_hf_megatron_parity.sh <train_run_dir> [hf_dir] [seq_len] [batch]}"
HF_DIR="${2:-}"
SEQ_LEN="${3:-512}"
BATCH="${4:-2}"

REPO="${REPO:-$HOME/work/Projects/oellm-autoexp-hybrid}"
VENV="${OELLM_EVAL_VENV:-$HOME/work/venv}"
CONTAINER="${PARITY_CONTAINER:-${CONTAINER_CACHE_DIR:-/e/project1/e-sta-openeurollm/container}/MegatronTraining-JUPITER-linattn-mamba_aarch64_202608010917.sif}"
DATA_PREFIX="${PARITY_DATA_PREFIX:-/e/data1/datasets/products/openeurollm/pretokenized/openeurollm-tokenized-256k/dclm-baseline-1.0-10p-sample/dclm-10p-sample}"
WORK="${PARITY_WORKDIR:-${OUTPUT_DIR:-/tmp}/parity/$(basename "$RUN_DIR")}"

if [[ -z "$HF_DIR" ]]; then
    HF_DIR="$(dirname "$RUN_DIR")_hf/$(basename "$RUN_DIR")"
fi

mkdir -p "$WORK"
echo "run dir   : $RUN_DIR"
echo "hf dir    : $HF_DIR"
echo "workdir   : $WORK"
echo "container : $CONTAINER"

# --- 1. fixed token ids straight out of the pretokenized mix ----------------
"$VENV/bin/python" - "$DATA_PREFIX" "$WORK/tokens.npy" "$SEQ_LEN" "$BATCH" <<'PY'
import sys, numpy as np
sys.path.insert(0, __import__("os").environ.get("REPO", "."))
from oellm_autoexp.hf_export.eval_on_megatron_data import load_spans
prefix, out, seq_len, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
from pathlib import Path
spans = load_spans(Path(prefix), seq_len, batch, skip=1_000_000)
np.save(out, spans)
print(f"tokens {spans.shape} -> {out}")
PY

# --- 2. Megatron logits (inside the training container) ---------------------
cd "$REPO"
apptainer exec --nv \
    --bind /e --bind /dev/shm \
    --bind /usr/lib64/libcuda.so.1:/opt/nvidia/lib/libcuda.so.1:ro \
    --bind /usr/lib64/libcuda.so:/opt/nvidia/lib/libcuda.so:ro \
    --bind /usr/lib64/libnvidia-ml.so.1:/opt/nvidia/lib/libnvidia-ml.so.1:ro \
    --env PYTHONPATH=".:submodules/Megatron-LM" \
    --env TRITON_LIBCUDA_PATH=/opt/nvidia/lib \
    --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
    "$CONTAINER" \
    /usr/bin/python scripts/korbi/dump_megatron_logits.py \
        --run-dir "$RUN_DIR" \
        --tokens "$WORK/tokens.npy" \
        --out "$WORK/megatron_logits.pt"

# --- 3. HF logits + comparison ----------------------------------------------
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
"$VENV/bin/python" -m oellm_autoexp.hf_export.validate_hf_export \
    --model-dir "$HF_DIR" \
    --reference "$WORK/megatron_logits.pt"
