#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
TASK_TMP=/tmp/oellm_infer008_${SLURM_JOB_ID:-local}
export UV_CACHE_DIR=$TASK_TMP/uv-cache
mkdir -p "$TASK_TMP"
"$R/tools/uv" venv --system-site-packages --python /usr/bin/python3 "$TASK_TMP/.venv"
export INFER_PY=$TASK_TMP/.venv/bin/python
export DEST=$TASK_TMP/pylibs_vllm
"$R/tools/uv" pip install --no-index --find-links "$R/wheels_vllm032" \
  --python "$INFER_PY" --target "$DEST" 'vllm==0.32.0' > "$TASK_TMP/install.log" 2>&1 || {
  tail -40 "$TASK_TMP/install.log"; exit 1;
}
tar -C submodules/vllm/vllm --exclude='*.so' --exclude=__pycache__ -cf - . | tar -C "$DEST/vllm" -xf -
export PYTHONPATH="$DEST:$PWD:$PWD/flash-linear-attention:$R/pylibs_flag:$R/pylibs_hfx"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export HF_HOME=/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=INFO OELLM_ENGRAM_TABLES=mmap
export OMP_NUM_THREADS=8
