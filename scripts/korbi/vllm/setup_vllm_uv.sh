#!/bin/bash
# Source this inside the pinned MN5 training image. Dependencies come from offline wheels.
set -euo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
UV_BIN=${OELLM_UV_BIN:-$R/tools/uv}
[ -x "$UV_BIN" ] || UV_BIN=$(command -v uv)
VLLM_SOURCE_ID=$(cat submodules/vllm/vllm/model_executor/models/oellm_ambitious.py \
 submodules/vllm/vllm/model_executor/layers/oellm_mhc.py \
 submodules/vllm/vllm/model_executor/layers/oellm_fusion.py \
 submodules/vllm/vllm/model_executor/layers/oellm_graph.py \
 submodules/vllm/vllm/model_executor/layers/oellm_uva_table.py | sha256sum | cut -c1-16)
VLLM_RUNTIME=/tmp/oellm_vllm032_$VLLM_SOURCE_ID
mkdir -p "$VLLM_RUNTIME"
export UV_CACHE_DIR=$VLLM_RUNTIME/uv-cache
export INFER_PY=$VLLM_RUNTIME/.venv/bin/python
export DEST=$VLLM_RUNTIME/pylibs_vllm
exec 9> "$VLLM_RUNTIME/.setup.lock"
flock 9
if [ ! -x "$INFER_PY" ]; then
 "$UV_BIN" venv --system-site-packages --python /usr/bin/python3 "$VLLM_RUNTIME/.venv"
fi
if [ ! -f "$DEST/.installed" ]; then
 mkdir -p "$DEST"
 "$UV_BIN" pip install --no-index --find-links "$R/wheels_vllm032" \
  --python "$INFER_PY" --target "$DEST" 'vllm==0.32.0' \
  > "$VLLM_RUNTIME/install.log" 2>&1 || { tail -40 "$VLLM_RUNTIME/install.log"; exit 1; }
 touch "$DEST/.installed"
fi
tar -C submodules/vllm/vllm --exclude='*.so' --exclude=__pycache__ -cf - . | tar -C "$DEST/vllm" -xf -
printf '%s\n' "$VLLM_SOURCE_ID" > "$DEST/RUNTIME_SOURCE_ID"
flock -u 9
exec 9>&-
export PYTHONPATH="$DEST:$PWD:$PWD/flash-linear-attention:$R/pylibs_flag:$R/pylibs_hfx"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export HF_HOME=${FLAG_HF_HOME:-/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home}
export HF_HUB_CACHE=$HF_HOME/hub HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics/job_${SLURM_JOB_ID:-local}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1 NLTK_DATA=$R/nltk_data TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OELLM_CPU_THREADS:-8}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
