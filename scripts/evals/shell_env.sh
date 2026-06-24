#!/usr/bin/env bash
# Source this file INSIDE a singularity shell to replicate the eval job environment.
#
# Usage:
#   # From the host — launch the container:
#   IMG=/shared_silo/scratch/containers/vllm-dev_preview_releases_v0.20.0_20260422.sif
#   singularity shell --rocm -B /shared_silo/scratch:/shared_silo/scratch:rw \
#                             -B /usr/share/libdrm:/usr/share/libdrm:ro "$IMG"
#
#   # Then inside the container:
#   source "$(git rev-parse --show-toplevel)/scripts/evals/shell_env.sh"
#
# Override any variable before sourcing, e.g.:
#   MODEL=/path/to/model source shell_env.sh

# ---------------------------------------------------------------------------
# GPU / CUDA
# ---------------------------------------------------------------------------
export HIP_VISIBLE_DEVICES=0
unset CUDA_VISIBLE_DEVICES

# ---------------------------------------------------------------------------
# lm-eval paths  (mirrors eval_lmeval_preemptible.sbatch)
# ---------------------------------------------------------------------------
_EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
LOCAL_LMEVAL="${LOCAL_LMEVAL:-$(cd "${_EVAL_SCRIPT_DIR}/../.." 2>/dev/null && pwd)/submodules/lm-evaluation-harness}"
LMEVAL_ENV="${LMEVAL_ENV:-/shared_silo/scratch/shared/tw-dashboard/lm-eval-env-v0.4.11}"
LMEVAL_SITE="${LMEVAL_ENV}/lib/python3.12/site-packages"

# Local checkout takes precedence; shared venv provides Python + vLLM + deps
export PYTHONPATH="${LOCAL_LMEVAL}:${LMEVAL_SITE}:${PYTHONPATH:-}"
export PATH="${LMEVAL_ENV}/bin:${PATH}"

# Clean stale user-local packages that could shadow the above
rm -rf ~/.local/lib/python3.12/site-packages 2>/dev/null || true

# ---------------------------------------------------------------------------
# Eval settings (set these before sourcing, or override afterwards)
# ---------------------------------------------------------------------------
SCRATCH="${SCRATCH:-/shared_silo/scratch/${USER}}"
export MODEL="${MODEL:-}"
export TASKS="${TASKS:-}"
export NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
export INCLUDE_PATH="${INCLUDE_PATH:-${_EVAL_SCRIPT_DIR}/tasks}"

# ---------------------------------------------------------------------------
# Dashboard / tracking
# ---------------------------------------------------------------------------
export WANDB_API_KEY=local
export FASTTRACKML_URL="${FASTTRACKML_URL:-http://tus1-vm-amd-k8s-001:5000}"
export DASHBOARD_URL="${DASHBOARD_URL:-http://tus1-vm-amd-k8s-001:8080}"
export DASHBOARD_API_KEY="${DASHBOARD_API_KEY:-${TW_API_KEY:-}}"
export EVALS_DIR="${EVALS_DIR:-/shared_silo/scratch/shared/tw-dashboard/evals}"

# lm-eval request cache
export LM_HARNESS_CACHE_PATH="${SCRATCH}/lm_eval_request_cache"

# ---------------------------------------------------------------------------
# Convenience: print active config
# ---------------------------------------------------------------------------
echo "=== lm-eval shell environment ==="
python -c "import lm_eval, importlib.metadata as m; print(f'lm-eval : {m.version(\"lm_eval\")} from {lm_eval.__file__}')" 2>/dev/null \
    || echo "lm-eval : (import failed — check PYTHONPATH)"
echo "PYTHONPATH  : ${PYTHONPATH}"
echo "MODEL       : ${MODEL:-(not set)}"
echo "TASKS       : ${TASKS:-(not set)}"
echo "INCLUDE_PATH: ${INCLUDE_PATH}"
echo "SCRATCH     : ${SCRATCH}"
echo ""
echo "Run evals with:"
echo "  python -m lm_eval --model vllm \\"
echo "    --model_args \"pretrained=\${MODEL},gpu_memory_utilization=\${GPU_MEM_UTIL}\" \\"
echo "    --tasks \"\${TASKS}\" --num_fewshot \"\${NUM_FEWSHOT}\" \\"
echo "    --batch_size auto --output_path \"\${SCRATCH}/eval_results/debug\" \\"
echo "    --include_path \"\${INCLUDE_PATH}\""
