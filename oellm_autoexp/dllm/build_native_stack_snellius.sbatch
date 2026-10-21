#!/bin/bash
#SBATCH --job-name=dllm-native-stack
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
# Build the native (no-container) GPU stack of the dllm launcher on Snellius:
# torch 2.13.0+cu126 (the container's torch 2.13 line), transformers 5.14.1, and Transformer
# Engine 2.16.0 compiled with pip against that torch, with CUDA 12.6 and cuDNN 9.5 from the /sw
# modules and sm_90 only. The resulting pins are oellm_autoexp/dllm/requirements-native-snellius.txt.
#
# Create the venv first, then submit from the repository root:
#   python3 -m venv <venv dir> && ln -s <venv dir> .venv-gpu
#   sbatch -o "$OUTPUT_DIR/dllm-ablations/logs/p4/native-stack-%j.log" \
#       oellm_autoexp/dllm/build_native_stack_snellius.sbatch
# Reinstall exactly: .venv-gpu/bin/pip install -r oellm_autoexp/dllm/requirements-native-snellius.txt
# (torch from https://download.pytorch.org/whl/cu126; TE with --no-build-isolation and the same
# modules and NVTE_* variables as below).
set -euo pipefail

VENV="${VENV_GPU:-.venv-gpu}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/dllm-ablations/cache/pip}"
export TMPDIR="${TMPDIR:-$HOME/dllm-ablations/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

module purge
# NCCL is needed for its headers only (Transformer Engine includes nccl.h); at run time torch and
# Transformer Engine use the NCCL of torch's pip wheels.
module load 2024 Python/3.12.3-GCCcore-13.3.0 CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 \
    NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
export NCCL_HOME="$EBROOTNCCL"
export CUDA_HOME="$EBROOTCUDA"
export CUDA_PATH="$EBROOTCUDA"
export CUDNN_PATH="$EBROOTCUDNN"
export CUDNN_HOME="$EBROOTCUDNN"
export NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
echo "building on $(hostname) with CUDA_HOME=$CUDA_HOME CUDNN_PATH=$CUDNN_PATH MAX_JOBS=$MAX_JOBS"
nvcc --version | tail -2

"$VENV/bin/pip" install --upgrade pip setuptools wheel
"$VENV/bin/pip" install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu126
"$VENV/bin/pip" install "transformers==5.14.1" safetensors einops regex sentencepiece tiktoken \
    nltk pybind11 packaging six tensorboard wandb pyarrow ninja psutil
"$VENV/bin/pip" install --no-build-isolation \
    "transformer_engine[pytorch]==2.16.0" "transformer_engine_cu12==2.16.0"
# Transformer Engine 2.16 looks for the pip CUDA runtime in nvidia/cudart (or cu13/, cuda_cudart/),
# but the cu12 wheel installs it as nvidia/cuda_runtime; without the link TE's import fails with
# "cudart shared object not found" when no CUDA toolkit is loaded at run time.
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
[ -e "$SITE/nvidia/cudart" ] || ln -s cuda_runtime "$SITE/nvidia/cudart"
"$VENV/bin/pip" check || true
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
for name in ("torch", "transformer_engine", "transformer_engine_torch", "transformer_engine_cu12",
             "transformers", "triton", "nvidia-cudnn-cu12", "nvidia-nccl-cu12"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name} MISSING")
EOF
"$VENV/bin/pip" freeze > "${OUTPUT_DIR}/dllm-ablations/logs/p4/native-stack-freeze-${SLURM_JOB_ID}.txt"
echo "NATIVE_STACK_BUILD_DONE"
