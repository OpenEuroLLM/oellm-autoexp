#!/usr/bin/env bash
# Build the hybrid-architecture Megatron image on NGC PyTorch (default 26.09: torch 2.14,
# TE 2.19; override with BASE_IMAGE=nvcr.io/nvidia/pytorch:<tag>) with
# NVRx, the CuTe DSL stack + cute-triadic-gdn and mamba_ssm / causal_conv1d
# (definition: MegatronTraining-hybrid.def.in). Needs apptainer with working
# --fakeroot (a host shell, not a user-namespaced one) and ~60 GB scratch space.
#
#   container/megatron/build_hybrid.sh [OUTPUT_DIR]   (from the feat/hybrid-v0.19 checkout)
#
# Then copy the .sif to marenostrum_new:/gpfs/projects/ehpc533/users/korbip/hybrid_v019/container/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HOME/.cache/containers}"
SCRATCH="${APPTAINER_SCRATCH:-$HOME/.cache}"
mkdir -p "$OUT" "$SCRATCH/apptainer_tmp" "$SCRATCH/apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/apptainer_tmp"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
export MAX_JOBS="${MAX_JOBS:-16}"   # parallel nvcc jobs for mamba_ssm / causal_conv1d
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.09-py3}"
cd "$HERE/.."
./build_container.sh \
    --definition MegatronTraining-hybrid-torch212 \
    --base-image nvcr.io/nvidia/pytorch:26.09-py3 \
    --output "$OUT" \
    --container-cmd apptainer 2>&1 | tee "$OUT/build_hybrid_torch214.log"
