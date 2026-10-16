#!/usr/bin/env bash
# Build the upstream-torchtitan image (definition: TitanTraining-main.def.in) on NGC PyTorch
# (default 26.09) with a PyTorch nightly venv and DeepEP/HybridEP, MoonEP, causal_conv1d,
# TileLang, CuTe DSL built against it. Needs apptainer with working --fakeroot (a host
# shell, not a user-namespaced one) and ~60 GB scratch space.
#
#   container/titan/build_titan.sh [OUTPUT_DIR]
#
# Then copy the .sif to marenostrum_new:/gpfs/projects/ehpc533/users/korbip/hybrid_v019/container/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HOME/.cache/containers}"
SCRATCH="${APPTAINER_SCRATCH:-$HOME/.cache}"
mkdir -p "$OUT" "$SCRATCH/apptainer_tmp" "$SCRATCH/apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/apptainer_tmp"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
export MAX_JOBS="${MAX_JOBS:-16}"
DEFINITION="${DEFINITION:-TitanTraining-main}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.09-py3}"
cd "$HERE/.."
./build_container.sh \
    --backend titan \
    --definition "$DEFINITION" \
    --base-image "$BASE_IMAGE" \
    --output "$OUT" \
    --container-cmd apptainer 2>&1 | tee "$OUT/build_${DEFINITION}_${BASE_IMAGE##*:}.log"
