#!/bin/bash
# Build deep_ep with NUM_MAX_NVL_PEERS=4 for JUPITER's 4-GPU-per-node topology.
#
# Root cause: The container's deep_ep was compiled with NUM_MAX_NVL_PEERS=8
# (designed for 8-GPU DGX servers). With EP=8 on 2-node JUPITER:
#   - num_nvl_ranks = min(8, 8) = 8  → tries CUDA IPC for cross-node ranks → CRASH
#   - num_rdma_ranks = 1              → RDMA/NVSHMEM disabled entirely
#
# With NUM_MAX_NVL_PEERS=4:
#   - num_nvl_ranks = min(8, 4) = 4  → CUDA IPC only within local 4-GPU node ✓
#   - num_rdma_ranks = 2             → NVSHMEM/RDMA active between nodes ✓
#
# Usage (on JUPITER login node):
#   bash scripts/build_deepep_jupiter4gpu.sh

set -euo pipefail

CONTAINER=/e/scratch/projectnucleus/poeppel1/container_cache/MegatronTraining-JUPITER-deepep-nvshmem_base2510_aarch64_202603310850.sif
SRC_DIR=/e/scratch/projectnucleus/poeppel1/deep_ep_jupiter4gpu_src
INSTALL_DIR=/e/scratch/projectnucleus/poeppel1/deep_ep_jupiter4gpu

echo "=== Step 1: Copy deep_ep source from container ==="
rm -rf "$SRC_DIR"
apptainer exec "$CONTAINER" cp -r /opt/deepep/ "$SRC_DIR"/
echo "Source copied to $SRC_DIR"

echo ""
echo "=== Step 2: Patch NUM_MAX_NVL_PEERS 8 → 4 ==="
grep "NUM_MAX_NVL_PEERS" "$SRC_DIR/csrc/kernels/configs.cuh"
sed -i 's/#define NUM_MAX_NVL_PEERS 8/#define NUM_MAX_NVL_PEERS 4/' "$SRC_DIR/csrc/kernels/configs.cuh"
echo "After patch:"
grep "NUM_MAX_NVL_PEERS" "$SRC_DIR/csrc/kernels/configs.cuh"

echo ""
echo "=== Step 2b: Patch internode.cu for NUM_MAX_NVL_PEERS=4 ==="
# internode.cu has 3 places hardcoded for 8-peer topology:
#
# 1. Static assert: EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, ...) → 4
sed -i 's/EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8,/EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 4,/g' \
    "$SRC_DIR/csrc/kernels/internode.cu" \
    "$SRC_DIR/csrc/deep_ep.hpp"
#
# 2. The `is_token_in_rank` array uses packed bool reads:
#    With 8 peers: 8 bools = 8 bytes → read as uint64_t in one load
#    With 4 peers: 4 bools = 4 bytes → read as uint32_t in one load
#    Changes: sizeof(uint64_t) → sizeof(uint32_t), uint64_t var → uint32_t var,
#             cast to uint64_t* → uint32_t*, and var name uint64 → uint32.
sed -i 's/EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS \* sizeof(bool) == sizeof(uint64_t),/EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint32_t),/' \
    "$SRC_DIR/csrc/kernels/internode.cu"
sed -i 's/\*reinterpret_cast<const uint64_t\*>(is_token_in_rank/*reinterpret_cast<const uint32_t*>(is_token_in_rank/g' \
    "$SRC_DIR/csrc/kernels/internode.cu"
sed -i 's/__ldg(reinterpret_cast<const uint64_t\*>(is_token_in_rank/__ldg(reinterpret_cast<const uint32_t*>(is_token_in_rank/g' \
    "$SRC_DIR/csrc/kernels/internode.cu"
sed -i 's/uint64_t is_token_in_rank_uint64/uint32_t is_token_in_rank_uint32/g' \
    "$SRC_DIR/csrc/kernels/internode.cu"
sed -i 's/is_token_in_rank_uint64/is_token_in_rank_uint32/g' \
    "$SRC_DIR/csrc/kernels/internode.cu"
sed -i 's/reinterpret_cast<const bool\*>(&is_token_in_rank_uint32)/reinterpret_cast<const bool*>(\&is_token_in_rank_uint32)/g' \
    "$SRC_DIR/csrc/kernels/internode.cu"

echo "Verifying patches applied (should show 4, uint32_t):"
grep -n "NVL_PEERS == 4\|NVL_PEERS \* sizeof\|is_token_in_rank_uint32\|const uint32_t\*.*is_token_in_rank" \
    "$SRC_DIR/csrc/kernels/internode.cu" | head -10

echo ""
echo "=== Step 2c: Patch buffer.py combine config for NUM_MAX_NVL_PEERS=4 (2 RDMA ranks) ==="
# get_combine_config(8) was tuned for NUM_MAX_NVL_PEERS=8 (1 RDMA rank, no actual RDMA used).
# With NUM_MAX_NVL_PEERS=4 and EP=8: num_rdma_ranks=2, num_warps_per_forwarder=max(24/2,1)=12.
# Assertions that must hold (internode.cu):
#   num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder  → need >= 12
#   num_max_nvl_chunked_recv_tokens / num_rdma_ranks > max(rdma_send, nvl_send) → nvl_recv/2 > max(16,128) → nvl_recv > 256
# Config args: (num_sms, num_sms_for_api, nvl_recv, rdma_send, nvl_send)
# Old (broken):  8: Config(Buffer.num_sms, 4, 256, 6, 128)
# New (fixed):   8: Config(Buffer.num_sms, 4, 288, 16, 128)
sed -i 's/8: Config(Buffer\.num_sms, 4, 256, 6, 128)/8: Config(Buffer.num_sms, 4, 288, 16, 128)/' \
    "$SRC_DIR/deep_ep/buffer.py"
echo "After patch:"
grep -n "8: Config" "$SRC_DIR/deep_ep/buffer.py"

echo ""
echo "=== Step 3: Build inside container (ensures ABI compatibility) ==="
mkdir -p "$INSTALL_DIR"
# Clean any prior build artifacts
rm -rf "$SRC_DIR/build" "$SRC_DIR/deep_ep.egg-info"

# Use --env to force container's Python/pip (not inherited venv/miniconda from host PATH)
# CUDA_HOME and NVSHMEM_DIR must be set explicitly — not inherited from host.
# TORCH_CUDA_ARCH_LIST=9.0: JUPITER is GH200 (sm_90); the default arch list includes sm_80
# which fails because deep_ep kernels use sm_90-only features (cp.async.bulk, elect, etc.)
# NVSHMEM_DIR: use 3.4.5 from venv (same version as container's runtime libnvshmem_host.so.3).
# Using /opt/nvshmem (3.5.21) creates device/host version mismatch at runtime because
# the container's /usr/local/cuda/lib64/libnvshmem_host.so.3 is 3.4.5 (symlink → 3.4.5).
#
# Note: the venv nvshmem has libnvshmem_host.so.3 but NOT libnvshmem_host.so (no unversioned
# symlink). deep_ep setup.py hardcodes 'libnvshmem_host.so' when NVSHMEM_DIR is explicitly set
# (it only calls get_nvshmem_host_lib_name() for auto-detected pip installs). Fix: copy to a
# staging dir and add the missing symlink.
NVSHMEM_SRC=/e/project1/projectnucleus/poeppel1/venv/lib/python3.12/site-packages/nvidia/nvshmem
NVSHMEM_DIR=/e/scratch/projectnucleus/poeppel1/nvshmem_344_staging
rm -rf "$NVSHMEM_DIR"
cp -r "$NVSHMEM_SRC" "$NVSHMEM_DIR"
# Add missing unversioned symlink that setup.py expects
ln -sf libnvshmem_host.so.3 "$NVSHMEM_DIR/lib/libnvshmem_host.so"
echo "NVSHMEM staging dir with symlink fix: $NVSHMEM_DIR/lib/libnvshmem_host.so → $(readlink $NVSHMEM_DIR/lib/libnvshmem_host.so)"

apptainer exec --nv \
    --env "PATH=/usr/local/bin:/usr/bin:/bin" \
    --env "CUDA_HOME=/usr/local/cuda" \
    --env "NVSHMEM_DIR=$NVSHMEM_DIR" \
    --env "TORCH_CUDA_ARCH_LIST=9.0" \
    --env "CPATH=/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl" \
    "$CONTAINER" \
    bash -c "cd '$SRC_DIR' && python3 -m pip install . --target '$INSTALL_DIR' --upgrade --no-build-isolation -v 2>&1"

SO="$INSTALL_DIR/deep_ep_cpp.cpython-312-aarch64-linux-gnu.so"
echo ""
echo "=== Step 4: Convert DT_RUNPATH → DT_RPATH (so NVSHMEM path wins over LD_LIBRARY_PATH) ==="
# Without this, the container's LD_LIBRARY_PATH=/opt/nvshmem/lib:... overrides DT_RUNPATH.
# DT_RPATH is searched BEFORE LD_LIBRARY_PATH, guaranteeing the right NVSHMEM version loads.
apptainer exec "$CONTAINER" patchelf --force-rpath --set-rpath "$NVSHMEM_DIR/lib" "$SO"
echo "RPATH after patchelf:"
apptainer exec "$CONTAINER" readelf -d "$SO" | grep -E 'RPATH|RUNPATH'

echo ""
echo "=== Build complete ==="
echo "Installed to: $INSTALL_DIR"
echo ""
echo "Verify: check for deep_ep_cpp.cpython-312-aarch64-linux-gnu.so"
ls "$INSTALL_DIR"/deep_ep_cpp*.so 2>/dev/null || ls "$INSTALL_DIR"/deep_ep_cpp* 2>/dev/null || echo "WARNING: .so not found, check build output above"
echo ""
echo "To use in experiments, add to backend.env.PYTHONPATH:"
echo "  PYTHONPATH: \"$INSTALL_DIR:.:submodules/Megatron-LM:/opt/\""
