#!/bin/bash
# NCCL device API (GIN = GPU-initiated networking) in isolation on MareNostrum 5: does a kernel-
# initiated put / signal across nodes work, and does a device-API alltoall across 4 nodes (the
# ncclep EP 16 shape) complete? ncclep (TE NCCL-EP) hung in its first dispatch at EP 16 / 64 nodes
# (2026-10-29, ep64_nep16_e224): GPUs at 100 % / ~125 W, neither ncclep's 101 s timeout nor the
# 45 min NCCL watchdog fired.
#
# nccl-tests 2.21.1 (github.com/NVIDIA/nccl-tests afd59ab), built in the production image
# (NCCL 2.31.2, HPC-X Open MPI 5.0.10) into $R/nccl_dev_test/nccl-tests/build:
#   CPATH=/usr/local/cuda/include:/opt/hpcx/ompi/include make -j 32 MPI=1 MPI_HOME=/opt/hpcx/ompi \
#     NCCL_HOME=/usr NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
#
# Run inside an allocation of >= 4 nodes (one launcher; every test is its own srun step, PMIx):
#   scripts/korbi/hybrid_speed/nccl_gin_test.sh <out dir> [node list, default: the allocation's]
set -uo pipefail
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
SIF=$R/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif
B=$R/nccl_dev_test/nccl-tests/build
OUT=${1:?out dir}
NODES=${2:-$SLURM_JOB_NODELIST}
ONLY=${3:-}  # e.g. "t3 t4": run only these tests
mkdir -p "$OUT"
N4=$(scontrol show hostnames "$NODES" | head -4 | paste -sd,)
N2=$(scontrol show hostnames "$NODES" | head -2 | paste -sd,)

# the production NCCL environment (slurm/marenostrum_ehpc533.yaml), plus INIT / NET / GIN logging
export NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 NCCL_NET=IB NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3 NCCL_MIN_NCHANNELS=4 NCCL_NCHANNELS_PER_PEER=4
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export SINGULARITYENV_PREPEND_PATH=/opt/hpcx/ompi/bin

run() {  # name, nodes, tasks per node, timeout s, binary + args
    local name=$1 nodes=$2 tpn=$3 to=$4; shift 4
    if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" ${name%%_*} "* ]]; then return; fi
    local nn; nn=$(echo "$nodes" | tr , '\n' | wc -l)
    echo "=== $name: $nn node(s) x $tpn rank(s), timeout ${to}s: $*" | tee -a "$OUT/summary.txt"
    local t0=$SECONDS
    timeout --signal=KILL "$to" srun --overlap --mem=0 --mpi=pmix -w "$nodes" -N "$nn" \
        --ntasks-per-node="$tpn" --cpus-per-task=$((160 / tpn)) \
        singularity exec --nv --bind /gpfs "$SIF" "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    echo "    rc=$rc after $((SECONDS - t0)) s; last lines:" | tee -a "$OUT/summary.txt"
    grep -v "NCCL INFO" "$OUT/$name.log" | tail -6 | sed 's/^/      /' | tee -a "$OUT/summary.txt"
    grep -h "GIN\|gin\|GDAKI\|Proxy\|proxy" "$OUT/$name.log" | sort | uniq -c | sort -rn | head -6 \
        | sed 's/^/      [gin] /' | tee -a "$OUT/summary.txt"
}

# 0. host-initiated alltoall across 4 nodes: the network path itself
run t0_alltoall_host "$N4" 4 300 $B/alltoall_perf -b 1M -e 64M -f 4 -g 1 -n 20
# 1. GIN put + signal, 2 ranks on 2 nodes (one GPU each)
run t1_gin_putsignal "$N2" 1 300 $B/device_api/gin/throughput/ginPutBW_perf --gin_op put_signal -b 4 -e 4M
# 2. GIN ping-pong latency, 2 ranks on 2 nodes
run t2_gin_pingpong "$N2" 1 300 $B/device_api/gin/latency/ginPutLatency_pingPong_perf -b 4 -e 1M
# 3. device-API alltoall, GIN only, 4 nodes x 4 GPUs (the EP 16 shape)
run t3_alltoall_gin "$N4" 4 300 $B/alltoall_perf -b 1M -e 64M -f 4 -g 1 -n 20 -R 2 -D 3
# 4. device-API alltoall, NVLink (LSA) in-node + GIN across nodes, 4 nodes x 4 GPUs
run t4_alltoall_hybrid "$N4" 4 300 $B/alltoall_perf -b 1M -e 64M -f 4 -g 1 -n 20 -R 2 -D 4
echo "done: $OUT/summary.txt"
