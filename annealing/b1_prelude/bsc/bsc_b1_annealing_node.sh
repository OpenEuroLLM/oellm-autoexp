#!/bin/bash
# One Slurm task per ACC node. Each task starts torchrun inside the BSC image.
set -euo pipefail
mode=$1
run=$2
source_tree=$3
image=$4
workers=$5
cd "$source_tree"
module load singularity/4.1.5
job_tmp=$(mktemp -d "${TMPDIR:-/tmp}/oellm-b1-${SLURM_JOB_ID}.XXXXXXXX")
trap 'rm -rf -- "$job_tmp"' EXIT
mkdir "$job_tmp"/{tmp,triton,inductor}
printf 'write-probe\n' > "$job_tmp/tmp/write-probe"
test "$(df -Pk "$job_tmp" | awk 'END{print $4}')" -ge 16777216
test -n "${CUDA_VISIBLE_DEVICES:-}"
export MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / workers))
export NCCL_NET=IB NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3 NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=0 CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export SINGULARITYENV_PYTHONPATH="$source_tree:$source_tree/submodules/Megatron-LM"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="$job_tmp/inductor"
export SINGULARITYENV_TRITON_CACHE_DIR="$job_tmp/triton"
export SINGULARITYENV_TMPDIR=/tmp
export SINGULARITYENV_B1_GPUS_PER_NODE=4
singularity exec --nv --bind /gpfs --bind "$job_tmp/tmp:/tmp" --bind "$job_tmp:$job_tmp" "$image" \
  python -u -m torch.distributed.run --nnodes "$SLURM_JOB_NUM_NODES" --nproc-per-node "$workers" \
  --max-restarts 0 --node-rank "$SLURM_NODEID" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  "$run/bsc_b1_annealing_worker.py" "$mode" "$run" "$source_tree"
