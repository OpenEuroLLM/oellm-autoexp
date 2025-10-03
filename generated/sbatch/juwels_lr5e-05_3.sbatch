#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --partition=dc-gpu
#SBATCH --account=project-default
#SBATCH --gres=gpu:4
#SBATCH --job-name=juwels_lr5e-05_3
#SBATCH --output=outputs/juwels/juwels_lr5e-05_3/slurm.out
#SBATCH --constraint=volta

export MACHINE_NAME=JUWELS
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=18
export NCCL_IB_DISABLE=0
export NCCL_IB_RETRY_CNT=20
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_NET_GDR_LEVEL=2
export NCCL_NET_GDR_READ=1
export HF_ALLOW_CODE_EVAL=1
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i | grep "Address: " | tail -n1 | awk "{print $2}")
export MASTER_PORT=20074
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export ARCH=$(uname -m)
export SLURM_CPUS_PER_TASK=12
export CUDA_DEVICE_MAX_CONNECTIONS=1

ml Stages/2025
ml GCC/13.3.0
ml Python/3.12.3
ml CUDA/12
ml cuDNN/9.5.0.50-CUDA-12
ml NCCL/default-CUDA-12
ml OpenMPI/5.0.5
ml ParaStationMPI/5.11.0-1
ml ParaStationMPI/5.11.0-1-mt
ml ScaLAPACK/2.2.0-fb

echo "Launching juwels_lr5e-05_3 on JUWELS"
srun scripts/run_megatron.sh --lr 5e-05
