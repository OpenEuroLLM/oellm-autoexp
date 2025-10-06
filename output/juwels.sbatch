#!/bin/bash
#SBATCH --account=debug
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=./output/%j.out
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --job-name=juwels

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
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i | grep "Address: " | tail -n1 | awk '{print $2}')
export MASTER_PORT=20074
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export ARCH=$(uname -m)
export SLURM_CPUS_PER_TASK=12
export SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
export PYTHONPATH=.:submodules/Megatron-LM
export PROJECT_DIR=.
export RUN_DIR=./submodules/Megatron-LM

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

echo "Launching juwels on JUWELS"
srun --exclusive --wait=60 --cpus-per-task=8 --threads-per-core=1 --gpu-bind=none bash -c 'export MACHINE_NAME='"$MACHINE_NAME"'; export NCCL_SOCKET_IFNAME='"$NCCL_SOCKET_IFNAME"'; export GLOO_SOCKET_IFNAME='"$GLOO_SOCKET_IFNAME"'; export NCCL_SOCKET_FAMILY='"$NCCL_SOCKET_FAMILY"'; export GLOO_SOCKET_FAMILY='"$GLOO_SOCKET_FAMILY"'; export TORCH_NCCL_ASYNC_ERROR_HANDLING='"$TORCH_NCCL_ASYNC_ERROR_HANDLING"'; export NCCL_IB_TIMEOUT='"$NCCL_IB_TIMEOUT"'; export NCCL_IB_DISABLE='"$NCCL_IB_DISABLE"'; export NCCL_IB_RETRY_CNT='"$NCCL_IB_RETRY_CNT"'; export NCCL_IB_HCA='"$NCCL_IB_HCA"'; export NCCL_NET_GDR_LEVEL='"$NCCL_NET_GDR_LEVEL"'; export NCCL_NET_GDR_READ='"$NCCL_NET_GDR_READ"'; export HF_ALLOW_CODE_EVAL='"$HF_ALLOW_CODE_EVAL"'; export MASTER_ADDR='"$MASTER_ADDR"'; export MASTER_PORT='"$MASTER_PORT"'; export NUM_NODES='"$NUM_NODES"'; export GPUS_PER_NODE='"$GPUS_PER_NODE"'; export NUM_GPUS_PER_NODE='"$NUM_GPUS_PER_NODE"'; export NUM_GPUS='"$NUM_GPUS"'; export ARCH='"$ARCH"'; export SLURM_CPUS_PER_TASK='"$SLURM_CPUS_PER_TASK"'; export SLURM_ARRAY_TASK_ID='"$SLURM_ARRAY_TASK_ID"'; export PYTHONPATH='"$PYTHONPATH"'; export PROJECT_DIR='"$PROJECT_DIR"'; export RUN_DIR='"$RUN_DIR"'; export LOCAL_ADDR=$(nslookup $(hostname | sed '"'"'s/.juwels//'"'"' )i | grep "Address: " | tail -n1 | awk '"'"'{print $2}'"'"') ; singularity exec --env MACHINE_NAME=$MACHINE_NAME --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME --env GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME --env NCCL_SOCKET_FAMILY=$NCCL_SOCKET_FAMILY --env GLOO_SOCKET_FAMILY=$GLOO_SOCKET_FAMILY --env TORCH_NCCL_ASYNC_ERROR_HANDLING=$TORCH_NCCL_ASYNC_ERROR_HANDLING --env NCCL_IB_TIMEOUT=$NCCL_IB_TIMEOUT --env NCCL_IB_DISABLE=$NCCL_IB_DISABLE --env NCCL_IB_RETRY_CNT=$NCCL_IB_RETRY_CNT --env NCCL_IB_HCA=$NCCL_IB_HCA --env NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL --env NCCL_NET_GDR_READ=$NCCL_NET_GDR_READ --env HF_ALLOW_CODE_EVAL=$HF_ALLOW_CODE_EVAL --env MASTER_ADDR=$MASTER_ADDR --env MASTER_PORT=$MASTER_PORT --env NUM_NODES=$NUM_NODES --env GPUS_PER_NODE=$GPUS_PER_NODE --env NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE --env NUM_GPUS=$NUM_GPUS --env ARCH=$ARCH --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK --env SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID --env PYTHONPATH=$PYTHONPATH --env PROJECT_DIR=$PROJECT_DIR --env RUN_DIR=$RUN_DIR  --nv ./containers/MegatronTraining_x86_64_202510011427.sif submodules/Megatron-LM/pretrain_gpt.py'
