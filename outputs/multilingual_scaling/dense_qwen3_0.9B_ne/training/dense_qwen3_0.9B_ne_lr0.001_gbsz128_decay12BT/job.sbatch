#!/bin/bash
#SBATCH --account=OELLM_prod2026
#SBATCH --nodes=4
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --output=outputs/multilingual_scaling/dense_qwen3_0.9B_ne/training/dense_qwen3_0.9B_ne_lr0.001_gbsz128_decay12BT/slurm-%j.log
#SBATCH --job-name=dense_qwen3_0.9B_ne_lr0.001_gbsz128_decay12BT
#SBATCH --ntasks=4

export HF_HOME=/leonardo_scratch/large/userexternal/donutu00/hf_home_cache
export WANDB_MODE=offline
export MACHINE_NAME=LEONARDO
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MASTER_ADDR=$(nslookup $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) | grep "Address: " | sed "s/Address: //" | tail) 
export MASTER_PORT=20074
export NUM_NODES=$SLURM_JOB_NUM_NODES
export GPUS_PER_NODE=4
export NUM_GPUS_PER_NODE=4
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export ARCH=$(uname -m)
export SLURM_CPUS_PER_TASK=32
export SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
export NCCL_IB_TIMEOUT=120
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64/stubs
export OMP_NUM_THREADS=1

echo "Launching job dense_qwen3_0.9B_ne_lr0.001_gbsz128_decay12BT"
srun bash -c 'export HF_HOME='$HF_HOME' ; \
 export WANDB_MODE='$WANDB_MODE' ; \
 export MACHINE_NAME='$MACHINE_NAME' ; \
 export HF_ALLOW_CODE_EVAL='$HF_ALLOW_CODE_EVAL' ; \
 export HF_DATASETS_OFFLINE='$HF_DATASETS_OFFLINE' ; \
 export TRANSFORMERS_OFFLINE='$TRANSFORMERS_OFFLINE' ; \
 export MASTER_ADDR='$MASTER_ADDR' ; \
 export MASTER_PORT='$MASTER_PORT' ; \
 export NUM_NODES='$NUM_NODES' ; \
 export GPUS_PER_NODE='$GPUS_PER_NODE' ; \
 export NUM_GPUS_PER_NODE='$NUM_GPUS_PER_NODE' ; \
 export NUM_GPUS='$NUM_GPUS' ; \
 export ARCH='$ARCH' ; \
 export SLURM_CPUS_PER_TASK='$SLURM_CPUS_PER_TASK' ; \
 export SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' ; \
 export NCCL_IB_TIMEOUT='$NCCL_IB_TIMEOUT' ; \
 export TRITON_LIBCUDA_PATH='$TRITON_LIBCUDA_PATH' ; \
 export OMP_NUM_THREADS='$OMP_NUM_THREADS' ; \
 export LOCAL_ADDR=$(nslookup $(hostname) | grep "Address: " | sed "s/Address: //" | tail) ; \
singularity exec --env HF_HOME=$HF_HOME \
 --env WANDB_MODE=$WANDB_MODE \
 --env MACHINE_NAME=$MACHINE_NAME \
 --env HF_ALLOW_CODE_EVAL=$HF_ALLOW_CODE_EVAL \
 --env HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE \
 --env TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE \
 --env MASTER_ADDR=$MASTER_ADDR \
 --env MASTER_PORT=$MASTER_PORT \
 --env NUM_NODES=$NUM_NODES \
 --env GPUS_PER_NODE=$GPUS_PER_NODE \
 --env NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE \
 --env NUM_GPUS=$NUM_GPUS \
 --env ARCH=$ARCH \
 --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK \
 --env SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID \
 --env NCCL_IB_TIMEOUT=$NCCL_IB_TIMEOUT \
 --env TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH \
 --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
 --env PYTHONPATH=.:submodules/Megatron-LM \
 --env RUN_DIR=./submodules/Megatron-LM \
 --env PROJECT_DIR=. \
 --nv --bind /leonardo_scratch \
 --bind /leonardo \
 --bind /leonardo_work/OELLM_prod2026/ \
 /leonardo_work/OELLM_prod2026/container_images/nemo_25.11.01.sif torchrun \
  --nproc-per-node 4\
  --nnodes 4\
  --max-restarts 0\
  --tee 3\
  --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT\
  --rdzv-backend static\
  --node-rank $SLURM_NODEID\
  --master-addr $MASTER_ADDR\
  --master-port $MASTER_PORT\
   ./submodules/Megatron-LM/pretrain_gpt.py \
   --adam-beta2 0.95\
   --disable-bias-linear\
   --attention-dropout 0.0\
   --bf16\
   --ckpt-step 18311\
   --data-cache-path /leonardo_scratch/large/userexternal/donutu00/MEGATRON_CACHEDIR\
   --data-path /leonardo_work/OELLM_prod2026/datasets/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all\
   --dataloader-type single\
   --eval-interval 1600\
   --ffn-hidden-size 4608\
   --global-batch-size 128\
   --group-query-attention\
   --hidden-dropout 0.0\
   --hidden-size 1536\
   --kv-channels 128\
   --legacy-tokenizer\
   --load outputs/multilingual_scaling/dense_qwen3_0.9B_ne/training/dense_qwen3_0.9B_ne_lr0.001_gbsz128_stable300BT/checkpoints\
   --log-interval 10\
   --log-progress\
   --log-throughput\
   --log-timers-to-tensorboard\
   --lr 0.001\
   --lr-decay-style WSD\
   --lr-warmup-iters 2000\
   --lr-wsd-decay-iters 4577\
   --lr-wsd-decay-style linear\
   --max-position-embeddings 4096\
   --merge-file /leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/merges.txt\
   --micro-batch-size 8\
   --moe-aux-loss-coeff 0.0\
   --moe-router-load-balancing-type aux_loss\
   --normalization RMSNorm\
   --num-attention-heads 16\
   --num-layers 28\
   --num-query-groups 8\
   --num-workers 8\
   --override-opt_param-scheduler\
   --position-embedding-type rope\
   --profile-ranks 0\
   --profile-step-start 5\
   --qk-layernorm\
   --save outputs/multilingual_scaling/dense_qwen3_0.9B_ne/training/dense_qwen3_0.9B_ne_lr0.001_gbsz128_decay12BT/checkpoints\
   --save-extra-steps 999999999 999999999 999999999 999999999 999999999 999999999 999999999 999999999 999999999\
   --save-interval 4000\
   --seq-length 4096\
   --sequence-parallel\
   --split 99,1,0\
   --swiglu\
   --tensorboard-dir outputs/multilingual_scaling/dense_qwen3_0.9B_ne/training/dense_qwen3_0.9B_ne_lr0.001_gbsz128_decay12BT/tensorboard\
   --tensorboard-queue-size 5\
   --train-iters 22889\
   --use-distributed-optimizer\
   --use-flash-attn\
   --vocab-file /leonardo_work/OELLM_prod2026/models/EleutherAI/gpt-neox-20b/vocab.json\
   --wandb-entity openeurollm-project\
   --wandb-exp-name dense_qwen3_0.9B_ne_lr0.001_gbsz128_decay12BT\
   --wandb-project dense_multilingual_scaling\
   --wandb-save-dir outputs/multilingual_scaling/dense_qwen3_0.9B_ne/training/dense_qwen3_0.9B_ne_lr0.001_gbsz128_decay12BT\
   --weight-decay 0.1'
