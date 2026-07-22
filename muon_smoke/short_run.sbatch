#!/bin/bash
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=00:10:00
#SBATCH --job-name=muon50m_mock
#SBATCH --output=/scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/03_mock_train_%j.log

# Short Muon smoke: 20 steps of the 50M model on MOCK data, 4 GPUs (DP=4).
# Hand-assembled (NOT dry-run-verified). If it dies on a flag it dies fast (~0 GPU).
# No real data, no tokenizer download (NullTokenizer), no checkpoints saved.

set -x
cd /scratch/project_465002530/users/laingsam/oellm-muon

SIF=/scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif
OVL=/scratch/project_465002530/users/laingsam/oellm-muon/muon-overlay.img
BINDS="-B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl"

srun --ntasks=1 singularity exec $BINDS --overlay $OVL $SIF bash -c '
  cd /scratch/project_465002530/users/laingsam/oellm-muon
  export TRITON_HOME=/tmp/tri HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CC=gcc-12 CXX=g++-12
  export PYTHONPATH=submodules/Megatron-LM
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    submodules/Megatron-LM/pretrain_gpt.py \
    --num-layers 12 --hidden-size 384 --ffn-hidden-size 1536 --num-attention-heads 6 \
    --seq-length 1024 --max-position-embeddings 1024 \
    --position-embedding-type rope --rotary-base 10000 --rotary-percent 1.0 \
    --normalization RMSNorm --norm-epsilon 1e-5 --qk-layernorm \
    --swiglu --disable-bias-linear --init-method-std 0.02 \
    --attention-dropout 0.0 --hidden-dropout 0.0 --bf16 --use-flash-attn \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --no-masked-softmax-fusion --no-bias-swiglu-fusion --no-bias-gelu-fusion \
    --no-bias-dropout-fusion --no-gradient-accumulation-fusion \
    --optimizer muon --muon-momentum 0.9 --muon-scalar-optimizer adam \
    --muon-scale-mode spectral --muon-num-ns-steps 5 --muon-coefficient-type quintic \
    --adam-beta1 0.9 --adam-beta2 0.99 --adam-eps 1e-8 \
    --clip-grad 1.0 --weight-decay 0.1 \
    --lr 1e-3 --min-lr 1e-4 --lr-decay-style constant --lr-warmup-iters 0 \
    --micro-batch-size 1 --global-batch-size 8 --train-iters 20 \
    --mock-data --tokenizer-type NullTokenizer --vocab-size 50304 \
    --num-workers 2 --log-interval 1 --eval-iters 0 --eval-interval 1000 --save-interval 100000 \
    --distributed-timeout-minutes 10
'
