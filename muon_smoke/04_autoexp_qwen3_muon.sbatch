#!/bin/bash
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --job-name=muon_qwen3_autoexp
#SBATCH --output=/scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/04_autoexp_qwen3_%j.log

# ---------------------------------------------------------------------------
# Does Muon train the REAL Qwen3 0.1B_ne architecture? MOCK data, no dataset.
#
# ONE variable vs the proven job 19928413: the model architecture.
# Same container, same overlay, same optimizer flags as that green run.
#
# Architecture mirrors config/backend/megatron/multilingual_scaling/models/
#   qwen3_dense_0.1B_ne.yaml (16L/512h/1536ffn/8heads/kv128/qk-norm).
# Vocab 256000 to match the real openeurollm 256k tokenizer's embedding size.
#
# Writes ONLY: the log below, and /tmp scratch inside the container.
# No checkpoints (no --save passed). No network (HF_HUB_OFFLINE=1, mock data).
# Part 1 is a cheap import sanity check that exits before any GPU work.
# ---------------------------------------------------------------------------

set -x
cd /scratch/project_465002530/users/laingsam/oellm-muon

# NOTE 2026-07-16: deliberately using MegatronTrainingLumi, NOT AutoExp_2026-07-02.
#   (a) it's the container YOUR branch's config/container/lumi.yaml points at, so
#       it's the one that matters for your pipeline;
#   (b) it's already proven to work with the overlay (job 19928413), so this run
#       changes exactly ONE variable: the model architecture.
# Testing the AutoExp container is a separate question -- and the PYTHONPATH
# approach in REAL_DATA_RUNBOOK.md makes it mostly moot.
SIF=/scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif
OVL=/scratch/project_465002530/users/laingsam/oellm-muon/muon-overlay.img
BINDS="-B /boot/config-6.4.0-150600.23.73_15.0.14-cray_shasta_c -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl"

echo "############ PART 1: sanity check -- emerging_optimizers importable? ############"
srun --ntasks=1 singularity exec $BINDS --overlay $OVL $SIF bash -c '
  export PYTHONPATH=submodules/Megatron-LM
  python -u -c "
import sys, importlib.util
print(\"python:\", sys.version.split()[0])
spec = importlib.util.find_spec(\"emerging_optimizers\")
print(\"emerging_optimizers found:\", spec is not None)
if spec is None:
    print(\">>> OVERLAY DOES NOT PORT to AutoExp container. Needs rebuild on this .sif.\")
    sys.exit(42)
import emerging_optimizers
print(\"emerging_optimizers:\", getattr(emerging_optimizers, \"__version__\", \"?\"), \"at\", emerging_optimizers.__file__)
print(\">>> OVERLAY PORTS OK\")
"
'
rc=$?
if [ $rc -ne 0 ]; then
  echo "############ PART 1 FAILED (rc=$rc) -- stopping before GPU work. ############"
  echo "emerging_optimizers not importable. No GPU hours wasted."
  exit $rc
fi

echo "############ PART 2: Muon trains the real Qwen3 0.1B_ne arch (mock data) ############"
srun --ntasks=1 singularity exec $BINDS --overlay $OVL $SIF bash -c '
  cd /scratch/project_465002530/users/laingsam/oellm-muon
  export TRITON_HOME=/tmp/tri HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CC=gcc-12 CXX=g++-12
  # Deliberately NOT setting AITER_JIT_DIR: ~/.aiter is already warm from job
  # 19928413. Pointing at a cold dir makes all 4 ranks race to JIT-compile into
  # one shared cache -> the "waiting for baton release" hang that burned an
  # allocation earlier. The real multilingual runs use a PER-RANK dir; with
  # torchrun a single shared dir is the worst of both. Keep the warm cache.
  export PYTHONPATH=submodules/Megatron-LM
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    submodules/Megatron-LM/pretrain_gpt.py \
    --num-layers 16 --hidden-size 512 --ffn-hidden-size 1536 \
    --num-attention-heads 8 --kv-channels 128 \
    --seq-length 4096 --max-position-embeddings 4096 \
    --position-embedding-type rope --rotary-base 10000 --rotary-percent 1.0 \
    --normalization RMSNorm --norm-epsilon 1e-5 --qk-layernorm \
    --swiglu --disable-bias-linear --init-method-std 0.02 \
    --attention-dropout 0.0 --hidden-dropout 0.0 --bf16 --use-flash-attn \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --ckpt-format torch_dist \
    --no-masked-softmax-fusion --no-bias-swiglu-fusion --no-bias-gelu-fusion \
    --no-bias-dropout-fusion --no-gradient-accumulation-fusion \
    --optimizer muon --muon-momentum 0.9 --muon-scalar-optimizer adam \
    --muon-scale-mode spectral --muon-num-ns-steps 5 --muon-coefficient-type quintic \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --clip-grad 1.0 --weight-decay 0.1 \
    --lr 5e-4 --min-lr 0.0 --lr-decay-style constant --lr-warmup-iters 0 \
    --micro-batch-size 1 --global-batch-size 8 --train-iters 20 \
    --mock-data --tokenizer-type NullTokenizer --vocab-size 256000 \
    --num-workers 2 --log-interval 1 --eval-iters 0 --eval-interval 1000 \
    --save-interval 100000 --distributed-timeout-minutes 20
'

echo "############ DONE ############"
echo "PASS = 'OVERLAY PORTS OK' in Part 1, then 'iteration N/20 ... lm loss' lines"
echo "       and 'use_layer_wise_distributed_optimizer .... True' in Part 2."
