#!/bin/bash
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --job-name=muon_qwen3_realdata
#SBATCH --output=/scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/06_real_data_%j.log

# ---------------------------------------------------------------------------
# Muon + Qwen3 0.1B_ne on REAL multilingual data (long-ctx-sample).
# ONE variable changed vs the green job 19941521: mock data -> real data + real
# 256k tokenizer. Everything else (arch, optimizer flags, container, env) is the
# same proven recipe.
#
# NO project_462000963 access needed: reads the READABLE long-ctx-sample copy in
# project_465002530. See REAL_DATA_RUNBOOK.md for the caveat -- long-ctx-sample
# oversamples long docs ~20x, so this is pipeline/plumbing validation and a
# controlled Muon-vs-Adam basis, NOT comparable to the real-mix baselines.
#
# Part A builds the datamix (pure-python, gates on success).
# Part B trains. First run also builds the blended dataset index in
# data_cache_path (one-time, cached) -- that may be the slow part for 155 files.
# Writes: the log, the datamix file, and the data cache (all in laingsam space).
# No checkpoints (no --save).
# ---------------------------------------------------------------------------

set -x
cd /scratch/project_465002530/users/laingsam/oellm-muon

SIF=/scratch/project_465002530/containers/MegatronTrainingLumi_x86_64.sif
OVL=/scratch/project_465002530/users/laingsam/oellm-muon/muon-overlay.img
BINDS="-B /boot/config-6.4.0-150600.23.73_15.0.14-cray_shasta_c -B /pfs -B /scratch -B /flash -B /opt/cray -B /var/spool/slurmd -B /appl"

DATAMIX=/scratch/project_465002530/users/laingsam/oellm-muon/muon_smoke/datamix4-longctx-lumi.txt
DATACACHE=/scratch/project_465002530/users/laingsam/cache/MEGATRON_256k/realdata_smoke_v2
HFCACHE=/scratch/project_465002530/cache/huggingface

mkdir -p "$DATACACHE"

echo "############ PART A: build the datamix (pure python, no GPU) ############"
srun --ntasks=1 singularity exec $BINDS --overlay $OVL $SIF bash -c '
  cd /scratch/project_465002530/users/laingsam/oellm-muon
  python3 -u muon_smoke/build_longctx_datamix.py --write
'
rc=$?
if [ $rc -ne 0 ] || [ ! -s "$DATAMIX" ]; then
  echo "############ PART A FAILED (rc=$rc) -- datamix not written. Stopping. ############"
  exit 1
fi
echo "datamix bytes: $(wc -c < "$DATAMIX")"

echo "############ PART B: Muon trains Qwen3 0.1B_ne on real multilingual data ############"
srun --ntasks=1 singularity exec $BINDS --overlay $OVL $SIF bash -c '
  cd /scratch/project_465002530/users/laingsam/oellm-muon
  export TRITON_HOME=/tmp/tri CC=gcc-12 CXX=g++-12
  export HF_HOME='"$HFCACHE"' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
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
    --data-args-path '"$DATAMIX"' \
    --data-cache-path '"$DATACACHE"' \
    --split 99,1,0 \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model openeurollm/tokenizer-256k \
    --trust-remote-code \
    --num-workers 4 --num-dataset-builder-threads 7 \
    --log-interval 1 --eval-iters 0 --eval-interval 1000 \
    --save-interval 100000 --distributed-timeout-minutes 30
'

echo "############ DONE ############"
echo "PASS = 'iteration N/20 ... lm loss' lines, loss finite, 0 nan iterations,"
echo "       and the loss should now DECREASE meaningfully (real data, not noise)."
