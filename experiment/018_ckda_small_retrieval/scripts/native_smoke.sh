#!/usr/bin/env bash
# Prepared compatibility check; not a production adaptation recipe.
# Run inside the training image after reviewing the focused baseline.
set -euo pipefail
arm=${1:?ckda or kda}
case "$arm" in
  ckda) export SKDA_ARM=skda-shipped-hybrid-lowrank ;;
  kda) export SKDA_ARM=kda-sig-hybrid-lowrank ;;
  *) exit 2 ;;
esac
export CKDA_WORK=${CKDA_WORK:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018}
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
export PYTHONPATH="$CKDA_WORK/source/native_megatron:$CKDA_WORK/source:$CKDA_WORK/source/lm_scaling:$R/pylibs_ep"
export CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export HF_HOME=/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home
export HF_HUB_OFFLINE=1
data=/gpfs/projects/ehpc533/preprocessed/gpt-neox-20b/nemotron-cc/1.0
python -m torch.distributed.run --standalone --nproc-per-node 4 \
  "$CKDA_WORK/source/native_megatron/pretrain_gpt.py" \
  --spec lm_scaling.megatron_ext.specs arm_spec \
  --load "$CKDA_WORK/assets/native/$arm" --ckpt-format torch \
  --finetune --no-load-optim --no-load-rng \
  --num-layers 24 --hidden-size 1536 --ffn-hidden-size 5952 \
  --num-attention-heads 24 --kv-channels 64 \
  --seq-length 4096 --max-position-embeddings 4096 \
  --normalization RMSNorm --norm-epsilon 1e-5 --swiglu --disable-bias-linear \
  --position-embedding-type rope --rotary-base 10000 \
  --bf16 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --micro-batch-size 1 --global-batch-size 4 --train-iters 2 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$CKDA_WORK/assets/tokenizer" \
  --train-data-path "$data/Nemotron-cc-2024-HQ-LUMI-sample-318BT/high-all" \
  --valid-data-path "$data/Nemotron-cc-2024-HQ-LUMI-sample-valid/high-all" \
  --data-cache-path "$CKDA_WORK/cache/native_data" \
  --lr 1e-5 --min-lr 1e-5 --lr-decay-style constant --lr-warmup-iters 0 \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --weight-decay 0.1 \
  --attention-dropout 0 --hidden-dropout 0 --clip-grad 1 \
  --no-gradient-accumulation-fusion --no-masked-softmax-fusion \
  --eval-iters 2 --eval-interval 100000 --log-interval 1 --seed 2026
