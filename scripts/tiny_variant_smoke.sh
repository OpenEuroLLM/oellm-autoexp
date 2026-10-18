#!/bin/bash
# Tiny local smoke test for the linear-attention variants (GDN / mLSTM) on a
# workstation GPU with very little free VRAM (~1GB suffices).
#
# Usage: ./scripts/tiny_variant_smoke.sh [gated_delta_net|mlstm] [extra megatron args...]
#
# Requires fla-core + flash-linear-attention + mlstm_kernels importable; if not
# in the environment, install them locally with:
#   pip install --no-deps --target=.local_pydeps fla-core "flash-linear-attention==0.3.2" mlstm_kernels
#
# Environment notes (this workstation):
# - CC/CXX forced to system gcc: the conda-forge toolchain is missing libstdc++
#   headers (breaks both the dataset-helpers build and Triton's runtime build).
# - TRITON_LIBCUDA_PATH set because /sbin/ldconfig is unavailable in sandboxed
#   shells and Triton uses it to locate libcuda.
# - TORCHDYNAMO_DISABLE=1: Inductor also needs ldconfig; production configs
#   disable dynamo anyway.
# - Small NCCL buffers so the single-rank communicator fits next to a nearly
#   full GPU.
set -euo pipefail

VARIANT="${1:-gated_delta_net}"
shift || true

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/submodules/Megatron-LM"

if [ "$VARIANT" = "mlstm" ]; then
    VARIANT_ARGS=(--experimental-attention-variant mlstm
                  --linear-num-key-heads 4 --linear-num-value-heads 4
                  --linear-key-head-dim 16 --linear-value-head-dim 32
                  --mlstm-chunk-size 128)
else
    VARIANT_ARGS=(--experimental-attention-variant gated_delta_net
                  --linear-num-key-heads 2 --linear-num-value-heads 4
                  --linear-key-head-dim 32 --linear-value-head-dim 32)
fi

env CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    TORCHDYNAMO_DISABLE=1 \
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
    NCCL_MAX_NCHANNELS=2 NCCL_MIN_NCHANNELS=1 NCCL_BUFFSIZE=1048576 \
    PYTHONPATH="$REPO_ROOT/.local_pydeps:." \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node=1 --master_port=29631 pretrain_gpt.py \
        --num-layers 4 --hidden-size 128 --num-attention-heads 4 --kv-channels 32 \
        --seq-length 512 --max-position-embeddings 512 \
        --micro-batch-size 1 --global-batch-size 2 --train-iters 10 \
        --lr 1e-3 --min-lr 1e-4 --lr-decay-style constant --lr-warmup-iters 2 \
        --bf16 --attention-backend unfused --no-gradient-accumulation-fusion \
        --swiglu --normalization RMSNorm \
        --linear-attention-freq 2 \
        "${VARIANT_ARGS[@]}" \
        --mock-data --tokenizer-type NullTokenizer --vocab-size 4096 \
        --log-interval 1 --eval-iters 0 --eval-interval 100 \
        --distributed-backend nccl "$@"
