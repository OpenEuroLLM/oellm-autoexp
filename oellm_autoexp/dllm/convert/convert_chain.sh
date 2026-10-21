#!/usr/bin/env bash
# Conversion chain for one Qwen3 size and tensor-parallel size (dllm-bd3-spec section 13):
#   1. template: 1-iteration pretrain_block_difflm.py run with mock data and lr 0 that saves a
#      torch checkpoint with the exact key set, shapes, TP shards and padded vocabulary,
#   2. convert: fill that template from the Hugging Face weights (CPU),
#   3. parity: AR-mode logits against Hugging Face, plus invariant I3 (GPU).
# The arguments are the Megatron, MegaDLMs and --bd-* flags of the template run (the experiment
# renders them). The chain is configured through environment variables, so that every "--" flag
# after this script is a launcher flag:
#   CHAIN_HF            HF model id or directory (for example Qwen/Qwen3-0.6B-Base)
#   CHAIN_TEMPLATE      template checkpoint directory (must match --save of the arguments)
#   CHAIN_OUT           converted checkpoint directory
#   CHAIN_DATA_PREFIX   Megatron .bin/.idx prefix of the held-out split (parity data)
#   CHAIN_NPROC         processes per node = tensor-parallel size
#   CHAIN_PARITY_OUT    parity JSON path (default: $CHAIN_OUT/parity.json)
set -euo pipefail
: "${CHAIN_HF:?}" "${CHAIN_TEMPLATE:?}" "${CHAIN_OUT:?}" "${CHAIN_DATA_PREFIX:?}" "${CHAIN_NPROC:?}"
PARITY_OUT="${CHAIN_PARITY_OUT:-$CHAIN_OUT/parity.json}"
RUN=(torchrun --standalone --nnodes 1 --nproc-per-node "$CHAIN_NPROC")

echo "[chain] 1/3 template -> $CHAIN_TEMPLATE"
"${RUN[@]}" oellm_autoexp/dllm/pretrain_block_difflm.py "$@"
test -f "$CHAIN_TEMPLATE/latest_checkpointed_iteration.txt"

echo "[chain] 2/3 convert $CHAIN_HF -> $CHAIN_OUT"
python -m oellm_autoexp.dllm.convert.hf_qwen3_to_megadlm \
  --hf "$CHAIN_HF" --template "$CHAIN_TEMPLATE" --out "$CHAIN_OUT"

echo "[chain] 3/3 parity -> $PARITY_OUT"
"${RUN[@]}" oellm_autoexp/dllm/eval/parity_check.py "$@" \
  --load "$CHAIN_OUT" --finetune --no-load-optim --no-load-rng \
  --parity-hf "$CHAIN_HF" --parity-data-prefix "$CHAIN_DATA_PREFIX" --parity-out "$PARITY_OUT"
echo "[chain] done"
