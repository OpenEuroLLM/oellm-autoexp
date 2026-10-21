#!/usr/bin/env bash
# Evaluation chain for one model on one GPU (PLAN 6d):
#   1. export: MegaDLMs checkpoint -> Hugging Face directory (CPU; skipped when EVAL_CKPT is empty,
#      then EVAL_HF_BASE itself is evaluated),
#   2. M1: validation NELBO (bd3) or exact NLL (ar) on the held-out split (eval_nelbo.py),
#   3. GSM8K: one run per decoding mode (eval_gsm8k.py).
# Results are $EVAL_DIR/<run>.m1.json and $EVAL_DIR/<run>.gsm8k_<mode>.json, the names that
# collect_results.py reads. The chain is configured through environment variables, because the
# job wraps it in bash -c '...' (values must not contain spaces or single quotes):
#   EVAL_RUN            run name, the prefix of the result files
#   EVAL_DIR            result directory
#   EVAL_HF_BASE        HF base model id or directory (config.json and tokenizer files)
#   EVAL_CKPT           MegaDLMs checkpoint root to export, or empty
#   EVAL_ITERATION      checkpoint iteration to export (default: the tracker file)
#   EVAL_LAYOUT         layer spec of the checkpoint: te (default) or local
#   EVAL_BLOCK_SIZE, EVAL_TOKEN_SHIFT, EVAL_MASK_ID   bd3 settings of the run (export only)
#   EVAL_OBJECTIVE      M1 estimator: bd3 or ar
#   EVAL_DATA_PREFIX    Megatron .bin/.idx prefix of the held-out split
#   EVAL_NUM_SEQS       M1 windows (default 1000; 0 skips M1)
#   EVAL_MODES          comma-separated GSM8K modes: ar, static, dynamic (default static,dynamic;
#                       none skips GSM8K)
#   EVAL_GSM8K_N        number of GSM8K test problems (default 250)
#   EVAL_TAU            confidence threshold of the dynamic mode (default 0.9)
set -euo pipefail
: "${EVAL_RUN:?}" "${EVAL_DIR:?}" "${EVAL_HF_BASE:?}" "${EVAL_OBJECTIVE:?}" "${EVAL_DATA_PREFIX:?}"
mkdir -p "$EVAL_DIR"
MODEL="$EVAL_HF_BASE"

if [ -n "${EVAL_CKPT:-}" ]; then
  : "${EVAL_BLOCK_SIZE:?}" "${EVAL_TOKEN_SHIFT:?}" "${EVAL_MASK_ID:?}"
  MODEL="$EVAL_DIR/hf/$EVAL_RUN"
  echo "[eval] 1/3 export $EVAL_CKPT -> $MODEL"
  ITER_ARGS=()
  [ -n "${EVAL_ITERATION:-}" ] && ITER_ARGS=(--iteration "$EVAL_ITERATION")
  python -m oellm_autoexp.dllm.convert.megadlm_to_hf_qwen3 \
    --megatron "$EVAL_CKPT" --hf-base "$EVAL_HF_BASE" --out "$MODEL" \
    --block-size "$EVAL_BLOCK_SIZE" --token-shift "$EVAL_TOKEN_SHIFT" \
    --mask-token-id "$EVAL_MASK_ID" --layout "${EVAL_LAYOUT:-te}" "${ITER_ARGS[@]}"
else
  echo "[eval] 1/3 no checkpoint: evaluating $MODEL"
fi

if [ "${EVAL_NUM_SEQS:-1000}" -gt 0 ]; then
  echo "[eval] 2/3 M1 ($EVAL_OBJECTIVE) -> $EVAL_DIR/$EVAL_RUN.m1.json"
  python -m oellm_autoexp.dllm.eval.eval_nelbo --model "$MODEL" \
    --data-prefix "$EVAL_DATA_PREFIX" --num-sequences "${EVAL_NUM_SEQS:-1000}" \
    --objective "$EVAL_OBJECTIVE" --out "$EVAL_DIR/$EVAL_RUN.m1.json"
else
  echo "[eval] 2/3 M1 skipped"
fi

MODES="${EVAL_MODES:-static,dynamic}"
if [ "$MODES" != "none" ]; then
  IFS=',' read -r -a MODE_LIST <<< "$MODES"
  for mode in "${MODE_LIST[@]}"; do
    echo "[eval] 3/3 GSM8K $mode -> $EVAL_DIR/$EVAL_RUN.gsm8k_$mode.json"
    python -m oellm_autoexp.dllm.eval.eval_gsm8k --model "$MODEL" --mode "$mode" \
      --tau "${EVAL_TAU:-0.9}" --n "${EVAL_GSM8K_N:-250}" \
      --out "$EVAL_DIR/$EVAL_RUN.gsm8k_$mode.json"
  done
else
  echo "[eval] 3/3 GSM8K skipped"
fi
echo "[eval] done"
