#!/usr/bin/env bash
# Evaluation chain for one model (PLAN 6d):
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
#   EVAL_PARALLEL       1 runs M1 and the GSM8K modes at the same time, one per visible GPU, with
#                       one log each in $EVAL_DIR/logs (default 0: one after another)
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

# One task per result file: its name and its shell-quoted command.
NAMES=()
COMMANDS=()
add_task() {
  NAMES+=("$1")
  shift
  COMMANDS+=("$(printf '%q ' "$@")")
}
if [ "${EVAL_NUM_SEQS:-1000}" -gt 0 ]; then
  add_task m1 python -m oellm_autoexp.dllm.eval.eval_nelbo --model "$MODEL" \
    --data-prefix "$EVAL_DATA_PREFIX" --num-sequences "${EVAL_NUM_SEQS:-1000}" \
    --objective "$EVAL_OBJECTIVE" --out "$EVAL_DIR/$EVAL_RUN.m1.json"
fi
MODES="${EVAL_MODES:-static,dynamic}"
if [ "$MODES" != "none" ]; then
  IFS=',' read -r -a MODE_LIST <<< "$MODES"
  for mode in "${MODE_LIST[@]}"; do
    add_task "gsm8k_$mode" python -m oellm_autoexp.dllm.eval.eval_gsm8k --model "$MODEL" \
      --mode "$mode" --tau "${EVAL_TAU:-0.9}" --n "${EVAL_GSM8K_N:-250}" \
      --out "$EVAL_DIR/$EVAL_RUN.gsm8k_$mode.json"
  done
fi

if [ "${EVAL_PARALLEL:-0}" = "1" ] && [ "${#NAMES[@]}" -gt 1 ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t GPUS < <(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 )))
  fi
  mkdir -p "$EVAL_DIR/logs"
  PIDS=()
  for i in "${!NAMES[@]}"; do
    gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
    log="$EVAL_DIR/logs/$EVAL_RUN.${NAMES[$i]}.log"
    echo "[eval] ${NAMES[$i]} on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" bash -c "${COMMANDS[$i]}" > "$log" 2>&1 &
    PIDS+=($!)
  done
  failed=0
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "[eval] ${NAMES[$i]} done"
    else
      echo "[eval] ${NAMES[$i]} FAILED; last lines of its log:"
      tail -n 20 "$EVAL_DIR/logs/$EVAL_RUN.${NAMES[$i]}.log"
      failed=1
    fi
  done
  [ "$failed" = "0" ] || exit 1
else
  for i in "${!NAMES[@]}"; do
    echo "[eval] ${NAMES[$i]}"
    bash -c "${COMMANDS[$i]}"
  done
fi
echo "[eval] done"
