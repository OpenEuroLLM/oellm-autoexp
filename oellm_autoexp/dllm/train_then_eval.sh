#!/usr/bin/env bash
# Train, then evaluate the final checkpoint in the same allocation, so that no evaluation job waits
# in the queue (PLAN P7). Used as the prefix of backend.dist_cmd (backend.aux.dist_prefix):
#   train_then_eval.sh torchrun <torchrun flags> <launcher> <launcher flags>
# After the training command succeeds, node 0 runs eval_chain.sh when the latest checkpoint is the
# last iteration (--train-iters). A run that stopped early (--exit-duration-in-mins) is resumed by
# resubmitting the same job, and that job evaluates.
# The evaluation settings come from the training flags: --save (checkpoint), --bd-objective,
# --bd-block-size (with --bd-block-schedule, its last entry), --bd-token-shift, --bd-doc-mask (the
# attention rule of M1) and --mask-token.
# The job sets EVAL_RUN, EVAL_DIR, EVAL_HF_BASE, EVAL_DATA_PREFIX and EVAL_MODES (eval_chain.sh).
# An AR run (--bd-objective ar) is exported with B = 1 and shift 1, M1 is its exact NLL, and its
# GSM8K modes become "ar". EVAL_CHAIN_CMD replaces eval_chain.sh (tests).
set -euo pipefail
"$@"
if [ "${SLURM_NODEID:-0}" != "0" ]; then
  exit 0
fi

declare -A FLAG=()
prev=""
for arg in "$@"; do
  if [[ "$prev" == --* ]]; then
    FLAG["$prev"]="$arg"
  fi
  prev="$arg"
done

save="${FLAG[--save]:-}"
iters="${FLAG[--train-iters]:-}"
latest="none"
if [ -n "$save" ] && [ -f "$save/latest_checkpointed_iteration.txt" ]; then
  latest="$(cat "$save/latest_checkpointed_iteration.txt")"
fi
if [ -z "$iters" ] || [ -z "$save" ]; then
  echo "[train_then_eval] WARNING: no --train-iters or --save in the training command: no evaluation"
  exit 0
fi
if [ "$latest" != "$iters" ]; then
  echo "[train_then_eval] latest checkpoint $latest, train iters $iters: no evaluation" \
    "(resubmit the job to resume)"
  exit 0
fi

objective="${FLAG[--bd-objective]:-bd3}"
if [ "$objective" = "ar" ]; then
  block=1
  token_shift=1
  if [ "${EVAL_MODES:-static,dynamic}" != "none" ]; then
    export EVAL_MODES=ar
  fi
else
  block="${FLAG[--bd-block-size]:?--bd-block-size is missing}"
  # Same reading as the launcher: blank or "none" (any case) means no schedule.
  schedule="$(printf '%s' "${FLAG[--bd-block-schedule]:-none}" | tr -d '[:space:]')"
  if [ -n "$schedule" ] && [ "${schedule,,}" != "none" ]; then
    block="${schedule##*:}"
  fi
  token_shift="${FLAG[--bd-token-shift]:-0}"
fi
export EVAL_CKPT="$save" EVAL_OBJECTIVE="$objective" EVAL_BLOCK_SIZE="$block"
export EVAL_TOKEN_SHIFT="$token_shift" EVAL_MASK_ID="${FLAG[--mask-token]:?--mask-token is missing}"
export EVAL_DOC_MASK="${FLAG[--bd-doc-mask]:-1}"
echo "[train_then_eval] evaluating $save at iteration $latest (objective $objective, B $block," \
  "shift $token_shift, doc mask $EVAL_DOC_MASK)"
# shellcheck disable=SC2086  # EVAL_CHAIN_CMD may hold an interpreter and a script
exec ${EVAL_CHAIN_CMD:-./oellm_autoexp/dllm/eval/eval_chain.sh}
