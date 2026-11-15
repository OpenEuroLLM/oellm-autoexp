#!/bin/bash
# The recall suite (swde / fda / squad_completion, 0-shot; config/eval_tasks/recall_evals.yaml) for the in-house
# OELLM 32B dense (docs/recall_surgery_ambitious.md §18): the 32B team's FLAG runs never had these rows. One JUPITER
# node (4 GH200), inside the training image, HF backend (small31_eval.py) on the team's HF exports (Qwen3ForCausalLM,
# 64 GB bf16 -> one GPU each):
#   GPU 0  b1cons_90k     swde, fda          (1.5T tokens, equal-token reference of ambitious 24000)
#   GPU 1  b1cons_90k     squad_completion
#   GPU 2  b1consTW_300k  swde, fda          (5T tokens)
#   GPU 3  b1consTW_300k  squad_completion
# --max-length 4096 = the 32B's max_position_embeddings (lm-eval left-truncates longer contexts); batch 4 (64 GB weights).
# Offline compute nodes: datasets cached in HF_HOME by small31_precache.sh (login node), as the testbed readouts.
# Results in the MareNostrum launcher layout (<out>/<export>/results/<task>_0/...), so rescore_fda_squad.py reads them.
#   o32_recall_readout.sh          # all rows;  LIMIT=20 o32_recall_readout.sh for a test
set -uo pipefail
HF=${HF:-/e/scratch/e-sta-openeurollm/production_training/downstream_eval_32b/hf}
OUT=${OUT:-/e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/o32_recall}
LIMIT=${LIMIT:-}
mkdir -p "$OUT"
cd "$(dirname "$0")/../../.."
lim=(); [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")
E=scripts/korbi/recall_surgery/small31_eval.py
echo "[o32 recall] $(hostname) $(date -Is)"
pids=()
gpu=0
for ck in b1cons_90k b1consTW_300k; do
  for rows in "swde:0 fda:0" "squad_completion:0"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu python "$E" --export "$HF/$ck" --out "$OUT" --max-length 4096 --batch-size 4 "${lim[@]}" --rows $rows \
      > "$OUT/$ck.gpu$gpu.log" 2>&1 & pids+=($!)
    gpu=$((gpu + 1))
  done
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[o32 recall] rc $rc ($(date -Is)); logs $OUT/*.gpu*.log"
exit $rc
