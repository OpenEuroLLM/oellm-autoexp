#!/bin/bash
# Phase-1 readout of one small-scale testbed checkpoint (docs/recall_surgery_ambitious.md §14), one JUPITER node
# (4 GH200), inside the training image: HF export (oellm_ambitious, bf16) of <run>/ckpt/iter_<STEP>, then the rows on
# one GPU each, in parallel:
#   GPU 0  coqa 0-shot, winogrande 0-shot
#   GPU 1  squadv2 10-shot, first 4000 documents (as the 1.7B readouts; samples logged: abstention / F1|answered)
#   GPU 2  swde, fda, squad_completion (0-shot, <= 1000 documents each; config/eval_tasks/recall_evals.yaml)
#   GPU 3  RULER subset at 4096 tokens, 100 samples per task (small31_eval.RULER_TASKS), then the v8ml-valid loss
# (CoQA and Winogrande complete; the limits keep a readout at ~1 node-hour for 24 readouts in the budget)
# Offline compute nodes: datasets / nltk / RULER essays cached by small31_precache.sh (login node) in HF_HOME / NLTK_DATA.
# Run as the 1-node pool jobs of config/experiments/korbi/recall_surgery/small31_readout_jupiter.yaml.
#   ARM=b7 STEP=954 small31_readout.sh
set -uo pipefail
: "${ARM:?}" "${STEP:?}"
ROOT=${ROOT:-/e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/small31}
RUN=$ROOT/$ARM
EXPORT=$ROOT/exports/small31_${ARM}_$STEP
OUT=$ROOT/evals
LIMIT=${LIMIT:-}  # tests only: documents per task
mkdir -p "$ROOT/exports" "$OUT"
cd "$(dirname "$0")/../../.."  # the repository (python -m oellm_autoexp...)
echo "[readout] $ARM step $STEP -> $EXPORT ($(hostname), $(date -Is))"
if [ ! -f "$EXPORT/.export_done" ]; then
  EXPERT_DTYPE=bf16 HF_EXPORTER=sequential bash scripts/korbi/hf_export/export_run.sh "$RUN" "$STEP" "$EXPORT" \
    || { echo "[readout] export failed"; exit 1; }
fi
lim=(); [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")
E=scripts/korbi/recall_surgery/small31_eval.py
pids=()
CUDA_VISIBLE_DEVICES=0 python "$E" --export "$EXPORT" --out "$OUT" "${lim[@]}" --rows coqa:0 winogrande:0 \
  > "$OUT/$(basename "$EXPORT").gpu0.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python "$E" --export "$EXPORT" --out "$OUT" "${lim[@]}" --rows squadv2:10:4000 --batch-size 8 \
  > "$OUT/$(basename "$EXPORT").gpu1.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=2 python "$E" --export "$EXPORT" --out "$OUT" "${lim[@]}" --rows swde:0:1000 fda:0:1000 squad_completion:0:1000 \
  > "$OUT/$(basename "$EXPORT").gpu2.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=3 python "$E" --export "$EXPORT" --out "$OUT" "${lim[@]}" --rows ruler:4096:100 --batch-size 4 \
  --valid-loss > "$OUT/$(basename "$EXPORT").gpu3.log" 2>&1 & pids+=($!)
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[readout] $ARM step $STEP: rc $rc ($(date -Is)); logs $OUT/$(basename "$EXPORT").gpu*.log"
exit $rc
