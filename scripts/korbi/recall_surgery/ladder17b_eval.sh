#!/bin/bash
# Native 1.7B ladder checkpoint -> HF export -> CoQA (0-shot, all 500 docs) / Winogrande (0-shot, all 1267) /
# SQuAD v2 (10-shot, first SQUAD_LIMIT docs, two shards) on one node, one task (shard) per GPU, through the
# original codebase's lm-eval wrapper (scripts/korbi/recall_surgery/ladder17b_evaluate.py). The recall-surgery
# testbed's readout (docs/recall_surgery_ambitious.md, section 8); launched by
# config/experiments/korbi/recall_surgery/ladder17b_eval_mn5.yaml.
#
# Environment: SKDA_ARM, CKPT (a run dir with latest_checkpointed_iteration.txt, an iter_ dir, or the .pt),
#   EXPORT_DIR, OUT_DIR, CODE_DIR / BASE_CODE_DIR (as ladder17b_train.sh; SKDA_INTERVENTION passes through),
#   SQUAD_LIMIT (default 4000; 0 = all 11873), BATCH (default 4)
set -uo pipefail
: "${SKDA_ARM:?}" "${CKPT:?}" "${EXPORT_DIR:?}" "${OUT_DIR:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BASE_CODE_DIR=${BASE_CODE_DIR:-/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018/source}
CODE_DIR=${CODE_DIR:-$BASE_CODE_DIR}
SQUAD_LIMIT=${SQUAD_LIMIT:-4000}; BATCH=${BATCH:-4}
HYBRID_ENV=${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
W018=/gpfs/projects/ehpc533/users/korbip/ckda_retrieval_018
TOKENIZER=${TOKENIZER:-$W018/assets/tokenizer}
fla=$BASE_CODE_DIR; [ -d "$CODE_DIR/fla" ] && fla=$CODE_DIR
# notitan first: the image ships torchtitan, and titan_ext would register train specs (titan_oellm, toml, ...) on
# import; the exporter only needs titan_ext.hybrid_layers
export PYTHONPATH="$ROOT/scripts/korbi/recall_surgery/notitan:$CODE_DIR:$CODE_DIR/lm_scaling:$fla:$ROOT/scripts/korbi/recall_surgery/compat:$HYBRID_ENV/pylibs_flag:$HYBRID_ENV/pylibs_hfx:$HYBRID_ENV/pylibs_ep"
export SKDA_ARM SKDA_INTERVENTION="${SKDA_INTERVENTION:-}"
# 018's offline evaluation environment (probe mn5-pool-p264.0665: all six datasets load offline)
export HF_HOME=/gpfs/projects/ehpc533/users/korbip/data/flag_evals/hf_home
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$W018/cache/datasets" HF_METRICS_CACHE="$W018/cache/metrics"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 NLTK_DATA="$HYBRID_ENV/nltk_data"
export TRITON_CACHE_DIR="/tmp/recall17b_eval_triton_${SLURM_JOB_ID:-x}_${SLURM_STEP_ID:-0}"
mkdir -p "$OUT_DIR"

# the checkpoint: a run dir -> its latest iter_ -> mp_rank_00/model_optim_rng.pt; CKPT=none: EXPORT_DIR is an
# existing export (config.json + model.safetensors), evaluated as is
if [ "$CKPT" = none ]; then
  [ -f "$EXPORT_DIR/config.json" ] && [ -e "$EXPORT_DIR/model.safetensors" ] || { echo "[ladder17b_eval] FAIL: no export in $EXPORT_DIR"; exit 1; }
fi
pt=$CKPT
if [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
  pt=$CKPT/iter_$(printf %07d "$(cat "$CKPT/latest_checkpointed_iteration.txt")")
fi
[ -d "$pt" ] && pt=$pt/mp_rank_00/model_optim_rng.pt
echo "[ladder17b_eval] arm=$SKDA_ARM intervention=${SKDA_INTERVENTION:-none} ckpt=$pt export=$EXPORT_DIR"
if [ "$CKPT" != none ] && [ ! -f "$EXPORT_DIR/.native_export_done" ]; then
  python "$CODE_DIR/lm_scaling/megatron_ext/export_megatron.py" --run "$pt" --arm "$SKDA_ARM" --geometry 1.7B \
    --out "$EXPORT_DIR" --tokenizer EleutherAI/gpt-neox-20b || { echo "[ladder17b_eval] export failed"; exit 1; }
  echo "$pt" > "$EXPORT_DIR/.native_export_done"
fi

E="python $ROOT/scripts/korbi/recall_surgery/ladder17b_evaluate.py --source $CODE_DIR --model $EXPORT_DIR --tokenizer $TOKENIZER --out $OUT_DIR --batch-size $BATCH"
CUDA_VISIBLE_DEVICES=0 $E --task winogrande --shots 0 > "$OUT_DIR/winogrande.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $E --task coqa --shots 0 > "$OUT_DIR/coqa.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $E --task squadv2 --shots 10 --limit "$SQUAD_LIMIT" --shard 0/2 > "$OUT_DIR/squadv2.0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 $E --task squadv2 --shots 10 --limit "$SQUAD_LIMIT" --shard 1/2 > "$OUT_DIR/squadv2.1.log" 2>&1 &
status=0
for job in $(jobs -p); do wait "$job" || status=1; done
for f in "$OUT_DIR"/*.log; do echo "== $f"; tail -3 "$f"; done
python "$ROOT/scripts/korbi/recall_surgery/ladder17b_table.py" "$OUT_DIR" | tee "$OUT_DIR/summary.txt"
# fail LOUDLY: every row must have produced its result file
for f in winogrande.json coqa.json squadv2.shard0of2.json squadv2.shard1of2.json; do
  [ -f "$OUT_DIR/$f" ] || { echo "[ladder17b_eval] FAIL: missing $OUT_DIR/$f"; status=1; }
done
echo "[ladder17b_eval] exit $status"
exit $status
