#!/bin/bash
# The three retrieval/commonsense rows on a jupiter 1.7B ladder export, on MareNostrum 5, through
# plain HF lm-eval (goal 2026-11-11). The MN5 FLAG runner (flag_lmeval.py -> oellm runner) is vLLM-only and
# builds the model from the MN5 config schema, so it cannot load a jupiter export; the weights here are
# byte-identical to the release format and load with trust_remote_code=True after
# scripts/korbi/production/prep_ladder17b_hf_for_flag.sh has renamed the classes.
#
#   ladder17b_lm_eval.sh <model dir> <out dir> [coqa winogrande squadv2]
#
# The prompts, the metric and the n-shot values are lm-eval's own CoQA / Winogrande / SQuAD v2 tasks, i.e. the
# same definitions the FLAG rows use (config/eval_tasks/ckdab_fast.yaml: coqa 0-shot, winogrande 0-shot,
# squadv2 10-shot), so the numbers are comparable with the ambitious v4 rows in
# docs/recall_surgery_ambitious.md.
set -u
M=${1:?model dir}
OUT=${2:?out dir}
shift 2
TASKS=${*:-coqa winogrande squadv2}
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
export PYTHONPATH=${PYLIBS_OELLM:-$R/pylibs_oellm}:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-$R/cache/hf}
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT"

run() { # run <task> <n_shot>
  local task=$1 shot=$2 bs=${3:-8}
  echo "[ladder17b] $task ${shot}-shot, $M"
  python -m lm_eval --model hf \
    --model_args "pretrained=$M,trust_remote_code=True,dtype=bfloat16" \
    --tasks "$task" --num_fewshot "$shot" --batch_size "$bs" \
    --output_path "$OUT/$task.json" 2>&1 | tail -25
}

for t in $TASKS; do
  case $t in
    coqa)       run coqa 0 8 ;;
    winogrande) run winogrande 0 16 ;;
    squadv2)    run squadv2 10 4 ;;
    *) echo "unknown task $t" >&2; exit 2 ;;
  esac
done
echo "[ladder17b] done: $OUT"
