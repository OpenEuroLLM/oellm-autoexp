#!/bin/bash
# The FLAG suite's aggregate score of MN5 exports (summarize-evals, github.com/spyysalo/summarize-evals),
# run on a workstation: the result JSONs of <out>/<export> come over ssh, are collected into the
# JUPITER CSV format (collect_raw_mn5.py) and summarized.
#
#   flag_summary_mn5.sh <export> [<export> ...]        e.g. ambitious_v4_2ph_hd256_fp8_13000
#   DEPTH=4 flag_summary_mn5.sh ...                    the tree down to the single evals
#
# summarize-evals is pinned to 3cb8568 (2026-10-09): its own task/weight definitions. Its later
# revisions read quickdash's configs and need a set (configs/sets/comprehensive.yaml) neither
# quickdash repo has. Needs python >= 3.12 with pandas + pyyaml ($PY). CSVs and summaries go to
# $FLAG_SUMMARY_DIR (default dump/flag_summaries).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PY=${PY:-$HOME/.miniconda3/envs/oellm/bin/python}
REMOTE=${REMOTE:-marenostrum_new}
OUT_ROOT=${OUT_ROOT:-/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_oellm}
DATASETS_REMOTE=${DATASETS_REMOTE:-jupiter:/e/project1/e-sta-openeurollm/vanosch1/v1-vs-v2/anneal/evals-438.datasets.csv}
SUM=${FLAG_SUMMARY_DIR:-$REPO/dump/flag_summaries}
CACHE=${FLAG_SUMMARY_CACHE:-$SUM/.cache}
SE_REV=3cb8568
mkdir -p "$SUM" "$CACHE"
if [ ! -d "$CACHE/summarize-evals" ]; then
  git clone -q https://github.com/spyysalo/summarize-evals.git "$CACHE/summarize-evals"
fi
git -C "$CACHE/summarize-evals" checkout -q "$SE_REV"
[ -f "$CACHE/evals-438.datasets.csv" ] || scp -q "$DATASETS_REMOTE" "$CACHE/evals-438.datasets.csv"
for exp in "$@"; do
  res=$CACHE/$exp
  mkdir -p "$res/stage/results"
  rsync -a --prune-empty-dirs --include='*/' --exclude='response_cache/***' --exclude='queue_lm/***' \
    --exclude='launches/***' --exclude='done/***' --include='results_*.json' --exclude='*' \
    "$REMOTE:$OUT_ROOT/$exp/" "$res/raw/"
  for d in results evalchemy lighteval; do
    [ -d "$res/raw/$d" ] && ln -sfn "$res/raw/$d" "$res/stage/results/$d"
  done
  csv=$SUM/$exp.flag-evals-438.tasks.csv
  "$PY" "$HERE/collect_raw_mn5.py" --checkpoints "$exp" --runs "$res/stage" --chrf-patched --lmeval-object \
    --datasets "$CACHE/evals-438.datasets.csv" -o "$csv"
  (cd "$CACHE/summarize-evals" && "$PY" summarize_evals.py "$csv" ${DEPTH:+-d "$DEPTH"}) | tee "$SUM/$exp.summary.txt"
done
