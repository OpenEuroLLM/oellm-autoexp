#!/bin/bash
# Accumulate LUMI training progress (all model sizes) into the shared,
# all-cluster files of the results repo:
#   $RESULTS_REPO/training/progress/progress.csv
#   $RESULTS_REPO/training/progress/progress.md
# Rows of LEO / MN5 already in those files are kept; only LUMI rows are replaced.
#
# Workflow: this script pulls the results repo first; afterwards review the diff,
# commit and push, so the next cluster's run starts from these rows.
#
# Overridable: AUTOEXP, RESULTS_REPO, CACHE, PY, NO_PULL=1, EXTRA_ARGS (e.g. "--compute-storage").
# Sizes run sequentially on purpose: they all write the same CSV.
set -euo pipefail

AUTOEXP=${AUTOEXP:-/scratch/project_465002530/users/dianaonutu/oellm-autoexp}
RESULTS_REPO=${RESULTS_REPO:-/scratch/project_465002530/users/dianaonutu/dense_multilingual_models_scaling_results}
CACHE=${CACHE:-/scratch/project_465002530/users/dianaonutu/cache/progress_tracker}
PY=${PY:-$AUTOEXP/.venv/bin/python}
RES=/scratch/project_465002530/multilingual_scaling
CFG=config/experiments/multilingual_scaling/training/lumi
OUT=$RESULTS_REPO/training/progress

cd "$AUTOEXP"
[ "${NO_PULL:-0}" = 1 ] || git -C "$RESULTS_REPO" pull --ff-only

track () {  # <size> <config> [monitor dirs...]  (no monitor dirs = auto-discover)
  local size=$1 cfg=$2; shift 2
  local mon=(); [ $# -gt 0 ] && mon=(--monitor-dirs "$@")
  echo "=== LUMI $size ==="
  # shellcheck disable=SC2086
  "$PY" tools/progress_tracker.py "$CFG/$cfg" --machine LUMI --accumulate \
    --results-dir "$RES/$size/training" --cache-dir "$CACHE" ${EXTRA_ARGS:-} \
    --csv "$OUT/progress.csv" --md "$OUT/progress.md" ${mon[@]+"${mon[@]}"}
}

track 0.1B_ne qwen3_dense_0.21B.yaml
track 0.2B_ne qwen3_dense_0.39B.yaml
track 0.4B_ne qwen3_dense_0.71B.yaml
track 0.9B_ne qwen3_dense_1.26B.yaml

echo "=== Done. Review and publish: ==="
echo "  git -C $RESULTS_REPO diff --stat -- training/progress"
echo "  git -C $RESULTS_REPO commit -m 'progress: LUMI update' -- training/progress && git -C $RESULTS_REPO push"
