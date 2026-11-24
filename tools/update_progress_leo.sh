#!/bin/bash
# Accumulate Leonardo training progress (all model sizes) into the shared,
# all-cluster files of the results repo:
#   $RESULTS_REPO/training/progress/progress.csv
#   $RESULTS_REPO/training/progress/progress.md
# Rows of LUMI / MN5 already in those files are kept; only LEO rows are replaced.
#
# Workflow: this script pulls the results repo first; afterwards review the diff,
# commit and push, so the next cluster's run starts from these rows.
#
# Required: RESULTS_REPO = your clone of dense_multilingual_models_scaling_results.
# Overridable: AUTOEXP, CACHE (passed as --cache-dir when set), PY, NO_PULL=1,
#              EXTRA_ARGS (e.g. "--compute-storage").
# Results dirs are read from each config's job.base_output_dir.
# Sizes run sequentially on purpose: they all write the same CSV.
set -euo pipefail

source /leonardo_work/OELLM_prod2026/users/donutu00/init.sh

: "${RESULTS_REPO:?set RESULTS_REPO to your clone of dense_multilingual_models_scaling_results}"
AUTOEXP=${AUTOEXP:-/leonardo_work/OELLM_prod2026/users/donutu00/oellm-autoexp}
PY=${PY:-python}
CFG=config/experiments/multilingual_scaling/training/leo
OUT=$RESULTS_REPO/training/progress
MON=$AUTOEXP/monitor_state

cd "$AUTOEXP"
[ "${NO_PULL:-0}" = 1 ] || git -C "$RESULTS_REPO" pull --ff-only

results_dir () {  # job.base_output_dir of a config, without the /${job.name} tail
  sed -n 's/^[[:space:]]*base_output_dir:[[:space:]]*"\{0,1\}\(.*\)\/\${job\.name}.*/\1/p' "$1" | head -1
}

track () {  # <size> <config> [monitor dirs...]  (no monitor dirs = auto-discover)
  local size=$1 cfg=$2; shift 2
  local mon=(); [ $# -gt 0 ] && mon=(--monitor-dirs "$@")
  local cache=(); [ -n "${CACHE:-}" ] && cache=(--cache-dir "$CACHE")
  echo "=== LEO $size ==="
  # shellcheck disable=SC2086
  "$PY" tools/progress_tracker.py "$CFG/$cfg" --machine LEO --accumulate \
    --results-dir "$(results_dir "$CFG/$cfg")" ${cache[@]+"${cache[@]}"} ${EXTRA_ARGS:-} \
    --csv "$OUT/progress.csv" --md "$OUT/progress.md" ${mon[@]+"${mon[@]}"}
}

track 0.1B_ne qwen3_dense_0.21B.yaml
track 0.2B_ne qwen3_dense_0.39B.yaml
track 0.4B_ne qwen3_dense_0.71B.yaml "$MON/1779225991" "$MON/1779664774" "$MON/1779668268"
track 0.9B_ne qwen3_dense_1.26B.yaml

echo "=== Syncing runs 0.4B_ne ==="
"$PY" tools/sync_runs.py -f "$(results_dir "$CFG/qwen3_dense_0.71B.yaml")"

echo "=== Done. Review and publish: ==="
echo "  git -C $RESULTS_REPO diff --stat -- training/progress"
echo "  git -C $RESULTS_REPO commit -m 'progress: LEO update' -- training/progress && git -C $RESULTS_REPO push"
