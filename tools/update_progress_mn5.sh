#!/bin/bash
# Accumulate MareNostrum 5 training progress (all model sizes) into the shared,
# all-cluster files of the results repo:
#   $RESULTS_REPO/training/progress/progress.csv
#   $RESULTS_REPO/training/progress/progress.md
# Rows of LUMI / LEO already in those files are kept; only MN5 rows are replaced.
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

source /gpfs/projects/ehpc533/users/diana/init.sh

: "${RESULTS_REPO:?set RESULTS_REPO to your clone of dense_multilingual_models_scaling_results}"
AUTOEXP=${AUTOEXP:-/gpfs/projects/ehpc533/users/diana/oellm-autoexp}
PY=${PY:-python}
CFG=config/experiments/multilingual_scaling/training/mn5
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
  echo "=== MN5 $size ==="
  # shellcheck disable=SC2086
  "$PY" tools/progress_tracker.py "$CFG/$cfg" --machine MN5 --accumulate \
    --results-dir "$(results_dir "$CFG/$cfg")" ${cache[@]+"${cache[@]}"} ${EXTRA_ARGS:-} \
    --csv "$OUT/progress.csv" --md "$OUT/progress.md" ${mon[@]+"${mon[@]}"}
}

# Monitor-state sessions of the original MN5 launches (some live in other users' clones).
track 0.1B_ne qwen3_dense_0.21B.yaml \
  "$MON/1778164338" "$MON/1778489512" "$MON/1778500854" "$MON/1778500975" "$MON/1778669257"
track 0.2B_ne qwen3_dense_0.39B.yaml \
  "$MON/1778164824" "$MON/1778494928"
track 0.4B_ne qwen3_dense_0.71B.yaml \
  /gpfs/projects/ehpc533/oellm-autoexp-joan/monitor_state/1778252010 \
  /gpfs/projects/ehpc533/oellm-autoexp-joan/monitor_state/1778413340
track 0.9B_ne qwen3_dense_1.26B.yaml \
  /gpfs/projects/ehpc533/users/slaing00/oellm-autoexp/monitor_state/1778490317 \
  /gpfs/projects/ehpc533/users/slaing00/oellm-autoexp/monitor_state/1778572714

for size_cfg in qwen3_dense_0.21B qwen3_dense_0.39B qwen3_dense_0.71B qwen3_dense_1.26B; do
  echo "=== Syncing runs ($size_cfg) ==="
  "$PY" tools/sync_runs.py -f "$(results_dir "$CFG/$size_cfg.yaml")"
done

echo "=== Done. Review and publish: ==="
echo "  git -C $RESULTS_REPO diff --stat -- training/progress"
echo "  git -C $RESULTS_REPO commit -m 'progress: MN5 update' -- training/progress && git -C $RESULTS_REPO push"
