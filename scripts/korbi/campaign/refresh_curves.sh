#!/bin/bash
# Learning-curve fits + health indicators for every run of the 3B campaign
# (docs/hybrid_moe_3b_campaign.md §8.2), refreshed from MareNostrum 5:
#
#   scripts/korbi/campaign/refresh_curves.sh [suite-glob ...]      (default: all suites)
#
# 1. rsyncs the Megatron logs (slurm-*.log, task_*.sh only) of campaign_3b from MN5 into
#    $CAMPAIGN_LOGS (default ~/.cache/campaign_3b_logs);
# 2. runs scripts/korbi/campaign/curves.py over every suite (in-progress runs are fitted on what
#    they have; runs still in warmup are skipped);
# 3. writes dump/campaign_3b/curves_all.csv (every column: E, A, alpha, t0, lr_power with
#    standard errors, fit RMSE, backtest error, flattening, spikes, growth, noise, predictions)
#    and curves_all.md (the main columns), and prints the per-suite summary.
set -euo pipefail
cd "$(dirname "$0")/../../.."
REMOTE=${REMOTE:-marenostrum_new:/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/campaign_3b}
LOGS=${CAMPAIGN_LOGS:-$HOME/.cache/campaign_3b_logs}
OUT=dump/campaign_3b
mkdir -p "$LOGS" "$OUT"
rsync -a --include="*/" --include="slurm-*.log" --include="task_*.sh" --exclude="*" \
    "$REMOTE/" "$LOGS/" 2>/dev/null || echo "warning: rsync from $REMOTE failed; using the cached logs"
if [ $# -gt 0 ]; then
    suites=()
    for g in "$@"; do for d in "$LOGS"/$g/; do [ -d "$d" ] && suites+=("$d"); done; done
else
    mapfile -t suites < <(ls -d "$LOGS"/*/)
fi
uv run --project ../loss_prediction python scripts/korbi/campaign/curves.py "${suites[@]}" \
    --csv "$OUT/curves_all.csv" 2>/dev/null | grep -v Warning > "$OUT/curves_all.md"
uv run --project ../loss_prediction python scripts/korbi/campaign/curves_summary.py "$OUT/curves_all.csv"
echo "wrote $OUT/curves_all.csv, $OUT/curves_all.md"
