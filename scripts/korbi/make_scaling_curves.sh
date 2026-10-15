#!/usr/bin/env bash
# One-shot "make" for the architecture scaling curves.
#
# Pulls the latest eval validation losses from the cluster, then:
#   1. picks the best (gbs, lr) per (arch, size, budget)  -- BEFORE any fitting,
#      because a scaling law is the envelope L*(N,D) = min over hyperparameters
#   2. compares every hybrid against its fullattn counterpart on matched cells
#      -> delta_vs_fullattn.png, compute_multiplier.png
#   3. fits the per-architecture compute-optimal frontier
#      -> scaling_flops.png, scaling_gpu_hours.png
#
# Usage:
#   scripts/korbi/make_scaling_curves.sh                 # refresh from JUPITER, fit cooldowns
#   scripts/korbi/make_scaling_curves.sh --no-refresh    # reuse existing dump/eval_val_metrics.csv
#   HOST=jupiter TYPE=both FLOPS_MODE=full OUT=dump/scaling_laws \
#       scripts/korbi/make_scaling_curves.sh
#
# Env / overrides:
#   HOST        ssh host for the metrics pull            (default: jupiter)
#   CSV         metrics CSV path                         (default: dump/eval_val_metrics.csv)
#   OUT         output dir for curves + tables           (default: dump/scaling_laws)
#   TYPE        cooldown | stable | both                 (default: cooldown)
#   FLOPS_MODE  6nd | full                               (default: 6nd)
#   HP_SELECT   per-arch | shared | none                 (default: per-arch)
#               per-arch = each arch gets its own best lr/bs (the honest envelope);
#               shared   = one lr/bs per cell for all archs (exactly matched pairs).
#               The head-to-head in step 2 always uses matched cells regardless.
#   --no-refresh  skip the cluster pull, use existing CSV
set -euo pipefail

# repo root = two levels up from this script
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

HOST="${HOST:-jupiter}"
CSV="${CSV:-dump/eval_val_metrics.csv}"
OUT="${OUT:-dump/scaling_laws}"
TYPE="${TYPE:-cooldown}"
FLOPS_MODE="${FLOPS_MODE:-6nd}"
HP_SELECT="${HP_SELECT:-per-arch}"
REFRESH=1
[ "${1:-}" = "--no-refresh" ] && REFRESH=0

PY="${PYTHON:-python}"

echo "==> scaling curves  (host=$HOST  type=$TYPE  flops=$FLOPS_MODE  hp=$HP_SELECT  out=$OUT)"

if [ "$REFRESH" -eq 1 ]; then
  echo "==> [1/2] refreshing metrics from $HOST"
  scripts/korbi/scaling_laws/refresh_metrics.sh "$HOST" "$CSV"
else
  echo "==> [1/2] --no-refresh: using existing $CSV"
  [ -f "$CSV" ] || { echo "ERROR: $CSV not found (drop --no-refresh to pull it)"; exit 1; }
fi

echo "==> [2/2] selecting lr/bs, comparing vs fullattn, fitting + plotting"
"$PY" scripts/korbi/scaling_laws/run.py --input "$CSV" --out "$OUT" --type "$TYPE" \
      --flops-mode "$FLOPS_MODE" --hp-select "$HP_SELECT"

echo
echo "==> done."
echo "  hybrid vs fullattn (read these FIRST):"
echo "    loss ratio vs D/N: $OUT/ratio_vs_fullattn.png    <- flat = holds at any scale"
echo "    loss ratio vs C  : $OUT/ratio_vs_gpu_hours.png  $OUT/ratio_vs_flops.png"
echo "    Δ loss vs scale  : $OUT/delta_vs_fullattn.png"
echo "    compute saving   : $OUT/compute_multiplier.png"
echo "    tables           : $OUT/hybrid_vs_fullattn.csv  $OUT/hybrid_vs_fullattn_trend.csv"
echo "                       $OUT/iso_loss_multiplier.csv"
echo "    which is best    : $OUT/pairwise_ranking.csv  $OUT/ranking_by_tier.csv"
echo "  Kaplan/Chinchilla scaling law:"
echo "    log-log          : $OUT/loglog_gpu_hours.png  $OUT/loglog_flops.png"
echo "    frontier         : $OUT/chinchilla_frontier.png"
echo "    L(N,D) + CIs     : $OUT/chinchilla_fits.csv"
echo "  compute curves:"
echo "    FLOPs-loss       : $OUT/scaling_flops.png"
echo "    compute-loss     : $OUT/scaling_gpu_hours.png"
echo "    tables           : $OUT/scaling_data.csv  $OUT/fits.csv  $OUT/chinchilla.json"
echo "                       $OUT/hypotheses.csv"
