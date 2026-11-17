#!/bin/bash
# Render, verify and submit the dataset-index prebuild for the v4 continuation pair (one rank).
# Runs on the MN5 login node from the frozen snapshot that holds the configs:
#
#   bash scripts/korbi/recall_surgery/launch_prebuild_v4_cont.sh [--val] [--no-submit]
#
# --val  build the index for the NORMAL continuation instead (v4_cont_mn5 with the original run's
#        validation back on: eval_interval 1000, eval_iters 10, full_validation True). The train split
#        is unchanged, so the eval-less build already covers it and this one only adds the valid
#        caches -- but run it only AFTER the eval-less prebuild has finished, never concurrently.
#
# Why: the 128-node control (v4_cont_mn5, 512 ranks over a 531-prefix blend) twice lost a rank to the
# host-RAM OOM killer while building the train index (mn5-pool-p264.1261/.1264, "Detected 1 oom_kill");
# the proven 264-node layout builds it with 1024 ranks, i.e. half the index per rank. The prebuild
# writes the same cache key once, on one rank, so the control starts without a build.
#
# The verifier (check_prebuild_render.py) refuses a render that loads a checkpoint, carries a node
# check, or differs from the run in any cache-key flag. Nothing is submitted unless it passes.
set -uo pipefail

REPO=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_v4cont1
PY=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_env/bin/python
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
STATE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/pool_state
CONTROL_RUN=${CONTROL_RUN:-mn5-pool-p264.1261}   # the control's step.sh: the key reference
SUBMIT=1 VAL=0
for arg in "$@"; do
    case $arg in
        --no-submit) SUBMIT=0 ;;
        --val) VAL=1 ;;
    esac
done
if [ "$VAL" = 1 ]; then
    CONFIG=experiments/korbi/recall_surgery/data_cache_prebuild_v4_cont_val_mn5
    SERIES=v4_cont_val_prebuild
    DRY=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/dryrun_launch_val
else
    CONFIG=experiments/korbi/recall_surgery/data_cache_prebuild_v4_cont_mn5
    SERIES=v4_cont_prebuild
    DRY=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/dryrun_launch
fi

cd "$REPO" || exit 97
rm -rf "$DRY"

echo "=== 1/4 render"
PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus \
    "$PY" scripts/run_autoexp.py --dry-run --dry-run-root "$DRY" --config-name "$CONFIG" --yes \
    > /tmp/prebuild_launch_dry.log 2>&1
rc=$?
echo "render rc=$rc"; tail -3 /tmp/prebuild_launch_dry.log
[ "$rc" = 0 ] || { echo "REFUSED: render failed"; exit 3; }

RENDER=$(find "$DRY" -name "*.sbatch" | head -1)
echo "render: $RENDER"
[ -n "$RENDER" ] || { echo "REFUSED: no sbatch rendered"; exit 3; }

echo "=== 2/4 verify against $CONTROL_RUN"
"$PY" scripts/korbi/recall_surgery/check_prebuild_render.py \
    "$RENDER" "$POOL/runs/$CONTROL_RUN/step.sh" || { echo "REFUSED: verifier said no"; exit 3; }

if [ "$SUBMIT" = 0 ]; then
    echo "=== 3/4 --no-submit: stopping after the verification"
    exit 0
fi

echo "=== 3/4 cancel a previous prebuild of this series (pool-level, ours)"
for d in "$POOL"/runs/*/; do
    rid=$(basename "$d")
    [ -f "$d/run.json" ] || continue
    name=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["name"])' "$d/run.json" 2>/dev/null)
    [ "$name" = "$SERIES" ] || continue
    [ -f "$d/rc" ] && continue
    echo "  cancelling $rid ($name)"
    PYTHONPATH=. "$PY" scripts/pool_ctl.py "$POOL" cancel "$rid" 2>&1 | tail -1
done

echo "=== 4/4 submit"
PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus \
    "$PY" scripts/run_autoexp.py --submit-and-exit --yes --monitor-state-dir "$STATE" \
    --pool "$POOL" --config-name "$CONFIG" 2>&1 | tail -3
