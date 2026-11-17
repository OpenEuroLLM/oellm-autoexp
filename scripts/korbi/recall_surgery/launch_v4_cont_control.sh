#!/bin/bash
# Render, verify and enqueue the v4 continuation from 35900 (128 nodes, mbs 2) -- the vanilla twin of
# the v4a shared-KV arm, and the line that carries the normal training on (the original run's
# validation included). Own base folder: production/v4_cont. Runs on the MN5 login node:
#
#   bash scripts/korbi/recall_surgery/launch_v4_cont_control.sh [--no-submit] [--allow-missing-cache]
#
# Gates, all of them before anything is enqueued:
#   1. render
#   2. check_control_render.py against KEY_REF -- by default the MAIN RUN's own command
#      (mn5-pool-p264.0987/step.sh): same recipe, same dataset cache key, vanilla stack, main 35900 load
#   3. the dataset index must be complete in the main run's cache dir: scripts/check_data_cache.py
#      (file presence is all Megatron checks, so a truncated -npy from a killed build would be accepted
#      and then die at load; the check is what makes the cache trustworthy). --allow-missing-cache skips
#      this and lets the run build in-run -- which is where the earlier attempts were OOM-killed.
set -uo pipefail

REPO=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_v4cont1
PY=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_env/bin/python
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
STATE=/gpfs/projects/ehpc533/users/korbip/hybrid_v019/pool_state
CONFIG=experiments/korbi/recall_surgery/v4_cont_mn5
CACHE=/gpfs/scratch/ehpc533/hybrid_nucleus/data_cache_shared
# Memory for the 128-node layout (512 GPUs instead of the main run's 1024): the backward OOMed in the
# activation-recompute path (mn5-pool-p264.1287), so every layer is recomputed. Passed as a Hydra
# command-line override because a file-level backend.megatron.recompute_num_layers: 98 did not win the
# merge (the rendered compose still carried 2, from a source I could not identify in the chain).
# These two keys are NOT honoured from the config file at submission time (only in the dry render):
# a file-level value of 98 rendered correctly but the submitted step.sh carried 2
# (mn5-pool-p264.1295/.1302). Command-line overrides do reach the submission, so they are passed here.
#   micro_batch_size 1 = one 64k sequence per micro-batch (v4's footprint); gbs 1024 at dp 512 then
#                        runs 2 micro-batches per step. mbs 2 doubles the activations and OOMed the
#                        backward (mn5-pool-p264.1287).
#   recompute_num_layers 2 = the main run's own setting (the OOM was the micro-batch, not the recompute).
OVERRIDES=(backend.megatron.micro_batch_size=1 backend.megatron.recompute_num_layers=2)
# the recipe keys are read from the parallel cache the kvshare_k_* runs built (same recipe, so
# the same keys); the run itself reads $CACHE
KEY_SOURCE=${KEY_SOURCE:-/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116/runs/data_cache_shared}
KEY_REF=${KEY_REF:-$POOL/runs/mn5-pool-p264.0987/step.sh}
DRY=/gpfs/scratch/ehpc533/hybrid_nucleus/production/v4_cont/dryrun_control
SUBMIT=1 ALLOW=0
for arg in "$@"; do
    case $arg in
        --no-submit) SUBMIT=0 ;;
        --allow-missing-cache) ALLOW=1 ;;
    esac
done

cd "$REPO" || exit 97
rm -rf "$DRY"

echo "=== 1/5 render"
PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus \
    "$PY" scripts/run_autoexp.py --dry-run --dry-run-root "$DRY" --config-name "$CONFIG" --yes \
    "${OVERRIDES[@]}" > /tmp/control_launch_dry.log 2>&1
rc=$?
echo "render rc=$rc"; tail -3 /tmp/control_launch_dry.log
[ "$rc" = 0 ] || { echo "REFUSED: render failed"; exit 3; }

RENDER=$(find "$DRY" -name "*.sbatch" | head -1)
echo "control render: $RENDER"
echo "key reference : $KEY_REF"
[ -n "$RENDER" ] || { echo "REFUSED: no control sbatch rendered"; exit 3; }
[ -f "$KEY_REF" ] || { echo "REFUSED: key reference missing"; exit 3; }

echo "=== 1b/5 the memory-relevant geometry must be in the render"
# mbs 1 = one 64k sequence per micro-batch (the main run's footprint); gbs 1024 at dp 512 then runs
# 2 micro-batches per step. mbs 2 doubles the activations and OOMed the backward.
GEOM=$(grep -oE "\-\-(micro-batch-size|global-batch-size) [0-9]+" "$RENDER" | sort -u | tr "\n" " ")
echo "  $GEOM"
echo "$GEOM" | grep -q -- "--micro-batch-size 1" || { echo "REFUSED: micro-batch-size is not 1"; exit 3; }
echo "$GEOM" | grep -q -- "--global-batch-size 1024" || { echo "REFUSED: global-batch-size is not 1024"; exit 3; }

echo "=== 2/5 verify the render against the main run's own command"
"$PY" scripts/korbi/recall_surgery/check_control_render.py "$RENDER" "$KEY_REF" \
    || { echo "REFUSED: verifier said no"; exit 3; }

echo "=== 3/5 dataset index ready for this recipe in $CACHE ?"
if [ "$ALLOW" = 1 ]; then
    echo "  skipped (--allow-missing-cache)"
else
    # targeted: only the (key, split) rows this recipe hashes to, the keys taken from the parallel
    # cache that was built with the same recipe. The full check_data_cache.py scan over the shared
    # cache (182k files) takes tens of minutes on GPFS and is not usable as a launch gate.
    "$PY" scripts/korbi/recall_surgery/cache_ready.py "$CACHE" "$KEY_SOURCE" | tail -8
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "REFUSED: the cache is not ready for this recipe (see above)"; exit 3; }
fi

if [ "$SUBMIT" = 0 ]; then
    echo "=== --no-submit: stopping after the verification"
    exit 0
fi

echo "=== 4/5 cancel a previous run of this series (pool-level, ours)"
for d in "$POOL"/runs/*/; do
    rid=$(basename "$d")
    [ -f "$d/run.json" ] || continue
    name=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["name"])' "$d/run.json" 2>/dev/null)
    case "$name" in ambitious_v4_cont_ctrl_128|ambitious_v4_cont_main_128) ;; *) continue ;; esac
    [ -f "$d/rc" ] && continue
    echo "  cancelling $rid ($name)"
    PYTHONPATH=. "$PY" scripts/pool_ctl.py "$POOL" cancel "$rid" 2>&1 | tail -1
done

echo "=== 5/5 submit"
PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus \
    "$PY" scripts/run_autoexp.py --submit-and-exit --yes --monitor-state-dir "$STATE" \
    --pool "$POOL" --config-name "$CONFIG" "${OVERRIDES[@]}" 2>&1 | tail -3

echo "=== 6/6 verify what was ACTUALLY submitted (the pool's own step.sh)"
NEWEST=$(ls -d "$POOL"/runs/mn5-pool-p264.* | sed -E "s/.*\.//" | sort -n | tail -1)
SUB="$POOL/runs/mn5-pool-p264.$NEWEST/step.sh"
GEOM2=$(grep -oE "\-\-(micro-batch-size|global-batch-size|recompute-num-layers) [0-9]+" "$SUB" 2>/dev/null | sort -u | tr "\n" " ")
echo "  run .$NEWEST: $GEOM2"
echo "$GEOM2" | grep -q -- "--micro-batch-size 1" || { echo "  WARNING: submitted step is not mbs 1 -> writing its stop marker"; date +%s.%N > "$POOL/runs/mn5-pool-p264.$NEWEST/stop"; exit 3; }
echo "$GEOM2" | grep -q -- "--recompute-num-layers 2" || { echo "  WARNING: submitted step does not have recompute 2 -> writing its stop marker"; date +%s.%N > "$POOL/runs/mn5-pool-p264.$NEWEST/stop"; exit 3; }
echo "  OK: submitted geometry is mbs 1 / gbs 1024 / recompute 2"
