#!/bin/bash
# Per-stream Engram gates on the Nucleus main line v4b (experiments/rluukkon/nucleus_engram_branch_gates_mn5):
# the checkpoint surgery job, then the 500-step warm-start arms into a pool. The arms wait (start conditions)
# for the surgery's checkpoints, so both are submitted at once. The control is v4b itself (no ctrl arm).
#
# Run from a snapshot with engram_branch_gates (autoexp and its Megatron-LM submodule), as the owner of v4b's
# checkpoint files (the surgery hard-links them):
#   POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_c136 PIN=host005 ARMS=stream \
#       bash scripts/rluukkon/launch_nucleus_engram_branch_gates.sh
# Environment:
#   POOL          the pool to enqueue the arms into (128 nodes each)
#   PIN           optional: pin the enqueued arms to this host of POOL (pool_ctl pin)
#   STEP          the source save of v4b's ckpt_eval (default 44000); the arms train STEP -> STEP + 500
#   ARMS          default read,learned,stream
#   SURGERY_ARGS  sbatch options of the surgery job (default "-A ehpc533 -q acc_ehpc"; it needs one node, ~5 min)
#   DRY_RUN=1     render the arms' sbatch scripts below ./.dryrun_bg (no surgery, nothing submitted)
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
STEP=${STEP:-44000}
END=$((STEP + 500))
IT=iter_$(printf %07d "$STEP")
ARMS=${ARMS:-read,learned,stream}
KV=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116
MAIN=$KV/runs/v4b_learned/main/ckpt_eval
OUT=$KV/engram_branch_gates
CONFIG=experiments/rluukkon/nucleus_engram_branch_gates_mn5
OVERRIDES=(aux.start="$STEP" aux.end="$END" aux.start_iter="$IT" "aux.only=[$ARMS]")

grep -q "engram_branch_gates" submodules/Megatron-LM/megatron/core/transformer/transformer_config.py \
    || { echo "this Megatron-LM has no engram_branch_gates: check out feat/nucleus-engram"; exit 1; }
# a save on a validation step is the 11-13 host-RAM OOM (v4_cont_mn5); eval_interval is 200
[ $((END % 200)) -ne 0 ] || { echo "STEP=$STEP: the final save at $END falls on a validation step"; exit 1; }

if [ -n "${DRY_RUN:-}" ]; then
    rm -rf .dryrun_bg
    PYTHONPATH=. "$R/autoexp_env/bin/python" scripts/run_autoexp.py --config-name "$CONFIG" \
        --dry-run-root .dryrun_bg --yes "${OVERRIDES[@]}"
    exit 0
fi
[ -n "${POOL:-}" ] || { echo "set POOL (e.g. /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264)"; exit 1; }
[ -f "$MAIN/$IT/.metadata" ] || { echo "no checkpoint $MAIN/$IT"; exit 1; }

for v in bg bg_qmix; do
    t=$OUT/surgery/$v/ckpt/latest_checkpointed_iteration.txt
    [ ! -f "$t" ] || [ "$(cat "$t")" = "$STEP" ] \
        || { echo "$OUT/surgery/$v holds step $(cat "$t"), not $STEP: move it away first"; exit 1; }
done
if [ -f "$OUT/surgery/bg/ckpt/latest_checkpointed_iteration.txt" ] \
    && [ -f "$OUT/surgery/bg_qmix/ckpt/latest_checkpointed_iteration.txt" ]; then
    echo "surgery checkpoints for $STEP exist"
else
    mkdir -p "$OUT/logs"
    # shellcheck disable=SC2086
    sbatch ${SURGERY_ARGS:--A ehpc533 -q acc_ehpc} --output="$OUT/logs/%x-%j.out" \
        --export=ALL,STEP="$STEP",SRC="$MAIN/$IT",OUT="$OUT/surgery" scripts/rluukkon/engram_branch_gates_surgery.sbatch
fi

PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py \
    --no-monitor --monitor-state-dir "$R/pool_state" --pool "$POOL" --config-name "$CONFIG" --yes "${OVERRIDES[@]}"

if [ -n "${PIN:-}" ]; then
    for arm in ${ARMS//,/ }; do
        rid=$(grep -l "\"name\": \"nucleus_engram_bg_$arm\"" "$POOL"/runs/*/run.json 2>/dev/null \
            | while read -r f; do d=$(dirname "$f"); [ -f "$d/rc" ] || [ -f "$d/started" ] || basename "$d"; done | tail -1)
        [ -n "$rid" ] || { echo "no queued run nucleus_engram_bg_$arm in $POOL to pin"; continue; }
        PYTHONPATH=. "$R/autoexp_env/bin/python" scripts/pool_ctl.py "$POOL" pin "$rid" "$PIN"
    done
fi
