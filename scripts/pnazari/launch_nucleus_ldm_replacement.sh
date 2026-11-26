#!/bin/bash
# LDM replacement intervention on the Nucleus main line v4b (experiments/pnazari/nucleus_ldm_replacement_mn5): the
# checkpoint surgery job, then the 500-step warm-start arms into a pool. The arms wait (start conditions) for the
# surgery's checkpoints, so both are submitted at once. The control is v4b itself (no ctrl arm).
#
# Run from a snapshot whose Megatron-LM and cute_kernels have the FP16 M / M checkpoint flags
# (--depth-memory-storage-dtype, --depth-memory-checkpoint-interval), as the owner of v4b's checkpoint files (the
# surgery hard-links them):
#   POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_c136 PIN=host005 \
#       bash scripts/pnazari/launch_nucleus_ldm_replacement.sh
# Environment:
#   POOL          the pool to enqueue the arms into (128 nodes each)
#   PIN           optional: print the pool_ctl pin command for this host of POOL (the run to pin exists only
#                 once the surgery checkpoint does)
#   STEP          the source save of v4b's ckpt_eval (default 44000); the arms train STEP -> STEP + 500
#   ARMS          default collapse,ldm_fixed
#   SURGERY_ARGS  sbatch options of the surgery job (default "-A ehpc533 -q acc_ehpc"; it needs one node, ~10 min)
#   DRY_RUN=1     render the arms' sbatch scripts below ./.dryrun_ldm (no surgery, nothing submitted)
# The step-0 loss jump (eval only, once the surgery is done):
#   ... run_autoexp.py --config-name experiments/pnazari/nucleus_ldm_replacement_eval_mn5 aux.step=$STEP \
#       "aux.only=[v4b,collapse,ldm_fixed]" --pool "$POOL" --yes
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${R:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
STEP=${STEP:-44000}
END=$((STEP + 500))
IT=iter_$(printf %07d "$STEP")
ARMS=${ARMS:-collapse,ldm_fixed}
KV=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116
MAIN=$KV/runs/v4b_learned/main/ckpt_eval
OUT=$KV/ldm_replacement
CONFIG=experiments/pnazari/nucleus_ldm_replacement_mn5
OVERRIDES=(aux.start="$STEP" aux.end="$END" aux.start_iter="$IT" "aux.only=[$ARMS]")

grep -q "depth_memory_checkpoint_interval" submodules/Megatron-LM/megatron/core/transformer/transformer_config.py \
    || { echo "this Megatron-LM has no depth_memory_checkpoint_interval (NVIDIA-Megatron-LM#8)"; exit 1; }
grep -q "checkpoint_interval" submodules/cute_kernels/cute_ldm/workspace.py \
    || { echo "this cute_kernels has no M checkpoints (cute_kernels#1)"; exit 1; }
# a save on a validation step is the 11-13 host-RAM OOM (v4_cont_mn5); eval_interval is 200
[ $((END % 200)) -ne 0 ] || { echo "STEP=$STEP: the final save at $END falls on a validation step"; exit 1; }

if [ -n "${DRY_RUN:-}" ]; then
    rm -rf .dryrun_ldm
    PYTHONPATH=. "$R/autoexp_env/bin/python" scripts/run_autoexp.py --config-name "$CONFIG" \
        --dry-run-root .dryrun_ldm --yes "${OVERRIDES[@]}"
    exit 0
fi
[ -n "${POOL:-}" ] || { echo "set POOL (e.g. /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264)"; exit 1; }
[ -f "$MAIN/$IT/.metadata" ] || { echo "no checkpoint $MAIN/$IT"; exit 1; }

for v in collapse ldm; do
    t=$OUT/surgery/$v/ckpt/latest_checkpointed_iteration.txt
    [ ! -f "$t" ] || [ "$(cat "$t")" = "$STEP" ] \
        || { echo "$OUT/surgery/$v holds step $(cat "$t"), not $STEP: move it away first"; exit 1; }
done
if [ -f "$OUT/surgery/collapse/ckpt/latest_checkpointed_iteration.txt" ] \
    && [ -f "$OUT/surgery/ldm/ckpt/latest_checkpointed_iteration.txt" ]; then
    echo "surgery checkpoints for $STEP exist"
else
    mkdir -p "$OUT/logs"
    # shellcheck disable=SC2086
    sbatch ${SURGERY_ARGS:--A ehpc533 -q acc_ehpc} --output="$OUT/logs/%x-%j.out" \
        --export=ALL,STEP="$STEP",SRC="$MAIN/$IT",OUT="$OUT/surgery" scripts/pnazari/mhc_to_ldm_surgery.sbatch
fi

PYTHONPATH=. HYBRID_OUTPUT_DIR=/gpfs/scratch/ehpc533/hybrid_nucleus scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py \
    --no-monitor --monitor-state-dir "$R/pool_state" --pool "$POOL" --config-name "$CONFIG" --yes "${OVERRIDES[@]}"

if [ -n "${PIN:-}" ]; then
    # the monitor creates the pool run only once the arm's start condition (the surgery checkpoint) passes,
    # so the pin cannot be set here
    echo "once the surgery is done and runs/<rid> exists, pin each arm:"
    echo "  PYTHONPATH=. $R/autoexp_env/bin/python scripts/pool_ctl.py $POOL pin <rid> $PIN"
fi
