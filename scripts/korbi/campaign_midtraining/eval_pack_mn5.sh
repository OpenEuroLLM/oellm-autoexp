#!/bin/bash
# Many single-GPU long-context evals in one multi-node pool run: every node (one srun task each) starts one worker per
# GPU; the workers take "<arm>:<task>[:<step>]" items from the list in turn (claimed with an atomic mkdir on GPFS,
# so a fast worker takes more), each item run by t16_long_evals_mn5.sh on that GPU, logged to <queue>/logs/. An item
# with a done marker is skipped, so a rerun of the same pack finishes the rest. A pack needs >= 4 nodes on p264: a
# 1-3 node run would fit host008's spare nodes (the v4b main run's buffer).
#
#   eval_pack_mn5.sh <queue dir> <arm:task[:step]> ...
set -u
Q=$1; shift
items=("$@")
gpus=${GPUS_PER_NODE:-4}
tag=${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0}
mkdir -p "$Q/claims" "$Q/done" "$Q/logs"
worker() {
    local g=$1 status=0
    for it in "${items[@]}"; do
        local key=${it//[:\/]/_}
        [ -e "$Q/done/$key" ] && continue
        # one claim per attempt: an earlier attempt's claim without a done marker is taken again
        mkdir "$Q/claims/$key.$tag" 2>/dev/null || continue
        local arm task step
        IFS=: read -r arm task step <<< "$it"
        echo "[eval_pack] $(hostname) gpu $g: $it"
        if CUDA_VISIBLE_DEVICES=$g bash scripts/korbi/campaign_midtraining/t16_long_evals_mn5.sh "$arm" "$task" \
                "${step:-3000}" > "$Q/logs/$key.$tag.log" 2>&1; then
            touch "$Q/done/$key"
            echo "[eval_pack] done $it"
        else
            status=1
            echo "[eval_pack] FAILED $it (log $Q/logs/$key.$tag.log)"
        fi
    done
    return $status
}
pids=()
for g in $(seq 0 $((gpus - 1))); do
    worker "$g" &
    pids+=($!)
done
status=0
for p in "${pids[@]}"; do
    wait "$p" || status=1
done
echo "[eval_pack] $(hostname) exit $status"
exit $status
