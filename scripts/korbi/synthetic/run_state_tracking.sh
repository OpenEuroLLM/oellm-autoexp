#!/bin/bash
# CKLSTM experiment B on one 4-GPU node (from the repo root, inside the training container; launched
# as kernel_tests_mn5's aux.bench): every (mixer, task, seed) of state_tracking.py, sharded over the
# GPUs.   run_state_tracking.sh <out dir> [extra state_tracking.py args, e.g. --layers 4]
set -u
OUT=$1; shift
mkdir -p "$OUT"
export CC=${CC:-gcc}
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/korbi/synthetic/state_tracking.py --seeds 0 1 --shard $i/4 \
        --out "$OUT/shard_$i.jsonl" "$@" > "$OUT/shard_$i.log" 2>&1 &
done
wait
cat "$OUT"/shard_*.jsonl | wc -l
