#!/bin/bash
# tokenizer-256k-v2-agent re-tokenization of the v8ml data on MN5 (scripts/tok_v2/retok_v2.py), one mode per
# launch (scripts/flag_decontam/run_step.sh tok2 <mode> ...; runs once per node, in the container):
#   pilot           tests, then a few shards (dclm / chat / Greek / Czech) + the fixed validation sets on one node
#   plan [parts]    the plan of every shard of datamix_v8ml_mn5.txt + the validation sets (default 4 parts)
#   run <i,j,..>    plan parts, one per node (SLURM_NODEID)
#   check [args]    counts + decoded-text samples of every finished shard
# New tree only: $OUT (the existing data is read, never written).
set -euo pipefail
mode=$1; shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
D=/gpfs/projects/ehpc533/users/korbip/data
H=${HYBRID_ENV:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019}
OUT=${TOK2_OUT:-$D/v8ml-tok2}
W=${TOK2_WORK:-/gpfs/scratch/ehpc533/korbip_tok2}
export PYTHONPATH="${PYTHONPATH:-}:$H/pylibs_gigatoken"
mkdir -p "$W"
R="python $HERE/retok_v2.py"
WORKERS=$(( $(python -c "import os; print(len(os.sched_getaffinity(0)))") - 2 ))   # not nproc: it follows OMP_NUM_THREADS
shards() {   # every shard of the mix + the validation sets
  awk 'NF && $1 !~ /^#/ {print $2}' "$H/data/datamix_v8ml_mn5.txt"
  awk '{print $3}' "$D/v8ml-valid/valid_sets.txt"
}
case "$mode" in
  pilot)
    python "$HERE/test_retok_v2.py"
    { shards | grep -E "dclm-1.0/megatron-lm/03_1_0000|Nemotron-Cascade-2-SFT-Data/megatron-lm/shard_0003|clean/ell_Grek/shard_0001|clean/ces_Latn/shard_0001|/v8ml-valid/"; } > "$W/pilot_shards.txt"
    $R plan --list "$W/pilot_shards.txt" --data-root "$D" --out "$OUT" --parts 1 > "$W/plan_pilot.json"
    $R run --plan "$W/plan_pilot.json" --part 0 --workers "$WORKERS"
    $R check --plan "$W/plan_pilot.json" --sample 200 ;;
  plan)
    shards > "$W/shards.txt"
    $R plan --list "$W/shards.txt" --data-root "$D" --out "$OUT" --parts "${1:-4}" > "$W/plan.json" ;;
  run)
    $R run --plan "$W/plan.json" --parts-of "$1" --workers "$WORKERS" ;;
  check)
    $R check --plan "$W/plan.json" "$@" ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo "[tok2 $mode] done"
