#!/bin/bash
# One step of the delimiter re-tokenization (docs/flag_retokenization.md) inside the training image on a
# MareNostrum 5 node; config/experiments/korbi/decontam/flag_retok_mn5.yaml runs it once per node under srun.
#
#   run_step.sh <step> [extra args for retok.py]
#     selftest  test_retok.py (tiny tokenizer, synthetic shards)
#     find      decode every sequence of the mix, find the delimiters ($WORK/find; split over the job's nodes)
#     write     flag-agenttok: re-encoded shards (those with delimiters), symlinks for the rest
#     decon     flag-decon-v2-agenttok: flag-decon-v2's dropped documents applied to flag-agenttok
#     verify    decoded text / untouched sequences / counts against flag/
set -euo pipefail
step=$1; shift || true
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-/gpfs/projects/ehpc533/training-data-collection/flag}
WORK=${WORK:-/gpfs/scratch/ehpc533/users/korbip/flag_retok}
OUT=${OUT:-/gpfs/projects/ehpc533/users/korbip/data/flag-agenttok}
mkdir -p "$WORK"
cd "$HERE/../.."
awk -v r="$ROOT" 'NF {print r "/" $2}' "$ROOT/datamix.txt" > "$WORK/shards_all.txt.${SLURM_NODEID:-0}.$$"
mv -f "$WORK/shards_all.txt.${SLURM_NODEID:-0}.$$" "$WORK/shards_all.txt"
echo "[$step] node ${SLURM_NODEID:-0}/${SLURM_NNODES:-1} on $(hostname), work $WORK, out $OUT"
R="python scripts/flag_retok/retok.py"
case $step in
selftest) python scripts/flag_retok/test_retok.py ;;
find)     $R find --root "$ROOT" --work "$WORK" --shards "$WORK/shards_all.txt" "$@" ;;
write)    $R write --root "$ROOT" --work "$WORK" --out "$OUT" "$@" ;;
decon)    $R decon --root "$ROOT" --work "$WORK" --out "$OUT" "$@" ;;
verify)   $R verify --root "$ROOT" --work "$WORK" --out "$OUT" "$@" ;;
*) echo "unknown step $step" >&2; exit 2 ;;
esac
echo "[$step] done"
