#!/bin/bash
# Submit the GPFS contention benchmark (scripts/run_contention_bench.sbatch). Run ON MareNostrum
# from the repo root. Prints the job id; aggregate with parse_contention_bench.py when it finishes.
#   bash scripts/run_contention_bench.sh
set -e
cd "$(dirname "$0")/.."
export OELLM_REPO="$PWD"
OUT=/gpfs/projects/ehpc390/outputs/gpfs_contention
mkdir -p "$OUT"
JID=$(sbatch --parsable --account="${ACCOUNT:-ehpc390}" --export=ALL --output="$OUT/slurm-%j.log" scripts/run_contention_bench.sbatch)
echo "submitted job $JID"
echo "log: $OUT/slurm-$JID.log"
echo "aggregate when done: python3 scripts/parse_contention_bench.py $OUT/slurm-$JID.log"
