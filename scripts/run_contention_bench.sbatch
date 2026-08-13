#!/bin/bash
# GPFS contention benchmark: reproduce the "many workers random-faulting one huge GPFS file"
# throughput breakdown, and show the node-local mirror (bounded working set) fixes it.
#
# Three arms, each swept over procs-per-node P in {1,8,32,64} (= 8..512 concurrent readers on 8 nodes):
#   direct-global : random reads over the FULL 1.2 TB merged.bin on GPFS   (today's global-shuffle behavior)
#   direct-window : random reads over a small window of merged.bin on GPFS (windowed-shuffle locality, still GPFS)
#   mirror-window : random reads over a node-local staged copy of that window (the full node-local solution)
#
# Submit via scripts/run_contention_bench.sh (sets OELLM_REPO + output dir).
#SBATCH --job-name=gpfs_contention
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=00:30:00
set -x

FILE=/gpfs/scratch/ehpc390/data/cerebras-SlimPajama-627B/train/merged.bin
WINDOW_GB=${WINDOW_GB:-80}
READ_BYTES=${READ_BYTES:-16384}
DUR=${DUR:-20}
BENCH="$OELLM_REPO/scripts/gpfs_contention_bench.py"
DIR=/scratch/tmp/$SLURM_JOB_ID
LOCAL=$DIR/win.bin   # identical node-local path on every node

echo "[bench] nodes=$SLURM_NNODES window=${WINDOW_GB}G read_bytes=$READ_BYTES dur=${DUR}s"

# --- stage a WINDOW_GB local window per node (one sequential GPFS read -> local NVMe) ---
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 bash -c \
  "mkdir -p $DIR && dd if=$FILE of=$LOCAL bs=16M count=$((WINDOW_GB*64)) 2>/dev/null && echo [staged] \$(hostname) \$(du -h $LOCAL | cut -f1)"

# --- concurrency scan x 3 arms ---
for P in 1 8 32 64; do
  for ARM in direct-global direct-window mirror-window; do
    case $ARM in
      direct-global) PATHV=$FILE;  SPAN=0 ;;
      direct-window) PATHV=$FILE;  SPAN=$WINDOW_GB ;;
      mirror-window) PATHV=$LOCAL; SPAN=0 ;;
    esac
    echo "===== STEP arm=$ARM procs_per_node=$P total_readers=$((P*SLURM_NNODES)) ====="
    srun --nodes="$SLURM_NNODES" --ntasks-per-node="$P" --cpus-per-task=1 \
      python3 "$BENCH" --path "$PATHV" --span-gb "$SPAN" --read-bytes "$READ_BYTES" \
        --duration "$DUR" --arm "$ARM" --procs "$P"
  done
done
echo "[bench] done"
