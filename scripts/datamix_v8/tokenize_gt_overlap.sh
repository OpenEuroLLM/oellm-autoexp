#!/bin/bash
# datamix v8: run tokenize_gt.py on the spare nodes of a running JUPITER pool host (the 32B pool's
# 516-node allocation runs its training on 512), as --overlap --mem=0 steps started from a login node:
# - --overlap --mem=0: the run's own --overlap helper steps (signal relay before the wall clock, the
#   supervisor's graceful stop) can still start on these nodes; no GPUs requested;
# - a guard kills our step as soon as the pool claims one of our nodes (hosts/<h>/inuse/*, e.g. a run
#   restarted after a node failure) or the host job is gone; tokenize_gt.py resumes from finished units.
#   tokenize_gt_overlap.sh JOBID HOSTDIR NODELIST SOURCE [SOURCE ...]
set -u
JOB=$1 H=$2 NODES=$3
shift 3
V=/e/scratch/e-ext-2025e02-108/poeppel1/v8
PY=$V/gtenv/bin/python3
N=$(scontrol show hostnames "$NODES" | grep -c .)
log() { echo "[v8tok] $(date +%FT%T) $*"; }

guard_ok() {
  [ "$(squeue -h -j "$JOB" -o %T 2>/dev/null)" = RUNNING ] || { log "host job $JOB not running"; return 1; }
  local held
  held=$(cat "$H"/inuse/* 2>/dev/null | tr ',' '\n' | grep -xF -f <(scontrol show hostnames "$NODES"))
  [ -z "$held" ] || { log "pool claimed our node(s): $held"; return 1; }
}

step() { # step <mode> <source>
  guard_ok || exit 3
  srun --jobid="$JOB" --overlap --mem=0 --nodelist="$NODES" --nodes="$N" --ntasks="$N" --ntasks-per-node=1 \
    --cpus-per-task=288 --kill-on-bad-exit=1 --job-name="v8tok-$1" \
    bash -c "cd $V && $PY tokenize_gt.py $1 --source $2 --part \$SLURM_PROCID --nparts \$SLURM_NTASKS --procs 280" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    if kill -0 "$pid" 2>/dev/null && ! guard_ok; then
      log "aborting $1 $2"
      kill -TERM "$pid"; sleep 2; kill -TERM "$pid" 2>/dev/null
      wait "$pid"
      exit 3
    fi
  done
  wait "$pid" || { log "$1 $2 failed"; exit 1; }
}

for s in "$@"; do
  log "encode $s"
  step encode "$s"
done
for s in "$@"; do
  log "assemble $s"
  step assemble "$s"
done
log "DONE"
