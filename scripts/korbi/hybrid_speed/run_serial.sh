#!/bin/bash
# Run tasks of a rendered speed-suite array script one after another in ONE allocation --
# for queues without job arrays and with one job per user, e.g. MareNostrum's acc_debug:
#
#   sbatch -q acc_debug -A ehpc533 --time 02:00:00 --nodes 1 --gres gpu:4 --exclusive \
#       -o <group dir>/serial-%j.out scripts/korbi/hybrid_speed/run_serial.sh <group dir>/<group>.sbatch 0 3 7
#
# Each task's output goes to <group dir>/slurm-<jobid>_<task>.log, where speed_table.py
# picks it up like an array task's log (the newest attempt of a task wins). Run from the
# synced checkout (the array script's commands are relative to it).
set -u
script=$1; shift
dir=$(dirname "$script")
for task in "$@"; do
    echo "[serial] task $task -> $dir/slurm-${SLURM_JOB_ID}_${task}.log"
    SLURM_ARRAY_TASK_ID=$task bash "$script" > "$dir/slurm-${SLURM_JOB_ID}_${task}.log" 2>&1
    echo "[serial] task $task exit $?"
done
