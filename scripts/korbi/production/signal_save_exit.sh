#!/bin/bash
# Ask a running Megatron job to save a checkpoint and exit (exit_signal_handler, exit_signal
# SIGUSR2): SIGUSR2 to the training ranks only, i.e. the direct children of each node's torchrun.
# `scancel --signal=USR2 --full` also hits torchrun (default action: terminate), which killed
# job 47974985 without a save; pgrep -f on a pattern the calling shell carries matches the
# shell itself, hence the [b]racket pattern.
#   scripts/korbi/production/signal_save_exit.sh <jobid>
set -u
J=$1
NN=$(squeue -h -j "$J" -o %D)
[ -n "$NN" ] || { echo "job $J not running"; exit 1; }
srun --jobid="$J" --overlap -N"$NN" --ntasks-per-node=1 bash -c '
    n=0
    for t in $(pgrep -f "[b]in/torchrun"); do
        for c in $(pgrep -P "$t"); do kill -USR2 "$c" && n=$((n + 1)); done
    done
    echo "$(hostname -s): SIGUSR2 to $n ranks"'
