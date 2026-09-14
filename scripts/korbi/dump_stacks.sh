#!/usr/bin/env bash
# Make a hung Megatron job print every rank's Python stack into its own log.
#
#   scripts/korbi/dump_stacks.sh <jobid> [node ...]
#
# With no node argument it signals ONE node (the first in the allocation).
# Pass `all` to signal every node, or explicit hostnames to pick stages.
#
# WHY THIS EXISTS
# ---------------
# A silent hang is the one failure this tree cannot debug after the fact: no
# traceback, no NCCL warning, the log just stops (the 32B v2 flagship at
# iteration 40000, seven times). `py-spy` does NOT work here — the ranks run
# inside an apptainer mount namespace, and a py-spy started in a second
# instance of the same image cannot resolve the target's binaries:
#
#     Error: Failed to find python version from target process
#
# So `megatron/training/training.py::install_stack_dumper` registers
# faulthandler on SIGUSR1 in every rank at the top of `pretrain()`. The process
# then dumps its OWN stack, from inside its own namespace, to stderr — which
# lands in the job log.
#
# THE TWO WAYS TO GET THIS WRONG, both of which KILL THE JOB
# ----------------------------------------------------------
# Naive `pkill -USR1 -f pretrain_gpt.py` matches THREE kinds of process:
#
#   1. the pkill command itself (its own argv contains the pattern) — the
#      classic self-match, which is why the pattern below is "[p]retrain";
#   2. the torchrun AGENT (`python -m torch.distributed.run ... pretrain_gpt.py
#      ...`). The agent never calls `pretrain()`, so it has NO handler, and
#      SIGUSR1's default action is to terminate. Killing the agent tears down
#      every worker on the node;
#   3. the worker ranks — the only ones we actually want.
#
# Measured 2026-09-15 on job 1798490: signalling (1)+(2)+(3) killed all four
# tasks with "User defined signal 1" and produced zero stack dumps. This script
# signals (3) only, by skipping any process whose cmdline names the launcher.
#
# SAFETY: only ever signals processes matching the worker pattern, and prints
# what it will signal before doing it. The handler must already be installed,
# i.e. the job must be PAST `pretrain()` — signalling during startup/import
# still kills the ranks. In practice a job that is hung mid-training is always
# past it.
set -uo pipefail

JOBID=${1:?usage: dump_stacks.sh <jobid> [node ...|all]}
shift || true

mapfile -t ALL_NODES < <(scontrol show hostnames "$(squeue -j "$JOBID" -h -o '%N')")
if [ $# -eq 0 ]; then
  NODES=("${ALL_NODES[0]}")
elif [ "$1" = "all" ]; then
  NODES=("${ALL_NODES[@]}")
else
  NODES=("$@")
fi

echo "job $JOBID: signalling ${#NODES[@]} of ${#ALL_NODES[@]} node(s): ${NODES[*]}"

for node in "${NODES[@]}"; do
  srun --jobid="$JOBID" --overlap -w "$node" --ntasks=1 --cpu-bind=none bash -c '
    signalled=0
    for p in $(pgrep -f "[p]retrain_gpt.py"); do
      cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null) || continue
      # Skip the torchrun agent: it has no SIGUSR1 handler and killing it
      # takes every worker on this node down with it.
      case "$cmd" in
        *torch.distributed.run*|*torch/distributed/run.py*|*torchrun*) continue ;;
      esac
      kill -USR1 "$p" && signalled=$((signalled + 1))
    done
    echo "$(hostname): signalled $signalled worker rank(s)"
  '
done

echo
echo "Stacks go to the job log. Read them with:"
echo "  grep -a -A40 'Stack (most recent call first)' <logfile>"
