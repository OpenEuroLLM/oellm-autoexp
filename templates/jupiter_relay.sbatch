#!/bin/bash
{sbatch_directives}

{env_exports}

module purge
ulimit -c 0

# Worker-only signal relay (2026-09-06). With `#SBATCH --signal=B:USR1@<margin>` SLURM signals this batch
# shell only, with SIGUSR1 (a batch-shell SIGTERM makes slurmstepd print the '*** JOB N ... CANCELLED AT ***'
# line that the policy reads as a scancel; USR1 prints nothing). The trap forwards SIGTERM to the Megatron
# rank processes on every node through an --overlap step: the direct children of torchrun's elastic agents,
# not the agents themselves (an agent that receives SIGTERM tears its workers down ~30 s later, before their
# exit checkpoint completes; 4-node smoke 1691088) and not the ranks' forked dataloader / async-save
# children, which share the ranks' command line (smoke 1691431 hit 72 processes per node with a pattern).
# The agent pattern anchors on the python executable (`python -u -m torch.distributed.run ...`): the srun
# task's bash and apptainer wrappers carry the same string in their command lines and must not be hit.
# Megatron's --exit-signal-handler then saves and exits 0, the agents exit 0, the job ends COMPLETED and the
# policy's sigterm_rollover resubmits. bash runs a trap only while it is not blocked in a foreground child,
# hence the background srun and the wait loop (wait is interrupted by the signal, the trap runs, wait resumes).
relay_workers() {{
  echo "[relay] $(date +%FT%T) signal for the batch shell: forwarding SIGTERM to the rank processes on $SLURM_JOB_NUM_NODES node(s)"
  srun --overlap --jobid="$SLURM_JOB_ID" --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'n=0; for a in $(pgrep -f "^[^ ]*python[^ ]*( +-[A-Za-z]+)* +-m +torch.distributed.run"); do for w in $(pgrep -P "$a"); do kill -TERM "$w" && n=$((n+1)); done; done; echo "[relay] $(hostname): SIGTERM sent to $n rank process(es)"' 2>&1
  echo "[relay] $(date +%FT%T) relay step finished"
}}
trap relay_workers USR1 TERM

echo "Launching {job_name} on JUPITER (worker-only signal relay armed)"
srun {srun_opts}bash -c '{launcher_cmd} {command}' &
main_pid=$!
rc=0
while kill -0 "$main_pid" 2>/dev/null; do
  wait "$main_pid"; rc=$?
done
echo "[relay] main step ended with exit code $rc"
exit "$rc"
