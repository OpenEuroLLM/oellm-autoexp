#!/bin/bash
{sbatch_directives}

{env_exports}

# WAIT_FOR_JOB_NAME (optional): start only once no job of that name runs any more -- a run that
# takes over another one's checkpoints (e.g. 32 -> 128 nodes) must not load before the other's
# final save is written
if [ -n "$WAIT_FOR_JOB_NAME" ]; then
    while squeue -u "$USER" -h -n "$WAIT_FOR_JOB_NAME" -t R,CG -o %i | grep -q .; do
        echo "waiting for $WAIT_FOR_JOB_NAME to end"; sleep 30
    done
fi

# --- Watchdog: one per node, on the host, next to the training step ---------------
# GPU memory trace + hang kill (no iteration for WATCHDOG_STALL_S, or none within
# WATCHDOG_STARTUP_S): py-spy stacks of every rank, then scancel.
# scripts/korbi/hybrid_speed/job_watchdog.py; both steps use --overlap.
echo "Launching watchdog on $SLURM_NNODES node(s)"
srun --ntasks-per-node=1 --nodes=$SLURM_NNODES --overlap --cpu-bind=none python3 $WATCHDOG_SCRIPT &
WATCHDOG_PID=$!
# -------------------------------------------------------------------------------

echo "Launching job {job_name}"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
RC=$?

kill $WATCHDOG_PID 2>/dev/null; wait $WATCHDOG_PID 2>/dev/null || true
echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"
exit $RC
