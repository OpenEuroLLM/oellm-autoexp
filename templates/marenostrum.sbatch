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

echo "Launching job {job_name}"
srun {srun_opts} bash -c '{launcher_cmd} {command}'
# keep the training step's exit status: the echo lines below would otherwise make every job
# COMPLETED 0:0, so a crashed or OOM-killed step was closed as finished and never retried
RC=$?

echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"
exit $RC
