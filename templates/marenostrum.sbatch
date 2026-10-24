#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching job {job_name}"
srun {srun_opts} bash -c '{launcher_cmd} {command}'
# keep the training step's exit status: the echo lines below would otherwise make every job
# COMPLETED 0:0, so a crashed or OOM-killed step was closed as finished and never retried
RC=$?

echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"
exit $RC
