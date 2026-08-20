#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching job {job_name}"
srun {srun_opts} bash -c '{launcher_cmd} {command}'

echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"
