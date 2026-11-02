#!/bin/bash
{sbatch_directives}

{env_exports}

ulimit -c 0

echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"
echo "[SLURM] MASTER:   $MASTER_ADDR:$MASTER_PORT"

echo "Launching {job_name} on CLARIDEN"
srun {srun_opts}bash -c 'ulimit -c 0; {launcher_cmd} {command}'
