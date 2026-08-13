#!/bin/bash
{sbatch_directives}

{env_exports}

# --- Phase 1: stage the dataset prefix to node-local storage on every node ----
# Runs once per node (one task/node) on the host, before the training srun.
# Driven entirely by STAGE_* env vars exported above (set in the slurm config).
echo "Staging dataset to node-local storage on $SLURM_NNODES node(s)"
srun --ntasks-per-node=1 --nodes=$SLURM_NNODES --cpu-bind=none bash $STAGE_SCRIPT
# -----------------------------------------------------------------------------

echo "Launching {job_name} on MARENOSTRUM (node-local data)"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
