#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching {job_name} on MI325X (Apptainer/ROCm)"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
