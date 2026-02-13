#!/bin/bash
{sbatch_directives}

{env_exports}

module purge

echo "Launching {job_name} on JUPITER"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
