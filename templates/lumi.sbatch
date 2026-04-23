#!/bin/bash
{sbatch_directives}

{env_exports}
ml purge

echo "Launching {job_name} on LUMI"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
