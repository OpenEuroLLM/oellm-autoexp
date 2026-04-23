#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching job {job_name}"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
