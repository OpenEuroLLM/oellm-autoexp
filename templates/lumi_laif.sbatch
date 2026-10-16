#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching {job_name} on LUMI (LAIF runtime)"
srun --kill-on-bad-exit=1 {srun_opts} bash -c '{launcher_cmd} {command}'
