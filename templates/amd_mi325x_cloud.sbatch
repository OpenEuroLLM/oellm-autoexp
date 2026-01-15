#!/bin/bash
{sbatch_directives}

{env_exports}

echo "Launching {name} on AMD_MI325X_CLOUD_TENSORWAVE"
srun {srun_opts}bash -c '{launcher_cmd}'
