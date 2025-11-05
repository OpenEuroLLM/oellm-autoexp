#!/bin/bash
{sbatch_directives}

{env_exports}

module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3
module load CUDA/12

echo "Launching {name} on JUPITER"
srun {srun_opts}bash -c '{launcher_cmd}'
