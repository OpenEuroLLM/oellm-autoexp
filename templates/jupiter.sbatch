#!/bin/bash
{sbatch_directives}

{env_exports}

module load Stages/2024
module load GCC/12.3.0
module load CUDA/12.1
module load NCCL/2.20.5

echo "Launching {name} on JUPITER"
srun {srun_opts}{launcher_cmd}
