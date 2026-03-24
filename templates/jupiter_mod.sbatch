#!/bin/bash
{sbatch_directives}

{env_exports}

ml Stages/2026
ml GCC/14.3.0
ml Python/3.13.5
ml CUDA/13
ml NCCL/default-CUDA-13

echo "Launching {job_name} on JUPITER"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
