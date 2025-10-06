#!/bin/bash
{sbatch_directives}

{env_exports}

ml Stages/2025
ml GCC/13.3.0
ml Python/3.12.3
ml CUDA/12
ml cuDNN/9.5.0.50-CUDA-12
ml NCCL/default-CUDA-12
ml OpenMPI/5.0.5
ml ParaStationMPI/5.11.0-1
ml ParaStationMPI/5.11.0-1-mt
ml ScaLAPACK/2.2.0-fb

echo "Launching {name} on JUWELS"
srun {srun_opts}{launcher_prefix}{launcher_env_flags}{backend_cmd}
