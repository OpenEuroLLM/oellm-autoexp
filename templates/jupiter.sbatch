#!/bin/bash
{sbatch_directives}

{env_exports}

# ml Stages/2026
# ml GCC/14.3.0
# ml Python/3.13.5
# ml CUDA/13
# ml NCCL/default-CUDA-13
module purge

# Disable core dumps. core_pattern is relative (core.%h.%p), so a crashing rank
# would otherwise write a multi-GB memory dump into the job CWD — the repo on the
# quota-limited project filesystem — silently filling the quota. Set at both the
# batch shell and inside the srun/container shell so every rank inherits it.
ulimit -c 0

echo "Launching {job_name} on JUPITER"
srun {srun_opts}bash -c 'ulimit -c 0; {launcher_cmd} {command}'
