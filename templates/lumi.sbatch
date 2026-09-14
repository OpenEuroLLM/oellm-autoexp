#!/bin/bash
{sbatch_directives}

{env_exports}
ml purge

echo "Probing GPU and RCCL health on LUMI"
if ! timeout 360s srun --kill-on-bad-exit=1 {srun_opts}bash -c \
  '{launcher_cmd} python scripts/probe_distributed_health.py --timeout-seconds 300 --large-reduce-scatter-mib 4096'; then
    echo "GPU/RCCL health probe failed; aborting job" >&2
    exit 1
fi

echo "Launching {job_name} on LUMI"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
