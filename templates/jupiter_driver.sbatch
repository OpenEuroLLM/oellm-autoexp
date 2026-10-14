#!/bin/bash
{sbatch_directives}

{env_exports}

module purge
ulimit -c 0

# Driver template: the command runs HERE, in the batch shell on the first node -- not under an
# outer srun -- so it can place its own `srun` job steps on the allocation's nodes (compute jobs
# cannot sbatch). Used by config/slurm/jupiter_driver.yaml (e.g. the convert stage).
echo "Launching {job_name} on JUPITER (driver)"
bash -c '{launcher_cmd} {command}'
