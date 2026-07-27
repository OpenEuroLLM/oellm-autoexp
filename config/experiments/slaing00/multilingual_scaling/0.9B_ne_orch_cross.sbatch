#!/bin/bash
#SBATCH --job-name=orch_0.9B_ne_cross
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.log
#SBATCH --account=oellm_prod2026
#SBATCH --qos=normal

export SINGULARITY_BIND="/leonardo_work,/leonardo,/leonardo_scratch,/usr/bin/sbatch,/usr/bin/squeue,/usr/bin/scancel,/usr/bin/scontrol,/usr/lib64/slurm,/usr/lib64/libmunge.so.2,/usr/lib64/libmunge.so.2.0.1,/run/slurm,/run/munge,/etc/slurm,/var/spool/slurmd,/etc/passwd,/etc/group"

singularity exec /leonardo_work/OELLM_prod2026/container_images/OELLM_autoexp_MegatronTrainingNoRoot_base_2510_x86_64_202603060943.sif \
    bash -c "cd /leonardo_work/OELLM_prod2026/slaing00/oellm-autoexp && PYTHONPATH=. python3 scripts/run_autoexp.py \
        --config-name experiments/slaing00/multilingual_scaling/0.9B_ne \
        backend.megatron.aux.priority_tier=cross"
