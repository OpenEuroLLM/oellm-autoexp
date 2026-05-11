#!/bin/bash
#SBATCH --job-name=orch_0.1B_center
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.log
#SBATCH --account=ehpc533
#SBATCH --qos=acc_ehpc

module load intel impi mkl hdf5 python/3.11.5-gcc
export PYTHONPATH=/gpfs/projects/ehpc533/environments/oellm-autoexp_mn5_python3.11_20260408/lib/python3.11/site-packages
source /gpfs/projects/ehpc533/environments/oellm-autoexp_mn5_python3.11_20260408/bin/activate

cd /gpfs/projects/ehpc533/users/slaing00/oellm-autoexp

PYTHONPATH=. python scripts/run_autoexp.py \
    --config-name experiments/slaing00/multilingual_scaling/0.1B_ne \
    backend.megatron.aux.priority_tier=center
