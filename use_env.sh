#!/bin/bash

module load intel impi mkl hdf5 python/3.11.5-gcc

export PYTHONPATH=/gpfs/projects/ehpc533/environments/oellm-autoexp_mn5_python3.11_20260408/lib/python3.11/site-packages

# Env
source /gpfs/projects/ehpc533/environments/oellm-autoexp_mn5_python3.11_20260408/bin/activate

# variable
source /gpfs/projects/ehpc533/oellm-autoexp-joan/oellm_env.sh
