#!/bin/bash

# source activate_conda.sh
export PROJECT_DIR=/gpfs/projects/ehpc533/oellm-autoexp-joan
export OUTPUT_DIR=/gpfs/projects/ehpc533/oellm-autoexp-joan/outputs
export DATA_DIR=/gpfs/projects/ehpc533/data
export HF_HOME=/gpfs/projects/ehpc533/hf-cache
export CONTAINER_CACHE_DIR=/gpfs/projects/ehpc533/singularity_images/
export OELLM_TS=$(date +%Y%m%d_%H%M%S)



export SLURM_ACCOUNT=ehpc533
export SLURM_QOS=acc_ehpc
