#!/bin/bash

export OC_CAUSE=1
export SINGULARITY_BIND=/boot/config-6.4.0-150600.23.73_15.0.14-cray_shasta_c,/scratch/project_462000963/,/projappl/project_462000963/,/flash/project_462000963/,/var/spool/slurmd
export SINGULARITY_BIND="${SINGULARITY_BIND},/opt/rocm/lib/librccl.so:/opt/venv/lib/python3.12/site-packages/torch/lib/librccl.so"
export SLURM_ACCOUNT="project_462000963"
export DATA_ACCOUNT="project_462000963"
export SLURM_PARTITION="dev-g"
export SLURM_PARTITION_DEBUG="dev-g"
export SLURM_QOS="normal"
export SLURM_QOS_DEBUG="normal"
export LUMI_SCRATCH="/project_462000963/"
export WORK="/scratch/$DATA_ACCOUNT/users/$USER"
export DATA_DIR="/scratch/project_462000963/preprocessed"
export OUTPUT_DIR="$WORK/output"
export APPTAINER_CACHEDIR="/scratch/project_462000963/containers"
export APPTAINER_TMPDIR="$WORK/.tmp"
export CONTAINER_CACHE_DIR="/scratch/project_462000963/containers"
export ARCH="$(uname -m)"