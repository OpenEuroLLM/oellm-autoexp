#!/bin/bash

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
