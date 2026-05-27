#!/bin/bash
set -euo pipefail
CONTAINER_IMAGE="/shared_silo/scratch/containers/build-rocm_primus_v25.11_transformers-5.5.4_linear_FA/rocm_primus_v25.11_transformers-5.5.4_linear_FA.sif"
MEGATRON_LM_PATH=/shared_silo/scratch/rluukkon/oellm/oellm-autoexp/submodules/Megatron-LM
export PYTHONPATH=$MEGATRON_LM_PATH
# create overlay if it doesn't exist
if [ ! -f autoexp-update.img ]; then
    singularity overlay create --size 8192 autoexp-update.img
    echo "Overlay created"
    # install dependencies
    singularity exec -B /shared_silo/ --overlay autoexp-update.img $CONTAINER_IMAGE python -m pip install -e .
    echo "Overlay created and dependencies installed"
fi

singularity exec -B /shared_silo/ \
    --overlay autoexp-update.img \
    $CONTAINER_IMAGE python scripts/generate_megatron_config.py

singularity exec -B /shared_silo/ \
    --overlay autoexp-update.img \
    $CONTAINER_IMAGE python scripts/generate_megatron_dataclass.py

