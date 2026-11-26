#!/bin/bash
# Run a command inside the OpenEuroLLM environment: David's NGC PyTorch 26.09 image and his pip
# prefix (both read-only), our checkouts first on PYTHONPATH, caches on node-local disk.
#   run_in_env.sh <command ...>     (cwd: /fast/pnazari/ldm-work)
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/fast/pnazari/ldm-work
DW=/fast/dmartinez/oellm-ldm-work
SIF=$DW/images/pytorch_26.09-py3.sif
CUTE_DIR=${CUTE_DIR:-$W/cute_kernels}
MEGATRON_DIR=${MEGATRON_DIR:-$W/Megatron-LM}
PYPATH=$CUTE_DIR:$MEGATRON_DIR:$W/flash-linear-attention:$W/pylibs
SCR=${_CONDOR_SCRATCH_DIR:-/tmp/$USER-oellm}
C=$SCR/cache
mkdir -p $C
ENVS=PYTHONUSERBASE=$DW/pyuser,PIP_CACHE_DIR=$W/cache/pip,CUDA_CACHE_PATH=$C/nv,PYTHONPYCACHEPREFIX=$C/pyc
ENVS=$ENVS,TRITON_CACHE_DIR=$C/triton,TORCHINDUCTOR_CACHE_DIR=$C/inductor
ENVS=$ENVS,XDG_CACHE_HOME=$C/xdg,HF_HOME=$W/cache/hf,CUTE_DSL_CACHE_DIR=$C/cute
ENVS=$ENVS,TRITON_LIBCUDA_PATH=/.singularity.d/libs
ENVS=$ENVS,CUTE_DIR=$CUTE_DIR,PYTHONPATH=$PYPATH,PYTHONUNBUFFERED=1,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVS=$ENVS${EXTRA_ENV:+,$EXTRA_ENV}${CUDA_VISIBLE_DEVICES:+,CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
cd "$W"
exec apptainer exec --nv --cleanenv --home $W/home --bind /fast --bind $SCR --env "$ENVS" "$SIF" bash -c '
python3 -c "import torch, cute_ldm; print(\"torch\", torch.__version__, \"cute_ldm from\", cute_ldm.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
'"$*"
