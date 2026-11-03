#!/bin/bash
set -euo pipefail
R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019
S=$R/autoexp_mhc256_009
POOL=/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264
NODE=as07r5b16
# Fail rather than contend with an assigned production or pool-test node.
if cat "$POOL"/hosts/host001/inuse/* | tr ',' '\n' | grep -qx "$NODE"; then
  echo "Node $NODE is assigned; do not launch" >&2
  exit 1
fi
cd "$S"
srun --overlap --mem=0 --jobid=48203953 --nodelist="$NODE" --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-node=4 --job-name=mhc009_unit \
  singularity exec --nv --bind /gpfs \
  --env PYTHONPATH="$R/pylibs_moonep_ec3eb8b:.:submodules/Megatron-LM:flash-linear-attention:submodules/cute_kernels:$R/retention:$R/pylibs_ep" \
  --env OMP_NUM_THREADS=4 --env CUDA_DEVICE_MAX_CONNECTIONS=8 \
  "$R/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif" \
  torchrun --standalone --nproc-per-node=4 experiment/009_mhc256_sharding/scripts/check_hsdp_muown.py
