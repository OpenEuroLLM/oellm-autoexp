#!/bin/bash
#SBATCH --job-name=mhc-to-ldm-surgery
#SBATCH --account=ehpc533
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# The warm-start checkpoints of experiments/pnazari/nucleus_ldm_replacement_mn5 from v4b's save at STEP (default
# 44000, its ckpt_eval copy): collapse (mHC removed) and ldm (mHC removed + LDM sites with read gain 0). No ctrl copy:
# the control is v4b itself. Torch only (no GPU work); a few minutes per checkpoint at production size (the metadata
# twice, the V layers' linear_proj and the new LDM tensors), the 1.1 TB of .distcp files hard-linked: run as the
# owner of the source files, on the same file system. Run from the repo root (account / QoS: sbatch -A ... -q ...).
set -eu
STEP=${STEP:-44000}
it=iter_$(printf %07d "$STEP")
KV=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116
SRC=${SRC:-$KV/runs/v4b_learned/main/ckpt_eval/$it}
OUT=${OUT:-$KV/ldm_replacement/surgery}
SIF=${SIF:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
[ -f "$SRC/.metadata" ] || { echo "no checkpoint at $SRC"; exit 1; }
command -v singularity >/dev/null || module load singularity >/dev/null 2>&1 || true
py() { singularity exec --bind /gpfs --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= "$SIF" python3 "$@"; }
for v in collapse ldm; do
    dst=$OUT/$v/ckpt/$(basename "$SRC")
    if [ -f "$OUT/$v/ckpt/latest_checkpointed_iteration.txt" ]; then echo "$dst exists: skipped"; continue; fi
    extra=(); [ "$v" = ldm ] && extra=(--ldm)
    py scripts/korbi/production/mhc_to_ldm.py --src "$SRC" --dst "$dst" "${extra[@]}"
done
