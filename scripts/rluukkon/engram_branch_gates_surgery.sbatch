#!/bin/bash
#SBATCH --job-name=engram-branchgates-surgery
#SBATCH --account=ehpc533
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# The warm-start checkpoints of experiments/rluukkon/nucleus_engram_branch_gates_mn5 from v4b's save at STEP
# (default 44000, its ckpt_eval copy): bg (per-stream gates, for the read / stream arms) and bg_qmix
# (+ gate_terms.query_mix, for the learned arm). No ctrl copy: the control is v4b itself.
# Torch only (no GPU work); ~2 min per checkpoint at production size (the 228 MB metadata twice, 78 small
# tensors), the 1.1 TB of .distcp files hard-linked: run as the owner of the source files, on the same file
# system. Run from the repo root (account / QoS: sbatch -A ... -q ... on the command line).
set -eu
STEP=${STEP:-44000}
it=iter_$(printf %07d "$STEP")
KV=/gpfs/scratch/ehpc533/hybrid_nucleus/production/kvshare_k_fresh35900_20261116
SRC=${SRC:-$KV/runs/v4b_learned/main/ckpt_eval/$it}
OUT=${OUT:-$KV/engram_branch_gates/surgery}
SIF=${SIF:-/gpfs/projects/ehpc533/users/korbip/hybrid_v019/container/MegatronTraining-hybrid-ep_x86_64_202610161948.sif}
[ -f "$SRC/.metadata" ] || { echo "no checkpoint at $SRC"; exit 1; }
command -v singularity >/dev/null || module load singularity >/dev/null 2>&1 || true
py() { singularity exec --bind /gpfs --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= "$SIF" python3 "$@"; }
for v in bg bg_qmix; do
    dst=$OUT/$v/ckpt/$(basename "$SRC")
    if [ -f "$OUT/$v/ckpt/latest_checkpointed_iteration.txt" ]; then echo "$dst exists: skipped"; continue; fi
    extra=(); [ "$v" = bg_qmix ] && extra=(--query-mix)
    py scripts/korbi/production/add_engram_branch_gates.py --src "$SRC" --dst "$dst" "${extra[@]}"
done
