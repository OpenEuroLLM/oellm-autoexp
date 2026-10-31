#!/bin/bash
# datamix v8: GigaToken tokenization on JUPITER (CPU only; booster nodes, 288 Grace cores each).
#   sbatch --nodes 8 tokenize_gt.sbatch SOURCE [SOURCE ...]
# encode all sources (units split over the nodes), then assemble the shards; resumable (finished units / shards skipped).
#SBATCH --job-name=v8_tokenize
#SBATCH --account=e-sta-openeurollm
#SBATCH --partition=booster
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/e/scratch/e-ext-2025e02-108/poeppel1/v8/logs/%x-%j.out

V=/e/scratch/e-ext-2025e02-108/poeppel1/v8
PY=$V/gtenv/bin/python3
cd $V
set -e
for s in "$@"; do
    echo "=== encode $s $(date +%T)"
    srun --kill-on-bad-exit=1 bash -c "$PY tokenize_gt.py encode --source $s --part \$SLURM_NODEID --nparts \$SLURM_NNODES"
done
for s in "$@"; do
    echo "=== assemble $s $(date +%T)"
    srun --kill-on-bad-exit=1 bash -c "$PY tokenize_gt.py assemble --source $s --part \$SLURM_NODEID --nparts \$SLURM_NNODES"
done
echo "=== DONE $(date +%T)"
