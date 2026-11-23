#!/bin/bash
#SBATCH --account=a0248
#SBATCH --job-name=export_100m_hf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/store/cscs/swissai/a0248/users/swagatam/oellm-autoexp/config/experiments/swagatam/alps/runs/export_100m_hf-%j.log
#SBATCH --error=/capstor/store/cscs/swissai/a0248/users/swagatam/oellm-autoexp/config/experiments/swagatam/alps/runs/export_100m_hf-%j.log

# 11 runs x 5 checkpoints (1000, 2000, 3000, 4000, 4819):
#   two-bank engram_100m_1b_E{40,56,68,76,88,96}  (rho 40 through 100)
#   consolidated engram_100m_1b_cons_E{40,56,68,76,88}  (no rho 100)
# Each checkpoint is exported, then check_hf_export and megatron_parity.
# A checkpoint whose check log and parity JSON already passed is skipped.

ROOT=/capstor/store/cscs/swissai/a0248/users/swagatam/oellm-autoexp
srun --environment="$ROOT/container/alps/nemo-26.08.toml" --cpus-per-task=72 \
  bash "$ROOT/config/experiments/swagatam/alps/runs/export_100m_hf.sh"
