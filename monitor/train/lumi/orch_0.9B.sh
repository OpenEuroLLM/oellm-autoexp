#!/bin/bash
#SBATCH --account=project_465002530
#SBATCH --nodes=1
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=orch_0.9B
#SBATCH --output=/scratch/project_465002530/users/dianaonutu/oellm-autoexp/monitor/train/lumi/logs/slurm-%x-%j.out

cd /scratch/project_465002530/users/dianaonutu/oellm-autoexp

uv run --python 3.12 python scripts/monitor_autoexp.py --session-dir /scratch/project_465002530/users/dianaonutu/oellm-autoexp/monitor_state/1784646185