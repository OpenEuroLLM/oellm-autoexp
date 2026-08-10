#!/bin/bash
#SBATCH --job-name=muon_smoke
#SBATCH --partition=capella-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:h100_1g.12gb:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/01_optimizer_construct_%j.log

# RUNG 1 of NEW_CLUSTER_SETUP.md's smoke ladder: Newton-Schulz on GPU +
# HAVE_EMERGING_OPTIMIZERS check. See RESULT.md for context.

set -eo pipefail
cd /home/h1/sala597i/oellm-autoexp

SANDBOX=/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox

singularity exec --underlay --nv \
  --env PYTHONPATH=$PWD:$PWD/muon-pylibs:submodules/Megatron-LM \
  "$SANDBOX" python3 -u muon_smoke/muon_gpu_smoke.py
