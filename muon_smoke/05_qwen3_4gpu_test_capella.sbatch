#!/bin/bash
#SBATCH --job-name=muon_qwen3_4gpu_test
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:20:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/05_qwen3_4gpu_step_%j.log

# First multi-GPU attempt — single node, 4 full H100s. Same model/data/Muon
# config as the passing 1-GPU run (04_qwen3_real_data_test_capella.sbatch,
# see RESULT_qwen3.md), except /backend: megatron_torchrun (actually launches
# via torchrun, nproc_per_node driven by slurm.sbatch.gpus_per_node=4 — see
# muon_qwen3_0.1B_longctx_4gpu.yaml's comments) instead of plain python.
# Goal: just confirm training runs at all at dp_cp=4 — Muon's
# LayerWiseDistributedOptimizer has never been exercised beyond dp_cp=1 on
# this branch. No checkpoint save exercised here (kept out of scope).

set -eo pipefail
cd /home/h1/sala597i/oellm-autoexp

SANDBOX=/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox

singularity exec --underlay --nv -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:$PWD/muon-pylibs:submodules/Megatron-LM \
  --env TORCHDYNAMO_DISABLE=1 \
  "$SANDBOX" python3 -u scripts/run_autoexp.py \
    --config-name experiments/laingsam/muon_qwen3_0.1B_longctx_4gpu \
    '~backend.megatron.norm_epsilon' \
    backend.megatron.eval_iters=0 \
    --array-subset 0 \
    --local
