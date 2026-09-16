#!/bin/bash
#SBATCH --job-name=muon_throughput_4gpu
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:20:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/07_throughput_4gpu_%j.log

# One clean 4-GPU throughput point, matched exactly against the 1-GPU
# measurement (0.1B, micro_batch_size=4, train_iters=80, job 3876269-style)
# so the two are directly comparable — gives a real, measured DP
# scaling-efficiency ratio (98,495 tok/s/GPU at 1x) instead of an assumed
# one, per THROUGHPUT_SCALING_PLAN.md. Reuses
# muon_qwen3_0.1B_longctx_4gpu.yaml as base (already has
# /backend: megatron_torchrun, the checkpoint/async fixes, dp_cp=4 proven
# working) rather than building a new config from scratch.
#
# global_batch_size stays 32 (unchanged) — with micro_batch_size=4 and
# dp=4, that's exactly 1 grad-accum step per iteration (4*4=32), the
# cleanest possible throughput measurement (no extra accumulation loop
# diluting the per-iteration timing).
#
# wandb routed to heavy_weights/muon-throughput-testing, same as the 1-GPU
# sweep. MASTER_PORT randomized (same fix as 06, applied here for
# consistency even though only one job runs this time). Unique
# job.base_output_dir keeps this run's wandb/logs isolated from the earlier
# 4-GPU smoke test's ./output artifacts.

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
    backend.megatron.micro_batch_size=4 \
    backend.megatron.train_iters=80 \
    backend.megatron.wandb_entity=heavy_weights \
    backend.megatron.wandb_project=muon-throughput-testing \
    "slurm.env.MASTER_PORT=$((20000 + RANDOM % 20000))" \
    stage=throughput_4gpu_mbs4 \
    "++job.base_output_dir=outputs/throughput_sweep/qwen3_dense_0.1B_ne_mbs4_4gpu" \
    --array-subset 0 \
    --local
# Compare against the 1-GPU number: 98,495 tok/s/GPU at mbs=4. Aggregate
# 4-GPU throughput = per-GPU number here * 4; scaling efficiency =
# (this run's tok/s/GPU) / 98,495.
