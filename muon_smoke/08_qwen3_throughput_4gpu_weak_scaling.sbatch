#!/bin/bash
#SBATCH --job-name=muon_throughput_4gpu_weak
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:20:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/08_throughput_4gpu_weak_%j.log

# WEAK-scaling comparison point (vs. 07's STRONG-scaling one).
#
# 07_qwen3_throughput_4gpu.sbatch held global_batch_size=32 fixed while going
# 1->4 GPUs, so each GPU did LESS local work as GPU count grew (8 grad-accum
# steps/GPU at 1x, only 2 at 4x) — the measured 84.0% efficiency there
# conflates real communication overhead with reduced per-GPU compute-to-
# overhead ratio. This run isolates the communication cost specifically:
# global_batch_size=128 (not 32) keeps grad_accum_steps=8 per GPU identical
# to the 1-GPU baseline (job with mbs=4, global_batch=32, dp=1) — same local
# work per device, only the total/aggregate problem size grows with GPU
# count. Compare this run's tok/s/GPU directly against the 1-GPU baseline's
# 98,495 tok/s/GPU for a communication-only efficiency number.

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
    backend.megatron.global_batch_size=128 \
    backend.megatron.train_iters=80 \
    backend.megatron.wandb_entity=heavy_weights \
    backend.megatron.wandb_project=muon-throughput-testing \
    "slurm.env.MASTER_PORT=$((20000 + RANDOM % 20000))" \
    stage=throughput_4gpu_weak_mbs4 \
    "++job.base_output_dir=outputs/throughput_sweep/qwen3_dense_0.1B_ne_mbs4_4gpu_weak" \
    --array-subset 0 \
    --local
# Compare against the 1-GPU baseline: 98,495 tok/s/GPU (job at
# outputs/throughput_sweep/qwen3_dense_0.1B_ne_mbs4_1gpu). Weak-scaling
# efficiency = (this run's tok/s/GPU) / 98,495.
