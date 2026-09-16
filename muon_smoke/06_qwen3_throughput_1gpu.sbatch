#!/bin/bash
#SBATCH --job-name=muon_throughput_1gpu
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/h1/sala597i/oellm-autoexp/muon_smoke/logs/06_throughput_1gpu_%j.log

# Throughput sweep point: 1x H100, real Qwen3 (model size parametrized) +
# Muon + real Nemotron data (same passing config as
# 04_qwen3_real_data_test_capella.sbatch), varying micro_batch_size AND
# model size — everything else held fixed, per THROUGHPUT_SCALING_PLAN.md.
# train_iters bumped to 80 (vs the smoke test's 50) so there's a real
# steady-state window after discarding warmup — iteration 1-2 alone can eat
# 20-30s of one-time compile cost (see prior runs' logs). save_interval left
# at the huge yaml default (no checkpoint save — separate cost, would skew
# whichever iteration it lands on).
#
# wandb routed to the user's own heavy_weights org / muon-throughput-testing
# project, not openeurollm-project — kept separate from the real experiment
# tracking. Offline mode as always; `wandb sync <dir>` later uploads using
# whatever wandb login is active at sync time — unaffected by this change,
# just needs write access to heavy_weights.
#
# Usage: sbatch 06_qwen3_throughput_1gpu.sbatch <micro_batch_size> <model>
# model is one of: qwen3_dense_0.1B_ne, qwen3_dense_0.2B_ne,
#                   qwen3_dense_0.4B_ne, qwen3_dense_0.9B_ne
# e.g.:  sbatch 06_qwen3_throughput_1gpu.sbatch 8  qwen3_dense_0.4B_ne
#        sbatch 06_qwen3_throughput_1gpu.sbatch 16 qwen3_dense_0.9B_ne
# Submit each (mbs, model) combo as its own job — they queue and can run in
# parallel rather than paying the ~4-7min fixed startup cost repeatedly in
# one serial job.
#
# Larger models: mem bumped to 48G vs the smoke test's 32G as a conservative
# margin — not measured yet whether 0.9B actually needs it. If a combo OOMs,
# that itself is useful data (tells you the ceiling for that model size on a
# single GPU) — don't just keep raising micro_batch_size blindly across
# model sizes without checking.
#
# Safe to run in parallel: --local force-overrides job.base_output_dir to a
# flat ./output UNLESS the caller already sets it (run_autoexp.py:172-173) —
# this is exactly the bug that caused the earlier stale-checkpoint collision
# (see muon_qwen3_0.1B_longctx_4gpu.yaml's comment). Explicit
# ++job.base_output_dir override below gives each (model, mbs) combo its own
# dir, so concurrent jobs can't clobber each other's current.log/wandb/etc.
#
# ALSO fixed: /slurm: local hardcodes MASTER_PORT: "29500" for every job.
# Confirmed via real crash (jobs 3875237/3875239, same physical node c122,
# 8s apart): torch.distributed.DistNetworkError EADDRINUSE on port 29500 —
# two single-GPU jobs landing on the same node both tried to bind the same
# port for their own (independent, single-process) rendezvous. Randomized
# per-job below.

MBS=${1:?Usage: sbatch 06_qwen3_throughput_1gpu.sbatch <micro_batch_size> <model>}
MODEL=${2:?Usage: sbatch 06_qwen3_throughput_1gpu.sbatch <micro_batch_size> <model>}

set -eo pipefail
cd /home/h1/sala597i/oellm-autoexp

SANDBOX=/data/horse/ws/sala597i-slaing/containers/.tmp_sandbox/oellm_sandboxv514iw4i/sandbox

singularity exec --underlay --nv -B /data/horse/ws/sala597i-slaing \
  --env PYTHONPATH=$PWD:$PWD/muon-pylibs:submodules/Megatron-LM \
  --env TORCHDYNAMO_DISABLE=1 \
  "$SANDBOX" python3 -u scripts/run_autoexp.py \
    --config-name experiments/laingsam/muon_qwen3_0.1B_longctx \
    '~backend.megatron.norm_epsilon' \
    "backend/megatron/multilingual_scaling/models=$MODEL" \
    backend.megatron.eval_iters=0 \
    backend.megatron.micro_batch_size=$MBS \
    backend.megatron.train_iters=80 \
    backend.megatron.wandb_entity=heavy_weights \
    backend.megatron.wandb_project=muon-throughput-testing \
    "slurm.env.MASTER_PORT=$((20000 + RANDOM % 20000))" \
    stage=throughput_1gpu_${MODEL}_mbs${MBS} \
    "++job.base_output_dir=outputs/throughput_sweep/${MODEL}_mbs${MBS}_1gpu" \
    --array-subset 0 \
    --local
# Read results from the printed job.base_output_dir above (or its wandb
# offline dir) after it finishes — grep for "iteration", discard the first
# ~10 (warmup), average "throughput per GPU (TFLOP/s/GPU)" /
# "Tokens per second per GPU" over the rest.
