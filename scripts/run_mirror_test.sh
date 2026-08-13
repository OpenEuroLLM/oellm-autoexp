#!/bin/bash
# Submit a node-local mirror read-through test on MareNostrum with all known workarounds.
# Usage: run_mirror_test.sh <data_prefix_no_ext|-> [train_iters] [budget_gb] [qos] [time]
# Pass "-" as the prefix to take data_path + PREFETCH_PREFIX from the config (set CONFIG_NAME);
# this is how the multi-file blend test runs (commas/lists cannot be passed as CLI overrides).
# Example (big single-file test): bash scripts/run_mirror_test.sh \
#   /gpfs/scratch/ehpc390/data/cerebras-SlimPajama-627B/train/merged 500 400
# Example (blend test): CONFIG_NAME=experiments/megatron_marenostrum_blend_test \
#   bash scripts/run_mirror_test.sh - 1500 50
set -e
cd "$(dirname "$0")/.."
export OELLM_REPO="$PWD"
PREFIX="${1:?data prefix (no .bin/.idx), or - to use the config data_path}"
ITERS="${2:-200}"
BUDGET="${3:-400}"
QOS="${4:-acc_debug}"
TIME="${5:-00:30:00}"
CONFIG_NAME="${CONFIG_NAME:-experiments/megatron_marenostrum_speed_test}"
CACHE="/gpfs/projects/ehpc390/outputs/mirror_$$_dcache"
# data_path + PREFETCH_PREFIX come from the config when PREFIX is "-" (blend / list values).
DATA_OVERRIDES=()
if [ "$PREFIX" != "-" ]; then
  DATA_OVERRIDES=("backend.megatron.data_path=[$PREFIX]" "slurm.env.PREFETCH_PREFIX=$PREFIX")
fi
NODE_OVERRIDE=()
[ -n "${NODES:-}" ] && NODE_OVERRIDE=("slurm.sbatch.nodes=$NODES")
LOG_OVERRIDE=()
[ -n "${LOG_INTERVAL:-}" ] && LOG_OVERRIDE=("backend.megatron.log_interval=$LOG_INTERVAL")
rm -rf "$CACHE"
touch submodules/Megatron-LM/megatron/core/datasets/helpers_cpp*.so
PYTHONPATH=. python scripts/run_autoexp.py \
  --config-name "$CONFIG_NAME" slurm=marenostrum_mirror \
  backend.launcher_script=./scripts/pretrain_gpt_prefetch.py \
  "${DATA_OVERRIDES[@]}" "${NODE_OVERRIDE[@]}" "${LOG_OVERRIDE[@]}" \
  "backend.megatron.data_cache_path=$CACHE" "slurm.env.PREFETCH_CACHE_DIR=$CACHE" \
  "container.bind=[/gpfs/projects,/gpfs/scratch,/scratch]" \
  "backend.megatron.train_iters=$ITERS" \
  "backend.megatron.load=null" "backend.megatron.save=null" \
  "backend.megatron.dataloader_type=single" \
  "slurm.sbatch.partition=acc" "slurm.sbatch.qos=$QOS" "slurm.sbatch.time=$TIME" \
  "slurm.sbatch.job_name=null" "~slurm.env.SLURM_CPUS_PER_TASK" \
  "+slurm.env.CXX=g++" "+slurm.env.CC=gcc" \
  "+slurm.env.TORCHDYNAMO_DISABLE=1" "+slurm.env.TORCH_COMPILE_DISABLE=1" \
  "+slurm.env.OELLM_MIRROR_LOG_EVERY=${OELLM_MIRROR_LOG_EVERY:-20000}" \
  "slurm.env.OELLM_SHUFFLE_LANES=${OELLM_SHUFFLE_LANES:-256}" \
  "slurm.env.OELLM_SHUFFLE_BLOCK=${OELLM_SHUFFLE_BLOCK:-8192}" \
  "slurm.env.PREFETCH_MODE=${PREFETCH_MODE:-lanes}" \
  "slurm.env.PREFETCH_LANE_BLOCK=${PREFETCH_LANE_BLOCK:-32768}" \
  "slurm.env.OELLM_LOOKAHEAD_BLOCKS=${OELLM_LOOKAHEAD_BLOCKS:-3}" \
  "slurm.env.OELLM_RETAIN_BLOCKS=${OELLM_RETAIN_BLOCKS:-1}" \
  "slurm.env.PREFETCH_BUDGET_GB=$BUDGET" \
  --submit-and-exit
