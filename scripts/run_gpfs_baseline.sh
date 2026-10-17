#!/bin/bash
# Baseline arm of the training A/B: standard Megatron reading the data DIRECT from GPFS with the
# normal GLOBAL shuffle (no node-local mirror, no windowed shuffle, default launcher). This is the
# "today" behavior we compare the mirror against. Same model/data/nodes as the mirror arm
# (scripts/run_mirror_test.sh with NODES set) so only the data path differs.
#
#   NODES=8 bash scripts/run_gpfs_baseline.sh /gpfs/scratch/ehpc390/data/cerebras-SlimPajama-627B/train/merged 120
set -e
cd "$(dirname "$0")/.."
export OELLM_REPO="$PWD"
PREFIX="${1:?data prefix (no .bin/.idx)}"
ITERS="${2:-120}"
NODES="${NODES:-8}"
QOS="${QOS:-acc_debug}"
TIME="${TIME:-00:25:00}"
CONFIG_NAME="${CONFIG_NAME:-experiments/megatron_marenostrum_speed_test}"
CACHE="/gpfs/projects/ehpc390/outputs/baseline_$$_dcache"
LOG_OVERRIDE=()
[ -n "${LOG_INTERVAL:-}" ] && LOG_OVERRIDE=("backend.megatron.log_interval=$LOG_INTERVAL")
rm -rf "$CACHE"
touch submodules/Megatron-LM/megatron/core/datasets/helpers_cpp*.so
PYTHONPATH=. python scripts/run_autoexp.py \
  --config-name "$CONFIG_NAME" slurm=marenostrum \
  "${LOG_OVERRIDE[@]}" \
  "backend.megatron.data_path=[$PREFIX]" \
  "backend.megatron.data_cache_path=$CACHE" \
  "container.bind=[/gpfs/projects,/gpfs/scratch]" \
  "backend.megatron.train_iters=$ITERS" \
  "backend.megatron.load=null" "backend.megatron.save=null" \
  "backend.megatron.dataloader_type=single" \
  "slurm.sbatch.nodes=$NODES" \
  "slurm.sbatch.partition=acc" "slurm.sbatch.qos=$QOS" "slurm.sbatch.time=$TIME" \
  "slurm.sbatch.job_name=null" "~slurm.env.SLURM_CPUS_PER_TASK" \
  "+slurm.env.CXX=g++" "+slurm.env.CC=gcc" \
  "+slurm.env.TORCHDYNAMO_DISABLE=1" "+slurm.env.TORCH_COMPILE_DISABLE=1" \
  --submit-and-exit
