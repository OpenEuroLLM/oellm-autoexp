#!/usr/bin/env bash
# Standalone JUPITER scan of flagship checkpoints strictly before iteration 60k.
# Submit from the oellm-autoexp repository root:
#   sbatch scripts/scan_prior_checkpoints_jupiter.sbatch

#SBATCH --account=e-ext-2025e02-108
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --gpu-bind=none
#SBATCH --comment=NVIDIA_GPU_CLOCKS=1980
#SBATCH --job-name=oellm_32b_prior_scan
#SBATCH --open-mode=append
#SBATCH --output=/dev/null

set -euo pipefail

module purge
ulimit -c 0

REPO_ROOT="${OELLM_AUTOEXP_ROOT:-$SLURM_SUBMIT_DIR}"
CHECKPOINT_ROOT="/e/scratch/e-sta-openeurollm/production_training/oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4/checkpoints"
OUTPUT_ROOT="/e/home/jusers/luukkonen1/jupiter/e-sta-workdir/oellm-autoexp/checkpoint_scan_prior_${SLURM_JOB_ID}"
CONTAINER="/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif"
PYTHON_BIN="/opt/venv/bin/python"
ITERATIONS="4000,8000,12000,16000,20000,24000,28000,32000,34454,36000,40000,43200,44000,48000,52000,56000,57940"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/artifacts"
exec > >(tee -a "$OUTPUT_ROOT/logs/slurm-${SLURM_JOB_ID}.log") 2>&1

export MACHINE_NAME=JUPITER
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=64
export GPUS_PER_NODE=1
export NUM_GPUS_PER_NODE=1
export NUM_GPUS=1
export NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1
export TRITON_CACHE_DIR="/dev/shm/${USER}/checkpoint_scan_${SLURM_JOB_ID}/triton"
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/${USER}/checkpoint_scan_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="/dev/shm/${USER}/checkpoint_scan_${SLURM_JOB_ID}/extensions"
export CUDA_CACHE_PATH="/dev/shm/${USER}/checkpoint_scan_${SLURM_JOB_ID}/cuda"

cd "$REPO_ROOT"

for iteration in ${ITERATIONS//,/ }; do
    printf -v checkpoint "%s/iter_%07d" "$CHECKPOINT_ROOT" "$iteration"
    if [[ ! -f "$checkpoint/.metadata" ]]; then
        echo "FATAL: missing checkpoint metadata: $checkpoint/.metadata" >&2
        exit 2
    fi
done

echo "[prior-checkpoint-scan] repo       : $REPO_ROOT"
echo "[prior-checkpoint-scan] checkpoints: $ITERATIONS"
echo "[prior-checkpoint-scan] output     : $OUTPUT_ROOT"

srun --kill-on-bad-exit=1 apptainer exec \
    --env NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1 \
    --env PYTHONPATH="$REPO_ROOT/submodules/Megatron-LM:$REPO_ROOT" \
    --writable-tmpfs \
    --nv \
    --bind /e \
    --bind /dev/shm \
    "$CONTAINER" \
    "$PYTHON_BIN" scripts/scan_checkpoint_stats.py \
        "$CHECKPOINT_ROOT" \
        --iterations "$ITERATIONS" \
        --run flagship_prior \
        --optimizer off \
        --comparison-sample-elements 16384 \
        --strict \
        --fail-on-nonfinite \
        --output-dir "$OUTPUT_ROOT/artifacts/flagship_prior"

srun --kill-on-bad-exit=1 apptainer exec \
    --env PYTHONPATH="$REPO_ROOT/submodules/Megatron-LM:$REPO_ROOT" \
    --writable-tmpfs \
    --nv \
    --bind /e \
    --bind /dev/shm \
    "$CONTAINER" \
    "$PYTHON_BIN" scripts/compare_checkpoint_stats.py \
        --scan "flagship_prior=$OUTPUT_ROOT/artifacts/flagship_prior" \
        --output-dir "$OUTPUT_ROOT/artifacts/comparison"

echo "[prior-checkpoint-scan] completed"
echo "[prior-checkpoint-scan] artifacts: $OUTPUT_ROOT/artifacts"
