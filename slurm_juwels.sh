#!/bin/bash -x
#
# JUWELS SLURM Training Script
#
# Usage:
#   sbatch slurm_juwels.sh                                    # Default: juwels + norm_gpt + slimpajama + neox
#   sbatch slurm_juwels.sh --model.flavor=1B --training.steps=20000  # Override parameters
#
#   DATASET=fineweb_edu sbatch slurm_juwels.sh               # Use different dataset
#   TOKENIZER=llama3 sbatch slurm_juwels.sh                  # Use different tokenizer
#   CONFIG=base_plus.toml sbatch slurm_juwels.sh             # Use gpt_plus model
#   CLUSTER=jupiter sbatch slurm_juwels.sh                   # Use different cluster paths (for testing)
#   TITAN_USER=korbi sbatch slurm_juwels.sh                  # Use different user's config (default: joerg)
#
# Environment variables:
#   TITAN_USER - Username for user-specific configs (default: joerg)
#   CLUSTER    - Cluster name from cluster_paths.toml (default: juwels)
#   DATASET    - Dataset name from cluster_paths.toml (default: slimpajama_627b)
#   TOKENIZER  - Tokenizer name from cluster_paths.toml (default: neox)
#   CONFIG     - Base config file (default: base_norm.toml)
#
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=llmApp
#SBATCH --account=transfernetx
#SBATCH --partition=booster
#SBATCH --threads-per-core=1
#SBATCH --time=1:00:00
#SBATCH --output=/p/scratch/projectnucleus/franke5/experiments/slurm/mpi-out.%j
#SBATCH --error=/p/scratch/projectnucleus/franke5/experiments/slurm/mpi-err.%j

# [Keep your NCCL exports as they are]
export NCCL_SOCKET_TIMEOUT=60000
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=100
export NCCL_IB_RETRY_CNT=20
export NCCL_ALGO=Ring
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET


# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo,+inductor"   # adjust to your build's accepted env vars

# Dataset and Tokenizer configuration - set via environment or use defaults
export TITAN_USER="${TITAN_USER:-joerg}"     # Username for user-specific configs (user/{joerg,korbi})
CLUSTER="juwels"
DATASET="${DATASET:-slimpajama_627b}"    # Dataset name from cluster_paths.toml
TOKENIZER="${TOKENIZER:-neox}"           # Tokenizer name from cluster_paths.toml
CONFIG="${CONFIG:-base_norm.toml}"       # Base config file (base_norm.toml or base_plus.toml)

# Project and container configuration
PROJECT_DIR=$(pwd)                       # Assume script is run from project root
CONTAINER="titan_juwels_0.4.1.sif"      # Container filename

# ============================================================================
# LOAD CLUSTER CONFIGURATION
# ============================================================================
# Load cluster-specific cache directories and paths from user-specific
# cluster_paths.toml via cluster_config.py
#
# This runs inside the container where torchtitan is available
echo "Loading cluster configuration for user '$TITAN_USER' on cluster '$CLUSTER'..."

# Load all cluster configuration as environment variables (run in container)
eval "$(apptainer exec \
    --env TITAN_USER=$TITAN_USER \
    --bind $PROJECT_DIR:/opt/titan-sci \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-sci')
from titan_sci.cluster_config import get_env_exports
try:
    print(get_env_exports('$CLUSTER'))
except Exception as e:
    print(f'echo \"Error loading cluster config: {e}\"', file=sys.stderr)
    sys.exit(1)
")"

# Check if configuration was loaded successfully
if [ -z "$TRITON_CACHE_DIR" ]; then
    echo "ERROR: Failed to load cluster configuration"
    echo "Please ensure user/$TITAN_USER/cluster_paths.toml has [cluster.$CLUSTER] section"
    exit 1
fi

# Additional HuggingFace settings (not in cluster config)
export HF_HUB_OFFLINE="1"
export HF_ALLOW_CODE_EVAL="1"

echo "Configuration loaded successfully:"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  CONTAINER: $CONTAINER"
echo "  Cache base: $(dirname $TRITON_CACHE_DIR)"

# Create cache directories if they don't exist
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TORCH_HOME"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
MASTER_IP=$(nslookup ${nodes[0]}i | grep "Address:" | tail -n1 | awk '{print $2}')

echo "Master IP: $MASTER_IP"
echo "Nodes: ${nodes[@]}"

ml Stages/2025
ml GCC/13.3.0
ml Python/3.12.3
ml CUDA/12
ml cuDNN/9.5.0.50-CUDA-12
ml NCCL/default-CUDA-12
ml NVHPC/25.5-CUDA-12
ml Apptainer-Tools

# Define variables
APPTAINER="apptainer exec --nv \
    --pwd /opt/titan-sci \
    --env TITAN_USER=$TITAN_USER \
    --env NCCL_SOCKET_IFNAME=ib0 \
    --env GLOO_SOCKET_IFNAME=ib0 \
    --env MASTER_ADDR=$MASTER_IP \
    --env MASTER_PORT=29500 \
    --env TORCH_HOME=$TORCH_HOME \
    --bind $PROJECT_DIR:/opt/titan-sci \
    --bind $TRITON_CACHE_DIR:$TRITON_CACHE_DIR \
    --bind $HF_DATASETS_CACHE:$HF_DATASETS_CACHE \
    --bind $TORCH_HOME:$TORCH_HOME \
    $PROJECT_DIR/$CONTAINER"

# Generate cluster-specific path arguments (with automatic validation)
# Note: Runs in container on master node before distributing to compute nodes
echo "Validating configuration and paths..."
CLUSTER_ARGS=$(apptainer exec \
    --env TITAN_USER=$TITAN_USER \
    --bind $PROJECT_DIR:/opt/titan-sci \
    $PROJECT_DIR/$CONTAINER \
    python3 -c "
import sys
sys.path.insert(0, '/opt/titan-sci')
from titan_sci.cluster_config import get_cli_args
print(get_cli_args('$DATASET', '$TOKENIZER', '$CLUSTER', '$CONFIG', '/opt/titan-sci/titan_sci/configs'))
")

# Check if validation failed
if [ $? -ne 0 ]; then
    echo "Configuration validation failed. Aborting."
    exit 1
fi
echo "Validation passed."

for (( i=0; i<$SLURM_NNODES; i++ )); do
    node=${nodes[$i]}
    node_ip=$(nslookup ${node}i | grep "Address:" | tail -n1 | awk '{print $2}')

    SRUN_ARGS="--exclusive -N1 -n1 -w ${node}"

    LAUNCHER="torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --node_rank=$i \
        --master_addr=$MASTER_IP \
        --master_port=29500 \
        --local_addr=$node_ip \
        --rdzv_backend=static \
        --rdzv_endpoint=$MASTER_IP:29500"

    srun $SRUN_ARGS $APPTAINER bash -c '
        cd /opt/titan-sci
        exec '"$LAUNCHER -m torchtitan.train --job.config_file=/opt/titan-sci/titan_sci/configs/$CONFIG $CLUSTER_ARGS $@" &

    if [ $i -eq 0 ]; then
        sleep 10
    else
        sleep 2
    fi
done
wait

# LAUNCHER="torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --node_rank=$i --master_addr=$MASTER_IP --master_port=29500 --local_addr=$node_ip --rdzv_backend=static --rdzv_endpoint=$MASTER_IP:29500"
