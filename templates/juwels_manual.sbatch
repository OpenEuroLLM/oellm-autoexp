#!/bin/bash
{sbatch_directives}

{env_exports}

# --- Module Setup ---
ml Stages/2025
ml GCC/13.3.0
ml Python/3.12.3
ml CUDA/12
ml cuDNN/9.5.0.50-CUDA-12
ml NCCL/default-CUDA-12
ml OpenMPI/5.0.5
ml ParaStationMPI/5.11.0-1
ml ParaStationMPI/5.11.0-1-mt
ml ScaLAPACK/2.2.0-fb

echo "Launching {name} on JUWELS (Manual Loop)"

# Get nodes and master IP
# Use subshell to avoid polluting environment
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
MASTER_IP=$(nslookup ${{nodes[0]}}i | grep "Address:" | tail -n1 | awk '{{ print $2 }}')

echo "Master IP: $MASTER_IP"
echo "Nodes: ${{nodes[@]}}"

# Defaults if not provided in template
MASTER_PORT={master_port}
GPUS_PER_NODE={gpus_per_node}

# Loop over nodes to launch tasks explicitly
for (( i=0; i<$SLURM_NNODES; i++ )); do
    node=${{nodes[$i]}}
    # Resolve node IP (suffix 'i' for Infiniband interface on JUWELS)
    node_ip=$(nslookup ${{node}}i | grep "Address:" | tail -n1 | awk '{{ print $2 }}')

    echo "Starting rank $i on $node ($node_ip)"

    # Force execution on specific node
    SRUN_ARGS="--exclusive -N1 -n1 -w ${{node}}"

    # Construct Torchrun command with explicit node rank and address
    LAUNCHER="torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$i \
        --master_addr=$MASTER_IP \
        --master_port=$MASTER_PORT \
        --local_addr=$node_ip \
        --rdzv_backend=static \
        --rdzv_endpoint=$MASTER_IP:$MASTER_PORT"

    # Execute
    # We pass the launcher and command to the srun shell
    srun $SRUN_ARGS bash -c "
        export LOCAL_ADDR=$node_ip
        {command_prefix}
        exec $LAUNCHER {command}
    " &

    # Stagger start to ease load on master
    if [ $i -eq 0 ]; then
        sleep 10
    else
        sleep 2
    fi
done
wait
