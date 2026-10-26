#!/bin/bash
{sbatch_directives}

{env_exports}

# --- Async page-cache prefetcher: one per node, backgrounded, before training ----
# Warms (in consumption order, far ahead of the GPU cursor) the page cache that the
# trainer's mmap reader faults from. Reads the SAME GPFS inode the container mmaps, so
# the warmed pages are shared. No node-local copy; data_path stays on GPFS.
# Both this step and the training step use --overlap so they share the nodes.
echo "Launching async prefetcher on $SLURM_NNODES node(s)"
srun --ntasks-per-node=1 --nodes=$SLURM_NNODES --overlap --cpu-bind=none python $PREFETCH_SCRIPT --prefix $PREFETCH_PREFIX --cache-dir $PREFETCH_CACHE_DIR --dp-size $NUM_GPUS --gpus-per-node $NUM_GPUS_PER_NODE --micro-batch-size $PREFETCH_MBS --wait-for-cache 900 --budget-gb $PREFETCH_BUDGET_GB &
PREFETCH_PID=$!
# -------------------------------------------------------------------------------

echo "Launching {job_name} on MARENOSTRUM (async page-cache prefetch)"
srun {srun_opts}bash -c '{launcher_cmd} {command}'

kill $PREFETCH_PID 2>/dev/null; wait $PREFETCH_PID 2>/dev/null || true
