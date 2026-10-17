#!/bin/bash
{sbatch_directives}

{env_exports}

# Free stale node-local scratch left by earlier/cancelled jobs (nodes are exclusive, so anything
# under /scratch/tmp not owned by THIS job is dead) and create the mirror dir on EVERY node.
# Without this, one dirty node with a near-full /scratch ENOSPCs the prefetcher mid-stage and
# (one task exit -> whole step) kills the run. Also lays down the inspection symlink.
srun --ntasks-per-node=1 --nodes=$SLURM_NNODES --overlap bash -c \
  'find /scratch/tmp -maxdepth 1 -mindepth 1 ! -name "$SLURM_JOB_ID" -exec rm -rf {{}} + 2>/dev/null; mkdir -p "$OELLM_MIRROR_DIR"'
ln -sfn /scratch/tmp/$SLURM_JOB_ID $OELLM_NODE_LOCAL_LINK 2>/dev/null || true

# --- Async node-local mirror prefetcher: one per node, backgrounded, before training ---
# Continuously fills a node-local sparse mirror in per-node consumption order (partial:
# only this node's ranks' documents; bounded to --budget-gb). Training reads THROUGH the
# mirror (OELLM_MIRROR_DIR + pretrain_gpt_prefetch.py): blocks not yet present are served
# from GPFS, so training starts immediately and is correct at any fill state.
echo "Launching node-local mirror prefetcher on $SLURM_NNODES node(s)"
srun --ntasks-per-node=1 --nodes=$SLURM_NNODES --overlap --cpu-bind=none python $PREFETCH_SCRIPT --mode $PREFETCH_MODE --lanes $OELLM_SHUFFLE_LANES --lane-block-docs $PREFETCH_LANE_BLOCK --threads $PREFETCH_THREADS --mirror-dir $OELLM_MIRROR_DIR --prefix $PREFETCH_PREFIX --cache-dir $PREFETCH_CACHE_DIR --dp-size $NUM_GPUS --gpus-per-node $NUM_GPUS_PER_NODE --micro-batch-size $PREFETCH_MBS --wait-for-cache 1200 --budget-gb $PREFETCH_BUDGET_GB --passes $PREFETCH_PASSES &
PREFETCH_PID=$!
# --------------------------------------------------------------------------------------

echo "Launching {job_name} on MARENOSTRUM (node-local mirror read-through)"
srun {srun_opts}bash -c '{launcher_cmd} {command}'

kill $PREFETCH_PID 2>/dev/null; wait $PREFETCH_PID 2>/dev/null || true
