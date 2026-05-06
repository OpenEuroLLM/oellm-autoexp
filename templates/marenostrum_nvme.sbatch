#!/bin/bash
{sbatch_directives}

{env_exports}

# ── Per-node NVMe data staging ────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Starting NVMe data staging on $SLURM_JOB_NUM_NODES nodes..."
srun --ntasks-per-node=1 --cpus-per-task=8 --threads-per-core=1 \
    bash $PROJECT_DIR/scripts/stage_nvme.sh \
    "$NVME_SRC_DATA_ARGS" \
    "$NVME_DST_DATA_ARGS"
echo "[$(date '+%H:%M:%S')] Staging done."
# ─────────────────────────────────────────────────────────────────────────────

echo "Launching job {job_name}"
srun {srun_opts} bash -c '{launcher_cmd} {command}'
