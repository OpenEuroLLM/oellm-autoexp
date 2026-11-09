#!/bin/bash
# Can a pool host give nodes back while it runs? 3 nodes; `scontrol update JobId= NodeList=`
# drops the third one while a GPU step runs on the other two (like a training step), then:
#   - does the shrink work for a user on this cluster, what do squeue / SLURM_JOB_NODELIST say
#   - do later GPU steps on the kept nodes still start, or does the step that ran across the
#     shrink keep its GRES (feat/hybrid-v0.19 cd3efd75, MN5 2026-11-06: "Unable to create step
#     ... Insufficient GRES available in allocation" on every later step on that node)
#   - does a step on the removed node fail cleanly
# Run: sbatch --account=<acct> --output=<dir>/shrink_probe-%j.log scripts/pool/shrink_probe.sbatch
#SBATCH --job-name=shrink_probe
#SBATCH --partition=booster
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00

set -u
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
log() { echo "[shrink_probe] $(date +%T) $*"; }
gpu_step() { # name, nodelist, node count
  timeout 120 srun --overlap --mem=0 --jobid="$SLURM_JOB_ID" --nodelist="$2" --nodes="$3" \
    --ntasks="$3" --ntasks-per-node=1 --gpus-per-node=4 --job-name="$1" \
    bash -c 'echo "$(hostname -s) gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"' 2>&1 | paste -sd' ' -
}

log "start: ${NODES[*]}"
log "before: $(gpu_step pre "${NODES[0]},${NODES[1]},${NODES[2]}" 3)"

srun --overlap --mem=0 --jobid="$SLURM_JOB_ID" --nodelist="${NODES[0]},${NODES[1]}" --nodes=2 \
  --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 --job-name=long bash -c 'sleep 90' &
LONG=$!
sleep 20
log "shrink: drop ${NODES[2]}"
scontrol update JobId="$SLURM_JOB_ID" NodeList="${NODES[0]},${NODES[1]}" 2>&1 | sed 's/^/  scontrol: /'
log "rc=${PIPESTATUS[0]} squeue %N=$(squeue -h -j "$SLURM_JOB_ID" -o %N) %D=$(squeue -h -j "$SLURM_JOB_ID" -o %D) SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
log "during the long step, node0: $(gpu_step mid "${NODES[0]}" 1)"
wait "$LONG"
log "long step (ran across the shrink) ended rc=$?"
sleep 5
log "after, both kept nodes: $(gpu_step post "${NODES[0]},${NODES[1]}" 2)"
log "after, node0 alone: $(gpu_step post0 "${NODES[0]}" 1)"
log "removed node: $(gpu_step gone "${NODES[2]}" 1)"
sacct -j "$SLURM_JOB_ID" -n -o JobID%16,JobName%10,State%12,NodeList%45
log done
