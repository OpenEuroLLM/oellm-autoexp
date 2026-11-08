#!/bin/bash
# Fabric health probe: one 1 GB all-reduce over all GPUs of a few nodes (ib_pair_probe.py
# POOL_PROBE_MODE=all), the gradient reduction's pattern across the network cells the job spans.
# Appends one line per probe to $FABRIC_PROBE_RESULTS:
#   <time> <job> <nodes> <cells spanned> <[probe] allreduce ... busbw min/median/max>
# Bands from the 2026-10-03 survey (8 GPUs, 1 GB): >=130 GB/s fine, <100 degraded.
# Submitted every 15 min by scripts/pool/fabric_probe_loop.sh (JUPITER 2026-11-08: training at
# ~20-60% speed for 8 h on every allocation, node checks clean).
#SBATCH --job-name=fabric_probe
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:10:00

set -u
REPO=${FABRIC_PROBE_REPO:?}
SIF=${FABRIC_PROBE_SIF:?}
RESULTS=${FABRIC_PROBE_RESULTS:?}

export NCCL_SOCKET_IFNAME=ib0 NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=ib0 GLOO_SOCKET_FAMILY=AF_INET
export POOL_PROBE_MODE=all POOL_PROBE_GPUS=4 POOL_PROBE_MB=${FABRIC_PROBE_MB:-1024}

# a node's network cell: the 5 racks under one L2 switch, (rack-1)//5
cells=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk -F- '{print int(($2-1)/5)*5+1}' | sort -un | paste -sd,)
out=$(srun --ntasks-per-node=1 --gpu-bind=none \
  apptainer exec --nv --bind /e "$SIF" python3 "$REPO/scripts/pool/ib_pair_probe.py" 2>&1)
line=$(grep -m1 "^\[probe\] allreduce" <<<"$out")
[ -z "$line" ] && line="FAILED: $(grep -m1 -E "FAIL|Error|error" <<<"$out" | cut -c1-200)"
printf '%s\t%s\t%s\tcells=%s\t%s\n' "$(date +%FT%T)" "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" \
  "$cells" "$line" >>"$RESULTS"
echo "$line"
