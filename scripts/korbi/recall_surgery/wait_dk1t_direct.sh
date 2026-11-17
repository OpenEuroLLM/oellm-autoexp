#!/bin/bash
# Wait for the K surgery, add the head-temperature parameters, then start the DK1T training directly
# inside the Jupiter 256-node allocation (the pool preflight failed on a slow GPU, so the pool holds the
# host without runs; direct srun steps use the 249 usable nodes). Runs on the Jupiter login node:
#   nohup setsid bash wait_dk1t_direct.sh > .../logs/wait_dk1t.log 2>&1 &
set -uo pipefail
P=/e/scratch/e-sta-openeurollm/poeppel1/pools/jupiter_pool_amb84
KV=/e/scratch/e-ext-2025e02-108/poeppel1/ambitious_kvshare
OUR=/e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp-hybrid
SIF=/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif
LIB=/e/fscratch/e-sta-openeurollm/poeppel1/ambitious84/env/pylibs_ep
SURG=/e/fscratch/e-sta-openeurollm/poeppel1/ambitious84/kvshare_surgery
T=$SURG/DK1/ckpt/latest_checkpointed_iteration.txt
log() { echo "$(date '+%F %T') [dk1t] $*"; }

log "waiting for K surgery tracker"
while :; do
  v=""
  if [ -f "$T" ]; then v=$(tr -d '[:space:]' < "$T"); fi
  [ "$v" = "35900" ] && break
  sleep 30
done

log "surgery done; creating head-temperature checkpoint"
rm -rf "$SURG/DK1T"
cd "$OUR" || exit 97
apptainer exec --bind /e \
  --env PYTHONPATH=.:submodules/Megatron-LM:flash-linear-attention:submodules/cute_kernels:$LIB \
  --env TRITON_LIBCUDA_PATH=/.singularity.d/libs "$SIF" \
  python3 scripts/korbi/production/add_head_temperature.py \
  --src "$SURG/DK1/ckpt/iter_0035900" --dst "$SURG/DK1T/ckpt/iter_0035900"
rc=$?
if [ $rc -ne 0 ]; then log "add_head_temperature failed rc=$rc"; exit $rc; fi
echo 35900 > "$SURG/DK1T/ckpt/latest_checkpointed_iteration.txt"

mkdir -p "$KV/runs/DK1T"
log "starting direct DK1T launcher"
nohup setsid bash "$KV/run_dk1t_direct.sh" > "$KV/runs/DK1T/dk1t_direct.log" 2>&1 < /dev/null &
log "DK1T direct launcher started"
