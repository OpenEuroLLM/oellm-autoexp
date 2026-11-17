#!/bin/bash
# Capture the next 516-node B1 host (job 2928727, host121), shrink it to 256 nodes, and run the
# DK1T experiment (DK1 + per-head attention temperature) with its paired control. Runs on the
# JUPITER login node:
#
#   nohup setsid bash scripts/korbi/recall_surgery/capture_host121_256.sh \
#       > /e/scratch/e-sta-openeurollm/poeppel1/monitor_logs/capture_host121.log 2>&1 &
#
# The user 2026-11-16: kill host120 (done) and use 256 nodes, not 252. This script uses the B1
# pool's host supervisor through its `assign/` files; it does not submit or cancel Slurm jobs.
# The only Slurm change is the authorized `scontrol update JobId=2928727 NodeList=<256>`.
#
# Sequence on host121:
#   1. wait for the job RUNNING and its info.env
#   2. keep the first 256 nodes (including the batch node), write the 260 removed nodes to
#      hosts/host121/exclude_nodes, resize the job to 256, verify
#   3. 1-node K surgery (surgery/DK1/ckpt/iter_0035900 + tracker)
#   4. 1-node add_head_temperature -> surgery/DK1T/ckpt/iter_0035900 + tracker
#   5. control 128 nodes (main 35900) + DK1T 128 nodes (surgery/DK1T), gbs 1024 each
set -uo pipefail

POOL=${POOL:-/e/scratch/e-ext-2025e02-108/oellm_32b_dense_b1/pools/oellm32b_b1_relaxed_20261103}
JOB=${JOB:-2928727}
HOST=${HOST:-host121}
OUR=${OUR_REPO:-/e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp-hybrid}
KV=${KV:-/e/scratch/e-ext-2025e02-108/poeppel1/ambitious_kvshare}
SIF=${SIF:-/e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif}
LIB=${LIB:-/e/fscratch/e-sta-openeurollm/poeppel1/ambitious84/env/pylibs_ep}
H=$POOL/hosts/$HOST
STOP=${STOP:-$KV/capture_host121.stop}
POLL=${POLL_S:-30}
MAX_H=${MAX_H:-14}
log() { echo "$(date '+%F %T') [capture121] $*"; }
job_state() { squeue -h -j "$JOB" -o "%T" 2>/dev/null | head -1; }

# one manual pool run: <rid> <nodes> <source sbatch> <log> <name> [stop_signal]
make_run() {
  local rid=$1 nodes=$2 src=$3 logf=$4 name=$5 sig=${6:-USR2}
  local r=$POOL/runs/$rid
  mkdir -p "$r"
  { echo '#!/bin/bash'; echo "cd $OUR || { echo no-workdir >&2; exit 97; }"; tail -n +2 "$src"; } > "$r/step.sh"
  chmod +x "$r/step.sh"
  cat > "$r/meta.env" <<EOF
STOP_SIGNAL=$sig
WORKDIR=$OUR
NODES=$nodes
NTASKS_PER_NODE=1
MAX_RUNTIME_S=0
LOG=$logf
NAME=$name
EXCLUDE=
USE_EXCLUDED_NODES=
EOF
  printf '%s\n' "$rid" > "$H/assign/$rid"
  log "assigned $rid ($name, $nodes node(s))"
}

# 1. wait for the host job
t0=$(date +%s)
while :; do
  [ -f "$STOP" ] && { log "STOP file: aborting"; exit 1; }
  st=$(job_state)
  if [ "$st" = "RUNNING" ] && [ -f "$H/info.env" ]; then log "host $HOST RUNNING"; break; fi
  log "waiting for $JOB (state ${st:-?})"
  [ $(( $(date +%s) - t0 )) -ge $((MAX_H * 3600)) ] && { log "timeout waiting for host"; exit 1; }
  sleep "$POLL"
done
sleep 10

# 2. shrink to 256, keeping the first 256 (the batch node is first in the allocation order)
mkdir -p "$H"
alloc=$(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)")
n=$(printf '%s\n' "$alloc" | grep -c .)
if [ "$n" -lt 256 ]; then log "only $n nodes in the allocation; aborting"; exit 1; fi
keep=$(printf '%s\n' "$alloc" | head -256)
removed=$(printf '%s\n' "$alloc" | tail -n +257)
printf '%s\n' "$removed" >> "$H/exclude_nodes"
keep_list=$(printf '%s\n' "$keep" | paste -sd, - | xargs scontrol show hostlist)
log "shrinking $JOB: $n -> 256 (keeping first 256 incl batch; removing $(printf '%s\n' "$removed" | grep -c .))"
scontrol update JobId="$JOB" NodeList="$keep_list"
now=$(squeue -h -j "$JOB" -o %D)
log "after shrink: $now nodes"
[ "$now" = "256" ] || { log "shrink did not reach 256 (got $now); aborting"; exit 1; }

# 3. K surgery, 1 node
SURG=$KV/surgery
mkdir -p "$SURG"
rm -rf "$SURG/DK1T"
SID=kvshare_surg121_$(date +%s)
make_run "$SID" 1 "$KV/dry_surgery/e/scratch/e-ext-2025e02-108/poeppel1/ambitious_kvshare/surgery/logs/job.sbatch" \
  "$SURG/surgery121.log" kvshare_main_surgery121 TERM
log "waiting for surgery tracker $SURG/DK1/ckpt/latest_checkpointed_iteration.txt = 35900"
t0=$(date +%s)
while :; do
  [ -f "$STOP" ] && { log "STOP file: aborting"; exit 1; }
  v=$(tr -d '[:space:]' < "$SURG/DK1/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || true)
  [ "$v" = "35900" ] && { log "surgery done"; break; }
  [ $(( $(date +%s) - t0 )) -ge $((2 * 3600)) ] && { log "surgery timeout"; exit 1; }
  sleep "$POLL"
done

# 4. add_head_temperature -> DK1T checkpoint (1 node, same file system: hard links)
HT=$KV/headtemp121.sh
cat > "$HT" <<EOF
#!/bin/bash
cd $OUR || exit 97
apptainer exec --bind /e --env PYTHONPATH=.:submodules/Megatron-LM:flash-linear-attention:submodules/cute_kernels:$LIB --env TRITON_LIBCUDA_PATH=/.singularity.d/libs $SIF \\
  python3 scripts/korbi/production/add_head_temperature.py \\
  --src $SURG/DK1/ckpt/iter_0035900 --dst $SURG/DK1T/ckpt/iter_0035900
rc=\$?
if [ \$rc -eq 0 ]; then echo 35900 > $SURG/DK1T/ckpt/latest_checkpointed_iteration.txt; fi
exit \$rc
EOF
chmod +x "$HT"
HID=kvshare_headtemp121_$(date +%s)
R=$POOL/runs/$HID
mkdir -p "$R"
cp "$HT" "$R/step.sh"; chmod +x "$R/step.sh"
cat > "$R/meta.env" <<EOF
STOP_SIGNAL=TERM
WORKDIR=$OUR
NODES=1
NTASKS_PER_NODE=1
MAX_RUNTIME_S=0
LOG=$SURG/headtemp121.log
NAME=kvshare_headtemp121
EXCLUDE=
USE_EXCLUDED_NODES=
EOF
printf '%s\n' "$HID" > "$H/assign/$HID"
log "assigned $HID (head temperature)"
log "waiting for DK1T tracker"
t0=$(date +%s)
while :; do
  [ -f "$STOP" ] && { log "STOP file: aborting"; exit 1; }
  v=$(tr -d '[:space:]' < "$SURG/DK1T/ckpt/latest_checkpointed_iteration.txt" 2>/dev/null || true)
  [ "$v" = "35900" ] && { log "DK1T checkpoint done"; break; }
  [ $(( $(date +%s) - t0 )) -ge $((3600)) ] && { log "head-temperature timeout"; exit 1; }
  sleep "$POLL"
done

# 5. control + DK1T, 128 nodes each (gbs 1024)
CID=kvshare_ctrl121_$(date +%s)
DID=kvshare_dk1t121_$(date +%s)
make_run "$CID" 128 "$KV/dry_ctrl/e/scratch/e-ext-2025e02-108/poeppel1/ambitious_kvshare/runs/ctrl/amb84_jsc_ctrl.sbatch" \
  "$KV/runs/ctrl/ctrl121.log" amb84_jsc_ctrl121
make_run "$DID" 128 "$KV/dry_dk1t128/e/scratch/e-ext-2025e02-108/poeppel1/ambitious_kvshare/runs/DK1T/amb84_jsc_DK1T.sbatch" \
  "$KV/runs/DK1T/dk1t121.log" amb84_jsc_DK1T121
log "control $CID + DK1T $DID assigned; 256 nodes used. done."
