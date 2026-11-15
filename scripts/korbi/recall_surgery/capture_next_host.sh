#!/bin/bash
# Capture the next 516-node B1-32B host (currently job 2925729, host117, StartTime ~02:09) for the
# K-style ambitious intervention, without touching the Slurm job and without letting the B1 pool
# hand it to the 32B dense run. Runs on the JUPITER login node:
#
#   nohup setsid bash scripts/korbi/recall_surgery/capture_next_host.sh \
#       > /e/scratch/e-sta-openeurollm/poeppel1/monitor_logs/capture_host117.log 2>&1 &
#
# What it does (all pool-level, never scontrol/scancel/requeue):
#   1. at SETTINGS_AT (default 01:30) or as soon as the host starts:
#        B1 pool handover=false            (B1 stays on host115; no handover to host117)
#        B1 pool max_runs_per_host=5       (1-node surgery + four 128-node arms fit on host117)
#      The 32B pool's older code has no per-host hold/freeze: the first 128-node control run
#      assigned to host117 is the keeper that prevents the idle release.
#   2. waits for host117 RUNNING, its precheck log, and >= MIN_USABLE usable nodes
#   3. one read-only `srun --overlap` container/GPU check on one node
#   4. enqueues the arms (K, ctrl, Kd01, DK03 as the node count allows); ctrl starts at once and
#      keeps host117 busy, K/Kd01/DK03 wait for the surgery tracker
#   5. after every arm has ended (or MAX_H hours), restores handover=true / max_runs_per_host=0.
#      The B1 run on host115 is never signalled.
#
# The K arms wait for the surgery tracker (kvshare_main_surgery_jupiter). The control starts as soon
# as host117 is assigned; run the surgery on the existing small16 hosts before 02:00.
set -uo pipefail

POOL=${POOL:-/e/scratch/e-ext-2025e02-108/oellm_32b_dense_b1/pools/oellm32b_b1_relaxed_20261103}
JOB=${JOB:-2925729}
HOST=${HOST:-host117}
OUR=${OUR_REPO:-/e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp-hybrid}
CTL_REPO=${CTL_REPO:-/e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp}
PY=${PY:-$OUR/.venv/bin/python}
STATE=${STATE:-/e/scratch/e-sta-openeurollm/poeppel1/pool_state}
CFG=${CFG:-experiments/korbi/recall_surgery/kvshare_main_jupiter}
MIN_USABLE=${MIN_USABLE:-385}     # 3 x 128 + the 1-node surgery; below that only K + ctrl
FULL_USABLE=${FULL_USABLE:-513}   # 4 x 128 + the 1-node surgery
SETTINGS_AT=${SETTINGS_AT:-01:30} # local JUPITER time (CET); host StartTime is ~02:09
MAX_H=${MAX_H:-12}
POLL_S=${POLL_S:-30}
STOP=${STOP:-/e/scratch/e-sta-openeurollm/poeppel1/monitor_logs/capture_host117.stop}
log() { echo "$(date '+%F %T') [capture] $*"; }
ctl() { (cd "$CTL_REPO" && PYTHONPATH=. timeout 120 "$PY" scripts/pool_ctl.py "$POOL" "$@"); }
host_usable() { awk '{print $NF}' "$POOL/hosts/$HOST/state" 2>/dev/null || echo 0; }
job_state() { squeue -h -j "$JOB" -o "%T" 2>/dev/null | head -1; }

# 1. protect the host before the B1 monitor can hand it to the 32B run
settings() {
  log "setting B1 pool handover=false max_runs_per_host=5"
  ctl set handover=false max_runs_per_host=5 2>&1 | sed 's/^/  /'
  # queue the control before the host starts: it is the keeper that stops the idle release, and it
  # needs no surgery. host115 is full with the B1 run, so it can only land on host117.
  log "enqueuing the control keeper"
  (cd "$OUR" && PYTHONPATH=. timeout 600 "$PY" scripts/run_autoexp.py --no-monitor \
    --monitor-state-dir "$STATE" --pool "$POOL" --config-name "$CFG" --yes \
    "aux.only=[ctrl]" slurm.sbatch.account=e-sta-openeurollm 2>&1) \
    | grep -E "^Enqueued|Plan:|rror|Traceback" | sed 's/^/  /'
}
settings_done=0
# wait until SETTINGS_AT unless the host is already running
while :; do
  now=$(date +%H:%M)
  st=$(job_state)
  if [ "$st" = "RUNNING" ] || [ "$(date +%s)" -ge "$(date -d "$SETTINGS_AT" +%s 2>/dev/null || echo 0)" ]; then
    settings; settings_done=1; break
  fi
  log "waiting for $SETTINGS_AT or $JOB RUNNING (now $now, state ${st:-?})"
  sleep 60
done

# 2. wait for the host to start and its preflight to finish
t0=$(date +%s)
while :; do
  [ -f "$STOP" ] && { log "STOP file: aborting"; exit 1; }
  st=$(job_state)
  if [ "$st" = "RUNNING" ] && [ -f "$POOL/hosts/$HOST/info.env" ]; then
    u=$(host_usable); u=${u:-0}
    pf=$(ls -1t "$POOL/hosts/$HOST"/precheck-*.log 2>/dev/null | head -1)
    log "host $HOST RUNNING, usable $u / 516, precheck ${pf:--}"
    [ "$u" -ge "$MIN_USABLE" ] && [ -n "$pf" ] && break
  else
    log "host $HOST state ${st:-?}; waiting"
  fi
  [ $(( $(date +%s) - t0 )) -ge $((MAX_H * 3600)) ] && { log "timeout waiting for $HOST"; exit 1; }
  sleep "$POLL_S"
done

# 3. one read-only container / GPU check inside the allocation
log "container/GPU check on one node of job $JOB"
srun --overlap --jobid="$JOB" --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=0 \
  apptainer exec --nv --bind /e \
  /e/project1/e-sta-openeurollm/container/MegatronTraining-JUPITER-te218-fa3_aarch64_202608280932.sif \
  python3 -c 'import torch; print("cuda", torch.cuda.is_available(), torch.cuda.device_count())' \
  2>&1 | sed 's/^/  /' || log "container check failed; continuing (pool preflight already passed)"

# 4. enqueue as many arms as the usable nodes hold; ctrl keeps the host busy
u=$(host_usable); u=${u:-0}
if [ "$u" -ge "$FULL_USABLE" ]; then
  ARMS="K,Kd01,DK03"
elif [ "$u" -ge "$MIN_USABLE" ]; then
  ARMS="K,DK03"
else
  ARMS="K"
fi
log "enqueuing the 1-node surgery, then $ARMS into $POOL (usable $u; control already queued)"
(cd "$OUR" && PYTHONPATH=. timeout 600 "$PY" scripts/run_autoexp.py --no-monitor \
  --monitor-state-dir "$STATE" --pool "$POOL" \
  --config-name experiments/korbi/recall_surgery/kvshare_main_surgery_jupiter --yes \
  slurm.sbatch.account=e-sta-openeurollm job.metadata.pool_priority=90 2>&1) \
  | grep -E "^Enqueued|Plan:|rror|Traceback" | sed 's/^/  /'
(cd "$OUR" && PYTHONPATH=. timeout 600 "$PY" scripts/run_autoexp.py --no-monitor \
  --monitor-state-dir "$STATE" --pool "$POOL" --config-name "$CFG" --yes \
  "aux.only=[$ARMS]" slurm.sbatch.account=e-sta-openeurollm 2>&1) \
  | grep -E "^Enqueued|Plan:|rror|Traceback" | sed 's/^/  /'
log "enqueued; the control keeper is already queued, the surgery and $ARMS follow. K/Kd01/DK03 wait for the surgery tracker."

# 5. restore the B1 pool when every arm has ended (or on timeout / STOP)
deadline=$(( $(date +%s) + MAX_H * 3600 ))
while :; do
  [ -f "$STOP" ] && { log "STOP file: restoring"; break; }
  live=0
  for r in "$POOL"/runs/*; do
    [ -f "$r/meta.env" ] || continue
    name=$(grep -h '^NAME=' "$r/meta.env" | tail -1 | cut -d= -f2 | tr -d "'\"")
    case "$name" in amb84_jsc_K|amb84_jsc_ctrl|amb84_jsc_Kd01|amb84_jsc_DK03|kvshare_main_surgery) ;; *) continue ;; esac
    [ -f "$r/rc" ] || live=$((live + 1))
  done
  log "arms still live: $live"
  [ "$live" -eq 0 ] && break
  [ "$(date +%s)" -ge "$deadline" ] && { log "MAX_H reached: restoring with arms still live"; break; }
  sleep 120
done
log "restoring B1 pool handover=true max_runs_per_host=0"
ctl set handover=true max_runs_per_host=0 2>&1 | sed 's/^/  /'
log "done"
