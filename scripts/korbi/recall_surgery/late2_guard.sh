#!/bin/bash
# Guard of the C / M / L continuation on host108 of the 32B pool (docs/recall_surgery_ambitious.md §14.7; user
# 2026-11-15: an 18-node host of the 32B pool for small31; never the 32B run's 516-node host, whose spare nodes are the
# 32B run's failure buffer). The pool code cannot pin a job to a host (newest host first, packing via
# max_runs_per_host), so every POLL_S seconds: any run of ours (NAME small31_* / s31ro_*) placed on a host other than
# ALLOWED_HOST is cancelled at once (pool_ctl cancel <run>) and re-enqueued in jupiter_pool_small16. When the last
# readout (C / M / L at DONE_STEP) is done, or on a STOP file, max_runs_per_host is restored (OLD_MAX_RUNS) and the guard
# exits. Never touches the 32B run or any Slurm job. JUPITER login node:
#   nohup bash scripts/korbi/recall_surgery/late2_guard.sh > /e/scratch/e-sta-openeurollm/poeppel1/monitor_logs/late2_guard.log 2>&1 &
set -uo pipefail
P=${POOL:-/e/scratch/e-ext-2025e02-108/oellm_32b_dense_b1/pools/oellm32b_b1_relaxed_20261103}
CTL_REPO=${CTL_REPO:-/e/project1/e-sta-openeurollm/poeppel1/Projects/oellm-autoexp}  # the 32B pool's own code
OUR=${OUR_REPO:-/e/home/jusers/poeppel1/jupiter/work/Projects/oellm-autoexp-hybrid}
PY=${PY:-$OUR/.venv/bin/python}
SMALL16=${SMALL16:-/e/scratch/e-sta-openeurollm/poeppel1/pools/jupiter_pool_small16}
STATE=${STATE:-/e/scratch/e-sta-openeurollm/poeppel1/pool_state}
ROOT=${ROOT:-/e/scratch/e-sta-openeurollm/poeppel1/recall_surgery/small31}
ALLOWED_HOST=${ALLOWED_HOST:-host108}
OLD_MAX_RUNS=${OLD_MAX_RUNS:-0}
POLL_S=${POLL_S:-15}
DONE_STEP=${DONE_STEP:-41961}  # the guard ends once the readouts of ARMS at this step are complete (third leg: 44822)
ARMS=${ARMS:-C M L}
REENQUEUE_TRAIN=${REENQUEUE_TRAIN:-1}  # 0: a misplaced training run is only cancelled (its leg's config is ambiguous)
STOP=${STOP:-$ROOT/late2_guard.stop}
log() { echo "$(date '+%F %T') [late2_guard] $*"; }
ctl() { (cd "$CTL_REPO" && PYTHONPATH=. timeout 120 "$PY" scripts/pool_ctl.py "$P" "$@"); }
done_all() {
  for a in $ARMS; do
    d=$ROOT/evals/small31_${a}_$DONE_STEP
    [ -f "$d/done/ruler_4096.ok" ] && [ -f "$d/valid_loss.json" ] && [ "$(ls "$d/done" | wc -l)" -ge 7 ] || return 1
  done
}
reenqueue() {  # $1 = NAME of the cancelled run
  local name=$1 cfg only
  case "$name" in
    s31ro_*) only=$(echo "$name" | sed -E 's/^s31ro_([A-Za-z0-9]+)_([0-9]+).*/\1_\2/'); cfg=small31_readout_jupiter ;;
    small31_*) only=$(echo "$name" | sed -E 's/^small31_([A-Za-z0-9]+)_.*/\1/'); cfg=${TRAIN_CFG:-small31_phase2late2_jupiter} ;;
  esac
  (cd "$OUR" && PYTHONPATH=. timeout 600 "$PY" scripts/run_autoexp.py --no-monitor --monitor-state-dir "$STATE" \
    --pool "$SMALL16" --config-name "experiments/korbi/recall_surgery/$cfg" --yes "aux.only=[$only]" 2>&1 \
    | grep -E "^Enqueued|rror" | cut -c1-160)
  log "re-enqueued $only ($cfg) in jupiter_pool_small16"
}
log "start: pool $P, allowed host $ALLOWED_HOST, done at the $DONE_STEP readouts of $ARMS, restore max_runs_per_host=$OLD_MAX_RUNS at the end"
declare -A seen
while true; do
  for rd in "$P"/runs/*; do
    [ -f "$rd/rc" ] && continue
    [ -f "$rd/started" ] && [ -f "$rd/meta.env" ] || continue
    name=$(grep -h '^NAME=' "$rd/meta.env" | tail -1 | cut -d= -f2 | tr -d "'\"")
    case "$name" in small31_*|s31ro_*) ;; *) continue ;; esac
    host=$(grep -h '^host=' "$rd/started" | cut -d= -f2)
    rid=$(basename "$rd")
    if [ "$host" != "$ALLOWED_HOST" ]; then
      log "WRONG HOST: $rid $name on $host -> cancel"
      ctl cancel "$rid" 2>&1 | tail -1
      case "$name" in
        small31_*) [ "$REENQUEUE_TRAIN" = 1 ] && reenqueue "$name" || log "training run $name: cancelled, not re-enqueued" ;;
        *) reenqueue "$name" ;;
      esac
    elif [ -z "${seen[$rid]:-}" ]; then
      seen[$rid]=1; log "ok: $rid $name on $host"
    fi
  done
  if done_all || [ -f "$STOP" ]; then
    ctl set max_runs_per_host="$OLD_MAX_RUNS" 2>&1 | tail -1
    log "done (readouts complete or STOP): max_runs_per_host restored to $OLD_MAX_RUNS"
    exit 0
  fi
  sleep "$POLL_S"
done
