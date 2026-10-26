#!/bin/bash
# Pool host supervisor: runs in the BATCH SHELL of a pool host allocation
# (templates/pool_host.sbatch) and executes the runs the monitor assigns to this
# host, one `srun` step at a time. It decides nothing about WHAT to run: the
# monitor on the login node (oellm_autoexp/monitor/pool.py) writes assignments,
# this script runs them and reports through files. Layout: see pool.py.
#
#   supervisor.sh <pool_dir> <host_id>
#
# Files it reads:   <pool>/pool.env (re-sourced every poll, so settings are live)
#                   <host>/assign/<rid>          a run to execute, oldest rid first
#                   <host>/control/exit          preempt the current run, release the allocation
#                   <host>/control/drain         finish the current run, then release
#                   <run>/stop | <run>/preempt   stop one run (cancel / graceful requeue)
#                   <pool>/exclude_nodes, <host>/exclude_nodes, the run's EXCLUDE
# Files it writes:  <host>/info.env alive state exited, <run>/started rc,
#                   <pool>/ledger.tsv, <pool>/slurm-<jobid>.<k>-<name>.log -> run log
#
# Stopping a run gracefully = SIGTERM to the RANK processes (the children of the
# torchrun / ft_launcher agents) on the step's nodes, through an --overlap step;
# Megatron's --exit-signal-handler then saves a checkpoint and exits 0. A run
# without such processes (a bash job) gets the signal through `scancel` on its
# step instead. After the grace period the step is killed.
#
# SIGUSR1 / SIGTERM on the batch shell (`#SBATCH --signal=B:USR1@<margin>`, or
# the wall clock) preempt the current run the same way and end the host.

set -u

POOL=$(cd "$1" && pwd) || { echo "[pool-host] no pool dir: $1" >&2; exit 2; }
HID=$2
H="$POOL/hosts/$HID"
# the checkout the runs' step scripts use (this script lives in <repo>/scripts/pool/)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$H/assign" "$H/control"

log() { echo "[pool-host $HID] $(date +%FT%T) $*"; }
now() { date +%s; }
atomic() { # atomic <path> <content>
  local tmp
  tmp="$(dirname "$1")/.$(basename "$1").$$.tmp"
  printf '%s\n' "$2" >"$tmp" && mv -f "$tmp" "$1"
}
ledger() { printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$HID" "${SLURM_JOB_ID:-}" "$1" "$2" >>"$POOL/ledger.tsv"; }

load_cfg() {
  POOL_POLL_S=10 POOL_IDLE_EXIT_S=1200 POOL_MONITOR_DEAD_EXIT_S=1800
  POOL_STOP_GRACE_S=120 POOL_PREEMPT_GRACE_S=900
  POOL_RANK_PATTERN='^[^ ]*python[^ ]*( +-[A-Za-z]+)* +(-m +torch\.distributed\.run|[^ ]*/(torchrun|ft_launcher))( |$)'
  # shellcheck disable=SC1091
  [ -f "$POOL/pool.env" ] && . "$POOL/pool.env"
  export POOL_RANK_PATTERN
}

expand_hosts() { scontrol show hostnames "$1" 2>/dev/null; }

ALL_NODES=$(expand_hosts "$SLURM_JOB_NODELIST")
NUM_NODES=$(printf '%s\n' "$ALL_NODES" | grep -c .)

# usable nodes in allocation order, minus every exclusion source
usable_nodes() { # usable_nodes <extra exclude list>
  local excl
  excl=$(
    {
      cat "$POOL/exclude_nodes" "$H/exclude_nodes" 2>/dev/null
      printf '%s\n' "$1"
    } | tr ', ' '\n\n' | grep -v '^$' | while read -r e; do expand_hosts "$e"; done | sort -u
  )
  if [ -z "$excl" ]; then
    printf '%s\n' "$ALL_NODES"
  else
    printf '%s\n' "$ALL_NODES" | grep -vxF -f <(printf '%s\n' "$excl")
  fi
}

set_state() { atomic "$H/state" "$* $(now) $(usable_nodes "" | grep -c .)"; }

# the srun step of a run, found by its name (= run id)
step_of() { squeue -h -s -j "$SLURM_JOB_ID" -o "%i %j" 2>/dev/null | awk -v n="$1" '$2==n {print $1; exit}'; }
signal_step() { # signal_step <SIG> <rid>
  local id
  id=$(step_of "$2")
  if [ -n "$id" ]; then
    log "scancel --signal=$1 $id"
    scancel --signal="$1" "$id" 2>&1
  fi
}

graceful_stop() { # graceful_stop <rid> <nodelist> <nnodes>
  local out n
  out=$(srun --overlap --jobid="$SLURM_JOB_ID" --nodelist="$2" --nodes="$3" --ntasks="$3" \
    --ntasks-per-node=1 --cpus-per-task=1 --job-name="relay-$1" \
    bash -c 'n=0; for a in $(pgrep -u "$(id -u)" -f "$POOL_RANK_PATTERN"); do for w in $(pgrep -P "$a"); do kill -TERM "$w" && n=$((n+1)); done; done; echo "[relay] $(hostname): SIGTERM to $n rank process(es)"' 2>&1)
  printf '%s\n' "$out"
  n=$(printf '%s\n' "$out" | sed -n 's/.*SIGTERM to \([0-9][0-9]*\) rank.*/\1/p' | awk '{s+=$1} END {print s+0}')
  if [ "${n:-0}" -eq 0 ]; then
    log "run $1: no launcher rank processes found, signalling the step"
    signal_step TERM "$1"
  fi
}

# Which code this run is about to use, compared with what it was enqueued with
# (oellm_autoexp/utils/provenance.py, stdlib only). Writes <run>/launch_provenance.json
# and prints one line. Never blocks the run: no python or a timeout just says so.
launch_provenance() { # launch_provenance <rid>
  local py
  py=$(command -v python3 || command -v python) || { echo "provenance: unavailable (no python)"; return; }
  timeout 90 "$py" "$REPO/oellm_autoexp/utils/provenance.py" --repo "$REPO" \
    --out "$POOL/runs/$1/launch_provenance.json" --compare "$POOL/runs/$1/run.json" 2>&1 | tail -n 1
}

finish_run() { # finish_run <rid> <code> <reason>
  local R="$POOL/runs/$1"
  [ -e "$R/rc" ] || atomic "$R/rc" "$2 $3"
  ledger "$1" "end $2 $3"
  rm -f "$H/assign/$1"
  log "run $1 ended: rc=$2 ($3)"
}

STEP_INDEX=0
run_one() { # run_one <rid>
  local rid=$1
  local R="$POOL/runs/$rid"
  local NODES=1 NTASKS_PER_NODE=1 MAX_RUNTIME_S=0 LOG="$R/run.log" NAME=run EXCLUDE=""
  if [ -e "$R/rc" ]; then
    rm -f "$H/assign/$rid"
    return
  fi
  # shellcheck disable=SC1091
  . "$R/meta.env"
  if [ -e "$R/stop" ]; then
    finish_run "$rid" 0 cancelled
    return
  fi
  local nl
  nl=$(usable_nodes "$EXCLUDE" | head -n "$NODES" | paste -sd, -)
  if [ "$(printf '%s' "$nl" | tr ',' '\n' | grep -c .)" -lt "$NODES" ]; then
    log "run $rid needs $NODES node(s), only $(usable_nodes "$EXCLUDE" | grep -c .) usable"
    finish_run "$rid" -1 insufficient_nodes
    return
  fi
  STEP_INDEX=$((STEP_INDEX + 1))
  export AUTOEXP_RUN_ID="$rid" AUTOEXP_STEP_INDEX="$STEP_INDEX" AUTOEXP_STEP_NODELIST="$nl" \
    AUTOEXP_STEP_NUM_NODES="$NODES" AUTOEXP_STEP_NTASKS=$((NODES * NTASKS_PER_NODE)) \
    AUTOEXP_STEP_NTASKS_PER_NODE="$NTASKS_PER_NODE" AUTOEXP_POOL_DIR="$POOL" AUTOEXP_POOL_HOST="$HID"
  mkdir -p "$(dirname "$LOG")"
  ln -sfn "$LOG" "$POOL/slurm-$SLURM_JOB_ID.$STEP_INDEX-$NAME.log"
  local t0 t
  t0=$(now)
  atomic "$R/started" "host=$HID
job=${SLURM_JOB_ID}
step_index=$STEP_INDEX
nodes=$nl
ts=$t0"
  ledger "$rid" "start step=$STEP_INDEX nodes=$nl"
  log "run $rid ($NAME) starting as step $STEP_INDEX on $NODES node(s): $nl"
  echo "[pool] $(date +%FT%T) run $rid starting on host $HID (job $SLURM_JOB_ID, step index $STEP_INDEX), nodes $nl" >>"$LOG"
  local prov
  prov=$(launch_provenance "$rid")
  log "run $rid: $prov"
  echo "[pool] $prov" >>"$LOG"
  ledger "$rid" "$prov"
  bash "$R/step.sh" >>"$LOG" 2>&1 &
  local pid=$!
  set_state busy "$rid"
  local reason="" grace=0 stop_at=0 kill_at=0 killed=0
  while kill -0 "$pid" 2>/dev/null; do
    touch "$H/alive"
    t=$(now)
    if [ -z "$reason" ]; then
      load_cfg
      if [ -e "$R/stop" ]; then
        reason=cancelled grace=$POOL_STOP_GRACE_S
      elif [ -e "$R/preempt" ] || [ -e "$H/control/exit" ] || [ -n "$HOST_SIGNAL" ]; then
        reason=preempted grace=$POOL_PREEMPT_GRACE_S
      elif [ "$MAX_RUNTIME_S" -gt 0 ] && [ $((t - t0)) -ge "$MAX_RUNTIME_S" ]; then
        reason=timeout grace=$POOL_STOP_GRACE_S
      fi
      if [ -n "$reason" ]; then
        log "run $rid: stopping ($reason), grace ${grace}s"
        echo "[pool] $(date +%FT%T) stopping run $rid ($reason)" >>"$LOG"
        graceful_stop "$rid" "$nl" "$NODES"
        stop_at=$((t + grace))
      fi
    elif [ "$killed" -eq 0 ] && [ "$t" -ge "$stop_at" ]; then
      log "run $rid: grace period over, killing the step"
      signal_step KILL "$rid"
      killed=1 kill_at=$((t + 60))
    elif [ "$killed" -eq 1 ] && [ "$t" -ge "$kill_at" ]; then
      log "run $rid: step still there, killing its launcher shell"
      kill -KILL "$pid" 2>/dev/null
      killed=2
    fi
    # sleep in the background + wait: a trapped signal interrupts the wait
    sleep "$POOL_POLL_S" &
    wait $!
  done
  wait "$pid"
  local rc=$?
  if [ -z "$reason" ]; then
    if [ "$rc" -eq 0 ]; then reason=completed; else reason=failed; fi
  fi
  echo "[pool] $(date +%FT%T) run $rid ended: rc=$rc ($reason)" >>"$LOG"
  finish_run "$rid" "$rc" "$reason"
}

exit_host() {
  log "releasing the allocation: $1"
  atomic "$H/exited" "$1"
  set_state exited
  ledger "-" "host exit: $1"
  exit 0
}

HOST_SIGNAL=""
trap 'HOST_SIGNAL=USR1; log "SIGUSR1 received"' USR1
trap 'HOST_SIGNAL=TERM; log "SIGTERM received"' TERM

load_cfg
START_TS=$(now)
atomic "$H/info.env" "SLURM_JOB_ID=${SLURM_JOB_ID:-}
NUM_NODES=$NUM_NODES
NODELIST=$(printf '%s\n' "$ALL_NODES" | paste -sd, -)
START_TS=$START_TS
BATCH_HOST=$(hostname)"
touch "$H/alive"
set_state idle
ledger "-" "host start nodes=$NUM_NODES"
log "up: job ${SLURM_JOB_ID:-?}, $NUM_NODES node(s), pool $POOL"

idle_since=$(now)
while true; do
  touch "$H/alive"
  load_cfg
  [ -n "$HOST_SIGNAL" ] && exit_host "signal $HOST_SIGNAL"
  [ -e "$H/control/exit" ] && exit_host "exit requested: $(head -n1 "$H/control/exit")"
  next=$(ls -1 "$H/assign" 2>/dev/null | grep -v '^\.' | sort | head -n1)
  if [ -n "$next" ]; then
    run_one "$next"
    set_state idle
    idle_since=$(now)
    continue
  fi
  [ -e "$H/control/drain" ] && exit_host "drain requested: $(head -n1 "$H/control/drain")"
  # dead man's switch: nobody will assign work to an idle host whose monitor
  # went silent, so it releases itself (idle_exit_s is the MONITOR's decision)
  t=$(now)
  mon=$(stat -c %Y "$POOL/monitor.alive" 2>/dev/null || echo "$START_TS")
  if [ $((t - mon)) -ge "$POOL_MONITOR_DEAD_EXIT_S" ] && [ $((t - idle_since)) -ge "$POOL_POLL_S" ]; then
    exit_host "idle for $((t - idle_since))s and no monitor tick for $((t - mon))s"
  fi
  sleep "$POOL_POLL_S" &
  wait $!
done
