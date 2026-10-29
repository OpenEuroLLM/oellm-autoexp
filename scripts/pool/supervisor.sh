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
  POOL_STOP_GRACE_S=120 POOL_PREEMPT_GRACE_S=900 POOL_MAX_RUNS_PER_HOST=1
  POOL_RANK_PATTERN='^[^ ]*python[^ ]*( +-[A-Za-z]+)* +(-m +torch\.distributed\.run|[^ ]*/(torchrun|ft_launcher))( |$)'
  # shellcheck disable=SC1091
  [ -f "$POOL/pool.env" ] && . "$POOL/pool.env"
  export POOL_RANK_PATTERN
}

expand_hosts() { scontrol show hostnames "$1" 2>/dev/null; }

# An adopted host's supervisor is a 1-node, 1-cpu step started from outside: take the JOB's
# nodes, and drop the step-level variables srun would read as input for the run steps it
# starts (MareNostrum 5 sets SLURM_MEM_PER_CPU: "SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and
# SLURM_MEM_PER_NODE are mutually exclusive"), so they see what a batch shell would.
if [ -n "${AUTOEXP_POOL_ADOPTED_JOB:-}" ]; then
  SLURM_JOB_NODELIST=$(squeue -h -j "$AUTOEXP_POOL_ADOPTED_JOB" -o %N 2>/dev/null)
  export SLURM_JOB_NODELIST SLURM_JOB_ID="$AUTOEXP_POOL_ADOPTED_JOB"
  unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE SLURM_GRES SLURM_GPUS \
    SLURM_GPUS_PER_NODE SLURM_GPUS_PER_TASK SLURM_GPUS_ON_NODE SLURM_TRES_PER_TASK \
    SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_CPUS_PER_TASK \
    SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE \
    SLURM_DISTRIBUTION SLURM_STEP_ID SLURM_STEPID SLURM_STEP_NODELIST SLURM_STEP_NUM_NODES \
    SLURM_STEP_NUM_TASKS SLURM_STEP_TASKS_PER_NODE SLURM_STEP_GPUS SLURM_NODELIST \
    SLURM_JOB_CPUS_PER_NODE SLURM_EXCLUSIVE CUDA_VISIBLE_DEVICES
  SLURM_NNODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | grep -c .)
  export SLURM_NNODES SLURM_JOB_NUM_NODES=$SLURM_NNODES SLURM_NODELIST=$SLURM_JOB_NODELIST
fi
ALL_NODES=$(expand_hosts "$SLURM_JOB_NODELIST")
NUM_NODES=$(printf '%s\n' "$ALL_NODES" | grep -c .)

# usable nodes in allocation order, minus every exclusion source
# Nodes of this allocation SLURM itself reports unusable: down, drained/draining, failing,
# maintenance, unknown, or not responding (`*`). With `#SBATCH --no-kill` the allocation
# survives a node failure; this keeps the next run off the dead node before any fault
# scan has named it (one spare node per host is then enough to keep running).
sick_nodes() {
  sinfo -h -N -n "$SLURM_JOB_NODELIST" -o "%N %t" 2>/dev/null |
    awk '$2 ~ /(down|drain|drng|fail|maint|unk|npc|\*)/ {print $1}' | sort -u
}

usable_nodes() { # usable_nodes <extra exclude list>
  local excl
  excl=$(
    {
      cat "$POOL/exclude_nodes" "$H/exclude_nodes" 2>/dev/null
      printf '%s\n' "$1"
    } | tr ', ' '\n\n' | grep -v '^$' | while read -r e; do expand_hosts "$e"; done
    sick_nodes
  )
  excl=$(printf '%s\n' "$excl" | grep -v '^$' | sort -u)
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

graceful_stop() { # graceful_stop <rid> <nodelist> <nnodes> [<signal>]
  # <signal>: the run's STOP_SIGNAL (meta.env), the job's own Megatron --exit-signal
  local out n sig=${4:-TERM}
  # --mem=0: a full step holds all of a node's memory where SLURM allots memory per CPU
  # (MareNostrum 5: "Memory required by task is not available" for any explicit request);
  # --mem=0 is the one an --overlap step may share (phase-0 probe, 2026-10-27)
  # only this run's ranks (AUTOEXP_RUN_ID in their environment; packed runs may share a machine),
  # all launchers on the node if none carries it (a --cleanenv container drops it)
  out=$(POOL_STOP_SIG="$sig" POOL_STOP_RID="$1" srun --overlap --mem=0 --jobid="$SLURM_JOB_ID" --nodelist="$2" --nodes="$3" --ntasks="$3" \
    --ntasks-per-node=1 --cpus-per-task=1 --job-name="relay-$1" \
    bash -c 'all=$(pgrep -u "$(id -u)" -f "$POOL_RANK_PATTERN"); mine=""; for a in $all; do tr "\\0" "\\n" </proc/$a/environ 2>/dev/null | grep -qxF "AUTOEXP_RUN_ID=$POOL_STOP_RID" && mine="$mine $a"; done; n=0; for a in ${mine:-$all}; do for w in $(pgrep -P "$a"); do kill -"$POOL_STOP_SIG" "$w" && n=$((n+1)); done; done; echo "[relay] $(hostname): SIG$POOL_STOP_SIG to $n rank process(es)"' 2>&1)
  printf '%s\n' "$out"
  n=$(printf '%s\n' "$out" | sed -n 's/.*SIG[A-Z0-9]* to \([0-9][0-9]*\) rank.*/\1/p' | awk '{s+=$1} END {print s+0}')
  if [ "${n:-0}" -eq 0 ]; then
    log "run $1: no launcher rank processes found, signalling the step"
    signal_step "$sig" "$1"
  fi
}

# Which code this run is about to use, compared with what it was enqueued with
# (oellm_autoexp/utils/provenance.py, stdlib only). Writes <run>/launch_provenance.json
# and prints one line. Never blocks the run: no python or a timeout just says so.
launch_provenance() { # launch_provenance <rid>
  local py
  py=$(command -v python3 || command -v python) || { echo "provenance: unavailable (no python)"; return; }
  timeout 90 "$py" "$REPO/oellm_autoexp/utils/provenance.py" --repo "${WORKDIR:-$REPO}" \
    --out "$POOL/runs/$1/launch_provenance.json" --compare "$POOL/runs/$1/run.json" 2>&1 | tail -n 1
}

finish_run() { # finish_run <rid> <code> <reason>
  local R="$POOL/runs/$1"
  [ -e "$R/rc" ] || atomic "$R/rc" "$2 $3"
  ledger "$1" "end $2 $3"
  rm -f "$H/assign/$1"
  log "run $1 ended: rc=$2 ($3)"
}

# continue the step numbering of an earlier supervisor of this allocation (a replaced adopted
# supervisor): its <pool>/slurm-<job>.<step>-<name>.log links must stay unique
STEP_INDEX=$(ls "$POOL"/slurm-"${SLURM_JOB_ID:-x}".*-*.log 2>/dev/null | sed -n "s#.*/slurm-${SLURM_JOB_ID:-x}\.\([0-9][0-9]*\)-.*#\1#p" | sort -n | tail -n 1)
STEP_INDEX=${STEP_INDEX:-0}
# run_one <rid> <nodelist> <step index>: one run on the nodes the main loop picked for it, in the
# background (several runs side by side on disjoint nodes when POOL_MAX_RUNS_PER_HOST > 1). The
# main loop owns the host state; a host-wide stop reaches the run through $H/control/signal.
run_one() {
  local rid=$1 nl=$2 STEP_INDEX=$3
  local R="$POOL/runs/$rid"
  local NODES=1 NTASKS_PER_NODE=1 MAX_RUNTIME_S=0 LOG="$R/run.log" NAME=run EXCLUDE="" STOP_SIGNAL=TERM WORKDIR=""
  # shellcheck disable=SC1091
  . "$R/meta.env"
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
  local reason="" grace=0 stop_at=0 kill_at=0 killed=0
  while kill -0 "$pid" 2>/dev/null; do
    touch "$H/alive"
    t=$(now)
    if [ -z "$reason" ]; then
      load_cfg
      adopt_deadline_check
      if [ -e "$R/stop" ]; then
        reason=cancelled grace=$POOL_STOP_GRACE_S
      elif [ -e "$R/preempt" ] || [ -e "$H/control/exit" ] || [ -e "$H/control/signal" ]; then
        reason=preempted grace=$POOL_PREEMPT_GRACE_S
      elif [ "$MAX_RUNTIME_S" -gt 0 ] && [ $((t - t0)) -ge "$MAX_RUNTIME_S" ]; then
        reason=timeout grace=$POOL_STOP_GRACE_S
      fi
      if [ -n "$reason" ]; then
        log "run $rid: stopping ($reason), grace ${grace}s"
        echo "[pool] $(date +%FT%T) stopping run $rid ($reason)" >>"$LOG"
        graceful_stop "$rid" "$nl" "$NODES" "$STOP_SIGNAL"
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
  # adopted: let the held step 0 go (rc 0), so the job ends COMPLETED
  [ -n "$ADOPTED" ] && atomic "$POOL/adopt/$ADOPTED.release" "$(now) $1"
  exit 0
}

HOST_SIGNAL=""
trap 'HOST_SIGNAL=USR1; touch "$H/control/signal"; log "SIGUSR1 received"' USR1
trap 'HOST_SIGNAL=TERM; touch "$H/control/signal"; log "SIGTERM received"' TERM

# ADOPTED host (pool_ctl adopt): an ordinary queued job, not a pool host, whose own training
# step 0 is held by the adoption hook (scripts/pool/adopt_sitecustomize.py). This supervisor was
# started INTO that allocation as an --overlap step from the login node, so every run step must
# overlap step 0 too (and take no memory of its own: step 0 holds it all), and there is no
# B:USR1: the runs are stopped POOL_ADOPT_STOP_MARGIN_S before the job's own end time.
ADOPTED=${AUTOEXP_POOL_ADOPTED_JOB:-}
JOB_START_TS=""
END_TS=""
if [ -n "$ADOPTED" ]; then
  export SLURM_OVERLAP=1 SLURM_MEM_PER_NODE=0
  ADOPT_DIR="$POOL/adopt"
  read -r _start _end < <(squeue -h -j "$ADOPTED" -o "%S %e" 2>/dev/null)
  JOB_START_TS=$(date -d "${_start:-now}" +%s 2>/dev/null || now)
  END_TS=$(date -d "${_end:-}" +%s 2>/dev/null || echo "")
  atomic "$ADOPT_DIR/$ADOPTED.attached" "$(now) $(hostname) $HID"
fi
adopt_deadline_check() {
  [ -n "$ADOPTED" ] && [ -n "$END_TS" ] && [ -z "$HOST_SIGNAL" ] || return 0
  if [ "$(now)" -ge $((END_TS - ${POOL_ADOPT_STOP_MARGIN_S:-1800})) ]; then
    HOST_SIGNAL=USR1
    touch "$H/control/signal"
    log "adopted job $ADOPTED ends at $(date -d @"$END_TS" +%FT%T): stopping before its wall clock"
  fi
}

load_cfg
START_TS=$(now)
atomic "$H/info.env" "SLURM_JOB_ID=${SLURM_JOB_ID:-}
NUM_NODES=$NUM_NODES
NODELIST=$(printf '%s\n' "$ALL_NODES" | paste -sd, -)
START_TS=$START_TS
JOB_START_TS=${JOB_START_TS:-$START_TS}
ADOPTED=$ADOPTED
BATCH_HOST=$(hostname)"
touch "$H/alive"
set_state idle
ledger "-" "host start nodes=$NUM_NODES"
log "up: job ${SLURM_JOB_ID:-?}, $NUM_NODES node(s), pool $POOL"

# runs of this host: rid -> pid of its background run_one; the nodes of each in $H/inuse/<rid>
declare -A RUN_PID=()
mkdir -p "$H/inuse"
rm -f "$H/control/signal" "$H"/inuse/*
free_nodes() { # free_nodes <extra exclude>: usable nodes no running run holds
  local held
  held=$(cat "$H"/inuse/* 2>/dev/null | tr ',' '\n' | grep -v '^$')
  if [ -z "$held" ]; then
    usable_nodes "$1"
  else
    usable_nodes "$1" | grep -vxF -f <(printf '%s\n' "$held")
  fi
}

# A run with USE_EXCLUDED_NODES=1 (job.metadata.pool_use_excluded_nodes, e.g. CPU-only work) may
# also use the nodes the pool excludes for bad GPUs / NICs -- those first, so it takes no healthy
# GPU node it does not need. Nodes SLURM reports sick and the run's own EXCLUDE stay off-limits.
excluded_ok_nodes() { # excluded_ok_nodes <extra exclude>: pool-excluded, healthy to SLURM, free
  local listed off held
  listed=$(cat "$POOL/exclude_nodes" "$H/exclude_nodes" 2>/dev/null | tr ', ' '\n\n' | grep -v '^$' |
    while read -r e; do expand_hosts "$e"; done | sort -u)
  [ -n "$listed" ] || return 0
  off=$(
    printf '%s\n' "$1" | tr ', ' '\n\n' | grep -v '^$' | while read -r e; do expand_hosts "$e"; done
    sick_nodes
    cat "$H"/inuse/* 2>/dev/null | tr ',' '\n'
  )
  off=$(printf '%s\n' "$off" | grep -v '^$' | sort -u)
  printf '%s\n' "$ALL_NODES" | grep -xF -f <(printf '%s\n' "$listed") |
    { if [ -n "$off" ]; then grep -vxF -f <(printf '%s\n' "$off"); else cat; fi; }
}

run_free_nodes() { # run_free_nodes <extra exclude> <use excluded (1 / empty)>: candidates in order
  if [ -n "$2" ]; then
    { excluded_ok_nodes "$1"; free_nodes "$1"; } | grep -v '^$' | awk '!seen[$0]++'
  else
    free_nodes "$1"
  fi
}
reap() {
  local rid
  for rid in "${!RUN_PID[@]}"; do
    if ! kill -0 "${RUN_PID[$rid]}" 2>/dev/null; then
      wait "${RUN_PID[$rid]}" 2>/dev/null
      unset "RUN_PID[$rid]"
      rm -f "$H/inuse/$rid"
    fi
  done
}
LAST_STATE=""
show_state() { # only on a change: the monitor measures idle time from the state's timestamp
  local want=idle
  [ "${#RUN_PID[@]}" -gt 0 ] && want="busy $(printf '%s\n' "${!RUN_PID[@]}" | sort | paste -sd, -)"
  [ "$want" = "$LAST_STATE" ] && return
  LAST_STATE=$want
  # shellcheck disable=SC2086
  set_state $want
}

idle_since=$(now)
while true; do
  touch "$H/alive"
  load_cfg
  adopt_deadline_check
  reap
  stopping=""
  [ -n "$HOST_SIGNAL" ] && stopping="signal $HOST_SIGNAL"
  [ -z "$stopping" ] && [ -e "$H/control/exit" ] && stopping="exit requested: $(head -n1 "$H/control/exit")"
  if [ -n "$stopping" ]; then
    # every run sees the stop (control/exit or control/signal) and preempts itself
    [ "${#RUN_PID[@]}" -eq 0 ] && exit_host "$stopping"
    show_state
    sleep "$POOL_POLL_S" &
    wait $!
    continue
  fi
  launched=0
  for rid in $(ls -1 "$H/assign" 2>/dev/null | grep -v '^\.' | sort); do
    [ -n "${RUN_PID[$rid]:-}" ] && continue
    [ "${#RUN_PID[@]}" -ge "${POOL_MAX_RUNS_PER_HOST:-1}" ] && break
    R="$POOL/runs/$rid"
    if [ -e "$R/rc" ]; then
      rm -f "$H/assign/$rid"
      continue
    fi
    if [ -e "$R/stop" ]; then
      finish_run "$rid" 0 cancelled
      continue
    fi
    meta=$(. "$R/meta.env"; printf '%s|%s|%s' "${NODES:-1}" "${USE_EXCLUDED_NODES:-}" "${EXCLUDE:-}")
    need=${meta%%|*}
    rest=${meta#*|}
    use_excl=${rest%%|*}
    run_excl=${rest#*|}
    nl=$(run_free_nodes "$run_excl" "$use_excl" | head -n "$need" | paste -sd, -)
    if [ "$(printf '%s' "$nl" | tr ',' '\n' | grep -c .)" -lt "$need" ]; then
      if [ "${#RUN_PID[@]}" -eq 0 ]; then
        log "run $rid needs $need node(s), only $(run_free_nodes "$run_excl" "$use_excl" | grep -c .) usable"
        finish_run "$rid" -1 insufficient_nodes
      fi
      continue  # (packed: it starts once another run frees its nodes)
    fi
    STEP_INDEX=$((STEP_INDEX + 1))
    atomic "$H/inuse/$rid" "$nl"
    run_one "$rid" "$nl" "$STEP_INDEX" &
    RUN_PID[$rid]=$!
    launched=1
  done
  show_state
  if [ "${#RUN_PID[@]}" -gt 0 ] || [ "$launched" -eq 1 ]; then
    idle_since=$(now)
    sleep "$POOL_POLL_S" &
    wait $!
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
