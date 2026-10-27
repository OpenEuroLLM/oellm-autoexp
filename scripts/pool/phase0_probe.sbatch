#!/bin/bash
#SBATCH --job-name=pool_phase0_probe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --time=00:06:00
#SBATCH --signal=B:USR1@150
#SBATCH --output=pool_phase0_probe-%j.log
# Phase-0 probe for sub-SLURM-job pools (docs/sub_slurm_jobs.md): the SLURM mechanics the pool
# supervisor relies on, on the real cluster, with nothing but bash + the system python3.
#   ssh <cluster> 'mkdir -p <dir> && cd <dir> && sbatch --account=<acc> --partition=<part>' < scripts/pool/phase0_probe.sbatch
# Each check prints "PROBE <name>: PASS|FAIL <detail>".
set -u
ok() { echo "PROBE $1: PASS ${2:-}"; }
bad() { echo "PROBE $1: FAIL ${2:-}"; }
USR1_TS=""
trap 'USR1_TS=$(date +%s); echo "[probe] batch shell got SIGUSR1 at $(date +%T)"' USR1
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "[probe] job $SLURM_JOB_ID on ${NODES[*]} (slurm $(sinfo --version 2>/dev/null))"
PORT=20074
PY=$(command -v python3 || command -v python)
echo "[probe] python: $PY"

# 1. two consecutive full-allocation steps bind the same TCP port on node 0
for i in 1 2; do
  out=$(srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=288 --job-name=bind$i bash -c \
    "if [ \$SLURM_NODEID = 0 ]; then $PY -c 'import socket,time; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((\"0.0.0.0\", $PORT)); s.listen(); print(\"bound\"); time.sleep(2)'; else sleep 2; echo idle; fi" 2>&1)
  echo "$out" | grep -q bound && ok "same_port_step_$i" || bad "same_port_step_$i" "$(echo "$out" | tr '\n' ' ')"
done

# 2. a step on a SUBSET of the nodes (the second node only): node ids are step-relative
out=$(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="${NODES[1]}" --job-name=subset bash -c \
  'echo "nodeid=$SLURM_NODEID nnodes=$SLURM_NNODES stepnodes=$SLURM_STEP_NODELIST jobnodes=$SLURM_JOB_NODELIST host=$(hostname)"' 2>&1)
echo "[probe] subset step: $out"
echo "$out" | grep -q "nodeid=0 nnodes=1" && ok subset_step_relative_ids "$out" || bad subset_step_relative_ids "$out"

# 3. a named step is findable with squeue -s, an --overlap step runs NEXT to it, and
#    scancel --signal=TERM <step> reaches its task
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=288 --job-name=probe.0001 bash -c \
  'trap "echo \"[task $SLURM_NODEID] got SIGTERM\"; exit 0" TERM; echo "[task $SLURM_NODEID] up"; while true; do sleep 1; done' > step3.out 2>&1 &
spid=$!
sleep 8
step=$(squeue -h -s -j "$SLURM_JOB_ID" -o "%i %j" | awk '$2=="probe.0001" {print $1}')
[ -n "$step" ] && ok named_step_found "$step" || bad named_step_found "$(squeue -h -s -j "$SLURM_JOB_ID" -o '%i %j' | tr '\n' ' ')"
t0=$(date +%s)
ov=$(timeout 60 srun --overlap --jobid="$SLURM_JOB_ID" --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 \
  --nodelist="${NODES[0]},${NODES[1]}" --job-name=relay bash -c 'echo "relay on $(hostname): $(pgrep -u $(id -u) -fc "while true")"' 2>&1)
t1=$(date +%s)
echo "[probe] overlap step ($((t1 - t0)) s): $(echo "$ov" | tr '\n' ' ')"
[ "$(echo "$ov" | grep -c 'relay on')" = 2 ] && [ $((t1 - t0)) -lt 30 ] && ok overlap_step_next_to_full_step || bad overlap_step_next_to_full_step "$ov"
[ -n "$step" ] && scancel --signal=TERM "$step"
wait "$spid"; rc=$?
cat step3.out
# the signal goes to EVERY process of the step (the tasks' `sleep` children too), so a task may
# die of it before its trap prints; what the supervisor needs is that the step ENDS
[ "$(grep -c 'got SIGTERM' step3.out)" -ge 1 ] && [ -z "$(squeue -h -s -j "$SLURM_JOB_ID" -o '%j' | grep -x probe.0001)" ] \
  && ok scancel_step_signal "step ended, rc=$rc" || bad scancel_step_signal "rc=$rc"

# 4. after a step that was signalled, the next full step starts right away
t0=$(date +%s)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=288 --job-name=after true
t1=$(date +%s)
[ $((t1 - t0)) -lt 30 ] && ok next_step_starts_promptly "$((t1 - t0)) s" || bad next_step_starts_promptly "$((t1 - t0)) s"

# 5. B:USR1 reaches the batch shell (150 s before the 6 min limit = at ~3.5 min)
while [ -z "$USR1_TS" ]; do sleep 1 & wait $!; done
ok batch_shell_usr1 "at $(date -d @"$USR1_TS" +%T)"
echo "[probe] done"
