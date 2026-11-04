#!/bin/bash
{sbatch_directives}

{env_exports}

# WAIT_FOR_JOB_NAME (optional): start only once no job of that name runs any more -- a run that
# takes over another one's checkpoints (e.g. 32 -> 128 nodes) must not load before the other's
# final save is written
if [ -n "$WAIT_FOR_JOB_NAME" ]; then
    while squeue -u "$USER" -h -n "$WAIT_FOR_JOB_NAME" -t R,CG -o %i | grep -q .; do
        echo "waiting for $WAIT_FOR_JOB_NAME to end"; sleep 30
    done
fi

# --- Watchdog: one per node, on the host, next to the training step ---------------
# GPU memory trace + hang kill (no iteration for WATCHDOG_STALL_S, or none within
# WATCHDOG_STARTUP_S): py-spy stacks of every rank, then scancel.
# scripts/korbi/hybrid_speed/job_watchdog.py; both steps use --overlap.
echo "Launching watchdog on $SLURM_NNODES node(s)"
srun --ntasks-per-node=1 --nodes=$SLURM_NNODES --overlap --cpu-bind=none python3 $WATCHDOG_SCRIPT &
WATCHDOG_PID=$!
# -------------------------------------------------------------------------------

# NODE_CHECK (off | light | thorough; unset = off): the node-health benchmark
# (oellm_autoexp/node_health/bench.py --profile ..., ~1-2 / ~4-5 min) on THIS step's nodes before
# the training, in the training's container. Pass: the training starts. Bad nodes (exit 3) in a
# pool step: they go into this host's <host>/exclude_nodes (honoured by the host's supervisor;
# never pushed into SLURM jobs) and the step exits 86, so the run fails, its job's restart policy
# requeues it and it is placed on other nodes -- at most NODE_CHECK_MAX_ROUNDS (default 2) times
# per NODE_CHECK_DIR, then the training starts anyway. Outside a pool, or a broken check (other
# exit codes): reported, never blocks the training. NODE_CHECK_ARGS: more bench.py flags.
if [ -n "$NODE_CHECK" ] && [ "$NODE_CHECK" != "off" ]; then
    NC_DIR=$NODE_CHECK_DIR
    [ -n "$NC_DIR" ] || NC_DIR=$HOME/node_check
    mkdir -p "$NC_DIR"
    NC_TAG=$AUTOEXP_RUN_ID
    [ -n "$NC_TAG" ] || NC_TAG=$SLURM_JOB_ID
    export NC_OUT=$NC_DIR/check_$NC_TAG.json NC_PORT=$(( MASTER_PORT + 101 )) NODE_CHECK NODE_CHECK_ARGS
    echo "[node_check] $NODE_CHECK check on $SLURM_NNODES node(s) before the training -> $NC_OUT"
    srun {srun_opts} bash -c '{launcher_cmd} torchrun --nnodes $SLURM_NNODES --nproc-per-node 4 --node-rank $SLURM_NODEID --master-addr $MASTER_ADDR --master-port $NC_PORT -m oellm_autoexp.node_health.bench --profile $NODE_CHECK --check-out $NC_OUT $NODE_CHECK_ARGS'
    NC_RC=$?
    NC_MAX=$NODE_CHECK_MAX_ROUNDS
    [ -n "$NC_MAX" ] || NC_MAX=2
    NC_ROUNDS=$(cat "$NC_DIR/exclusion_rounds" 2>/dev/null || echo 0)
    if [ "$NC_RC" = 0 ]; then
        echo "[node_check] passed"
        rm -f "$NC_DIR/exclusion_rounds"
    elif [ "$NC_RC" = 3 ]; then
        NC_BAD=$(python3 -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['exclude']))" "$NC_OUT" 2>/dev/null)
        if [ -n "$NC_BAD" ] && [ -n "$AUTOEXP_POOL_DIR" ] && [ -n "$AUTOEXP_POOL_HOST" ] && [ "$NC_ROUNDS" -lt "$NC_MAX" ]; then
            printf '%s\n' $NC_BAD >> "$AUTOEXP_POOL_DIR/hosts/$AUTOEXP_POOL_HOST/exclude_nodes"
            echo $(( NC_ROUNDS + 1 )) > "$NC_DIR/exclusion_rounds"
            echo "[node_check] excluded on host $AUTOEXP_POOL_HOST: $NC_BAD (round $(( NC_ROUNDS + 1 )) of $NC_MAX); restarting on other nodes"
            exit 86
        fi
        echo "[node_check] bad node(s) $NC_BAD, not excluded (outside a pool or round $NC_ROUNDS of $NC_MAX): training anyway"
    else
        echo "[node_check] the check itself failed (rc $NC_RC): training anyway"
    fi
fi

echo "Launching job {job_name}"
srun {srun_opts}bash -c '{launcher_cmd} {command}'
RC=$?

kill $WATCHDOG_PID 2>/dev/null; wait $WATCHDOG_PID 2>/dev/null || true
echo "[SLURM] JOB_ID:   $SLURM_JOB_ID"
echo "[SLURM] NODELIST: $SLURM_NODELIST"
echo "[SLURM] NNODES:   $SLURM_NNODES"
exit $RC
