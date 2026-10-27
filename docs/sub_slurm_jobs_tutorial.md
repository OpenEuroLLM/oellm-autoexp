# Sub-SLURM-job pools: a hands-on tutorial

Ten tasks, from cheap to real, for learning how pools behave. Read
[sub_slurm_jobs.md](sub_slurm_jobs.md) for the reference. Tasks 1-9 run locally against
`tests/fakeslurm`, a small local SLURM: they take seconds to minutes and cost nothing. Task 10
runs on a cluster. Each task has a goal, the commands, and questions; answers are at the end.

## The model in one picture

```
pool ─┬─ hosts (allocations)          host001, host002, ...   one SLURM job each
      └─ jobs  (your configs)         dummy_a, train, ...      one monitor record each
                └─ runs (attempts)    localpool.0007, ...      one run = one job × one host
```

* A **host** runs runs one after another, and they may belong to different jobs.
* A **job** belongs to the pool, not to a host: its runs may land on different hosts (handover,
  lost host).
* SLURM only knows about hosts. The monitor on the login node decides which run goes where.
* A host is never "restarted". A dead host is closed and, if there is work, replaced by a new
  allocation.

## Setup

In **every** terminal you use, from the repository root:

```bash
source scripts/pool/tutorial_env.sh
```

It must print `sbatch -> .../tests/fakeslurm/bin/sbatch`. A terminal without it has no `sbatch`:
the monitor then cannot start any host. It logs one clear error and backs off (30 s, doubling),
and `ledger.tsv` and the host logs never appear.

It defines `mkpool`, `enq`, `mon`, `ctl` and `crash`, and the variables `$R`, `$POOL`, `$MS`,
`$SESSION`, `$FAKESLURM_DIR`. Two fake-SLURM hooks the tasks use:

* `FAKESLURM_HOLD=1` in the monitor's environment keeps new allocations PENDING until you
  `rm $FAKESLURM_DIR/jobs/<jobid>/hold`;
* `crash <jobid>` simulates the allocation's nodes dying (SIGKILL, no cleanup).

`squeue` works as usual, because the fake one is first on `PATH`. Start over at any time with
`rm -rf $R`.

> **Use a fresh `aux.tag` per experiment.** The tag is part of `job.name`, and the same
> `job.name` means the same output and checkpoint directory. A reused tag RESUMES the old
> checkpoint. If that run had finished, the "new" run loads the final checkpoint and ends
> after about a second with `[after training is done]`. This is the mechanism of Task 3, and an
> easy way to fool yourself.

## Task 1: Anatomy of a pool

```bash
mkpool
enq aux.tag=a; enq aux.tag=b
mon        # 2nd terminal (after sourcing the env there); let it finish (~1 min)
```

Then read:
* `ls $POOL`, `cat $POOL/pool.json $POOL/ledger.tsv`;
* `cat $POOL/slurm-*.log`, the host's MAIN output;
* `ls -l $POOL/slurm-*.*-*.log`, the per-run symlinks ("sub-named" outputs);
* `cat $POOL/runs/localpool.0001/step.sh`;
* `ls $R/runs/dummy_a/`.

1. Where did the job's own `#SBATCH` header go? And where is the header that SLURM saw?
2. Which files prove which config ran, and from which code state?
3. When exactly does the monitor exit by itself?

Two directories called `runs` are easy to mix up. `$POOL/runs/<rid>/` is the pool's bookkeeping
for one attempt. `$R/runs/dummy_a/` is the JOB's output directory: its log, resolved config and
checkpoints.

## Task 2: A live queue

```bash
enq aux.tag=long aux.iters=400 aux.tick=0.1
mon &                                    # or in the 2nd terminal
enq aux.tag=c; enq aux.tag=d
enq aux.tag=urgent +job.metadata.pool_priority=5
ctl status                               # repeat a few times
```

In which order do `c`, `d` and `urgent` run after `long`? Why does none of them start a new
allocation?

## Task 3: Switch a flawed run without losing the allocation

Pretend `fix` has a wrong setting. Once it has saved a checkpoint, cut over to the fixed version.
Use the SAME tag for both, and a tag not used before.

```bash
enq aux.tag=fix aux.iters=400 aux.tick=0.1
# wait until its log ($R/runs/dummy_fix/slurm-*.log) shows "saved checkpoint from iteration 10" or later
enq aux.tag=fix aux.iters=400 aux.tick=0.1 +job.metadata.pool_priority=10   # the "fixed" one (job id ..._q2)
ctl cancel <job id of the OLD one, from ctl status>
```

`ctl cancel` prints what the monitor did (`monitor: cancelled [...]`). If the old job had already
ended, it refuses (`nothing to cancel: ... already ended (finished at HH:MM:SS)`) and writes
nothing. Then you were too late, and the fixed job just loads the final checkpoint.

1. Does the fixed job start from 0 or from the checkpoint, and why?
2. What changes if you give the fix a new tag?
3. Why does the priority matter?

## Task 4: Preempt and requeue

```bash
enq aux.tag=p aux.iters=300 aux.tick=0.1
ctl status                               # note its run id; preempt needs a RUNNING run
ctl preempt <run id>
```

Watch the run's log for `exiting program after receiving SIGTERM.`, then for the next run's
`loading checkpoint ... at iteration N`. In the job's record (`$SESSION/dummy_p_*.job.json`)
look at `attempts`.

1. What does preempt stop: the run, the job, or the allocation?
2. The job's own `sigterm_rollover` policy and the pool's built-in requeue both react to this
   exit. Why do you still get exactly ONE new run?

   Hint: `observe_once` in `oellm_autoexp/monitor/loop.py`, in the order of
   `_resume_deferred_restart`, `_pool_status_effect`, `_process_log_events`. Then
   `grep -E "deferred until|running the deferred restart|was preempted: requeueing" $SESSION/monitor.log`.

## Task 5: Handover rehearsal

```bash
ctl set keep_pending=1 handover=true handover_min_age_s=20
# restart the monitor as:  FAKESLURM_HOLD=1 mon
enq aux.tag=train aux.iters=1500 aux.tick=0.1
squeue                                   # two allocations pending
rm $FAKESLURM_DIR/jobs/<first id>/hold   # the first allocation starts
# when train is past iteration 100:
rm $FAKESLURM_DIR/jobs/<second id>/hold
```

1. What happens in the first 20 s after the second allocation is up, and what happens after?
2. Which allocation ends COMPLETED, and when?
3. Why is a third allocation suddenly pending?
4. Repeat with `ctl set successor_delay_s=60`. When is the successor queued now?
5. Why do `handover_min_age_s` and `successor_delay_s` exist? (See "Handover has two guards" in
   the reference.)

## Task 6: The wall clock

```bash
ctl close                                # let the current hosts go
mkpool slurm.sbatch.time=00:00:40 slurm.sbatch.signal=B:USR1@15   # new host definition
ctl open
enq aux.tag=wall aux.iters=600 aux.tick=0.1
mon
```

Follow it in `ledger.tsv`.

1. How many allocations does it take?
2. Does any of them end as TIMEOUT?
3. Why does `templates/pool_host.sbatch` `exec` the supervisor? Remove the `exec` and run
   `.venv/bin/python -m pytest tests/pool -k wall_clock` to see.

## Task 7: Breaking things

1. **Node death during a run.**
   ```bash
   enq aux.tag=crash1 aux.iters=600 aux.tick=0.1
   # wait until RUNNING and past iteration 25 (checkpoint every 10)
   crash <the RUNNING host's job id from squeue>
   ```
   What does the run's `rc` say? Where and from which iteration does the job continue? What do
   you lose compared with a preempt? A crash of an IDLE host loses nothing and shows almost
   nothing: the host log just stops.
2. **A bad node in a live host.** The local hosts have 2 nodes and the dummy jobs use 1. While
   a run trains:
   ```bash
   ctl status        # its run's node: $POOL/runs/<rid>/started, "nodes="
   ctl exclude <that node>; ctl preempt <rid>
   enq aux.tag=big slurm.sbatch.nodes=2
   ```
   * Where does the preempted job continue?
   * What happens to the host once the 1-node job is done and only `big` (2 nodes) is queued?
   * Look at `$POOL/hosts/host00*/exited` and at the next host's `host.sbatch` (`--exclude=`).
3. **Dead monitor.** Kill the monitor with `kill -9` while a run trains.
   * Does the run continue? Restart `mon`: does it pick the pool up again?
   * Kill it again while the host is idle, and read `$POOL/hosts/*/exited` after
   `monitor_dead_exit_s`.
4. **Too big.** Run `enq aux.tag=huge slurm.sbatch.nodes=4`. When is it rejected?
5. **No `sbatch`.** Start `mon` in a terminal WITHOUT the env. What does the monitor log say,
   and how often does it try?

## Task 8: Live steering

Try these while things run:

```bash
ctl set idle_exit_s=5
ctl drain
ctl close
ctl open
ctl exit-host host00X
ctl cancel <something finished>
ctl preempt <a queued run>
```

1. Which act through the monitor (`$POOL/requests/`, the result printed when it is done), and
   which act directly on files the supervisor reads? See `scripts/pool_ctl.py`.
2. What does `ctl cancel --no-wait` change, and what is printed if no monitor is running?

## Task 9: Read the two halves

`PoolManager._tick_pool` in `oellm_autoexp/monitor/pool.py`, and `run_one` in
`scripts/pool/supervisor.sh`.

1. A run's status is derived from which files, and who writes each of them?
2. What does the monitor do if a host's `alive` file is 20 min old while SLURM says RUNNING?
3. Where is the job's own `sbatch.time` enforced?
4. Open a run's `config-<rid>.yaml` (`metadata.provenance`) and `$POOL/runs/<rid>/launch_provenance.json`.
   What do they tell you, and when would `CODE CHANGED SINCE ENQUEUE` appear in the ledger?

Then add a scenario of your own to `tests/pool/test_pool_e2e.py`, for example two hosts in
parallel with `max_hosts_running=2`.

## Task 10: The real thing (small: a few GPU-hours)

On the cluster, in a clone that has this code, with the cluster's environment. On JUPITER:
`source ~/work/venv/bin/activate`, `export PYTHONPATH=. SLURM_ACCOUNT=<acc> OUTPUT_DIR=<dir>`, and
an `HF_HOME` whose `gpt-neox-20b` cache is complete.

```bash
MS=$OUTPUT_DIR/monitor_state
python scripts/run_autoexp.py --no-monitor --monitor-state-dir $MS --config-name experiments/korbi/pool/jupiter_pool_n2 \
    job.name=speed_pool job.pool_host.min_hosts=1 job.pool_host.idle_exit_s=1800
for mbs in 8 16; do
  python scripts/run_autoexp.py --no-monitor --monitor-state-dir $MS --pool $OUTPUT_DIR/pools/speed_pool \
    --config-name experiments/korbi/pool/megatron_130M_pool_smoke aux.additional_tag=mbs$mbs \
    backend.megatron.micro_batch_size=$mbs backend.megatron.train_iters=100 backend.megatron.save_interval=1000000
done
python scripts/monitor_autoexp.py --session-dir $MS/pool_speed-pool --poll-interval 30   # in tmux
```

When the first numbers are in, enqueue a third variant you pick from them (TP2, 1 node, ...).
Compare tok/s/GPU across the run logs, count the queue waits you paid, and finish with
`pool_ctl close`. For large or long pools, add `slurm=jupiter_autoexclude_pool_host` when
creating the pool, so that every host avoids the fleet-wide exclusion list.

## Answers

<details><summary>Task 1</summary>

1. The job's own header is kept, commented out, at the top of `$POOL/runs/<rid>/step.sh`. What
   SLURM saw is the HOST's script: `$POOL/hosts/<hid>/host.sbatch`, built from the host config.
2. `config-<rid>.yaml` next to the run's log. It is the fully resolved config, and it still
   parses and renders the same command. Its `metadata.provenance` holds the commit and the
   blob hash of every modified or untracked file, for the repository and each submodule, plus the
   path of the enqueue manifest (`$POOL/manifests/plan_*.json`: git diff, args, overrides).
   `$POOL/runs/<rid>/launch_provenance.json` holds the code the run actually started with.
3. When no ACTIVE record is left, and host records count. So it exits only after the last host
   has released itself (`idle_exit_s`), not when the last job finishes. With `min_hosts: 1` it
   runs until `pool_ctl close`. It then writes `.monitor.stop` "all jobs finished", which the
   next `mon` on a pool session clears by itself.
</details>

<details><summary>Task 2</summary>

`long`, then `urgent`, `c`, `d`: priority first, then enqueue time. All of them run in the
allocation that is already up: one host runs one job at a time, and the monitor only submits a
new host when the desired number of hosts is not reached.
</details>

<details><summary>Task 3</summary>

1. From the checkpoint: the same `job.name` means the same checkpoint directory. The cancel sent
   SIGTERM to the ranks, so the old run saved before it exited.
2. A new tag means a new directory, so it starts from 0.
3. When the cancelled run frees the host, the host takes the highest-priority, then oldest,
   queued job. Without the priority, older queued work would run before the fix. With more than
   one host (`max_hosts_running: 2`), CANCEL FIRST: an idle second host could otherwise start
   the fix while the old run still writes into the same checkpoint directory.
</details>

<details><summary>Task 4</summary>

1. The RUN. The job stays active: a new run of it starts, and loads the checkpoint the stopped
   run saved. The allocation keeps running.
2. Whichever reacts first switches the job to a new run id, so the other looks at a run the job
   no longer points to. Usually the log line arrives first: `sigterm_rollover` is DEFERRED until
   the run has ended (`wait_for_job_end`) and then resumed (line ~270), before the built-in
   PREEMPTED requeue is reached (~274). If the monitor first sees the run already ended, the
   built-in requeue wins, and the old run's log line is never read (the new run's log is read
   from 0). The monitor log shows `restart deferred ...` + `running the deferred restart` in the
   first case, and `was preempted: requeueing` in the second.
</details>

<details><summary>Task 5</summary>

1. For 20 s nothing: the young host keeps its run, and the successor waits without being
   released as idle. Then the old host is told to exit: its run is preempted (checkpoint) and
   requeued, and it starts on the new host.
2. The OLD allocation, right after its run has saved and exited, released by its own supervisor:
   COMPLETED, not TIMEOUT.
3. `keep_pending: 1`: a new successor, again without a dependency, so it ages.
4. Only once the running host is 60 s old.
5. With a short queue, successors start within minutes. Without the guards, every start would
   preempt the run again and queue the next successor: the run ping-pongs between fresh
   allocations and spends its time in startup. This was seen on JUPITER, 2026-10-26.
</details>

<details><summary>Task 6</summary>

1. Three or four allocations in a chain.
2. None: `B:USR1@15` reaches the supervisor, which stops the run gracefully and releases the
   allocation before the limit. The run continues on the next host.
3. `--signal=B:...` signals only the batch shell's own process. As a child, the supervisor would
   never see it, and the untrapped batch shell would die of USR1: no save, and the run reads
   `host_lost`.
</details>

<details><summary>Task 7</summary>

1. `-1 host_lost` (PREEMPTED). The job is requeued on a new host, from the last PERIODIC
   checkpoint (a multiple of 10). Unlike a preempt, the iterations since that save are lost.
2. Its next run uses the other node of the SAME allocation, because the host has a spare. When
   only the 2-node job is left, the degraded host cannot fit it and is released AT ONCE
   (`exited`: "cannot run any queued job (needs 2 node(s), 1 usable)"), not after
   `idle_exit_s`. The replacement's `host.sbatch` has `--exclude=<bad node>`: every new host
   avoids the pool's `exclude_nodes`, plus the fleet list for `*_autoexclude_pool_host`.
3. The run continues, and a new monitor picks the pool up (`Rehydrated ...`). An idle host
   releases itself once the monitor has been silent for `monitor_dead_exit_s`: "no monitor
   tick for ...".
4. At enqueue, before anything is written: "needs 4 nodes, the pool's hosts have 2".
5. `Unable to submit ...: No such file or directory: 'sbatch'`, then
   `host ... ended without ever starting ... Failed starts in a row: N`. It retries after 30 s,
   60 s, 120 s, ... (at most 10 min), not on every poll.
</details>

<details><summary>Task 8</summary>

1. Through the monitor: `cancel`, `drain`, `close`, `open`. `pool_ctl` waits for the result
   (`--wait`, default 120 s) and prints it. Directly on files: `preempt` (`runs/<rid>/preempt`),
   `exit-host` (`hosts/<hid>/control/exit`), `set` (`pool.json`, `pool.env`), `exclude`
   (`exclude_nodes`). `cancel` of something finished, and `preempt` of anything not RUNNING, are
   refused up front.
2. `--no-wait` only writes the request. Without a live monitor (no tick for 5 min) `pool_ctl`
   says so right away. The request then waits until a monitor attaches.
</details>

<details><summary>Task 9</summary>

1. `assigned` (monitor), `started` and `rc` (supervisor). `rc` is written by the monitor
   (`-1 host_lost`) when the host is gone. `stop` and `preempt` come from the monitor or
   `pool_ctl`.
2. It cancels the host (`scancel`), marks it `exited` ("stale heartbeat"), and the run reads
   `host_lost` and is requeued.
3. In the supervisor (`MAX_RUNTIME_S` from the job's `sbatch.time`): graceful stop, then TIMEOUT,
   which is final.
4. What code the job was planned with and what it ran with. A difference, from a sync or edit
   while the job waited in the queue, gives `CODE CHANGED SINCE ENQUEUE (n): <files>` in the
   ledger, the host log and the run log, and is flagged in `pool_ctl status`.
</details>
