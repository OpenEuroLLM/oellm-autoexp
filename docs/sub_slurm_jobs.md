# Sub-SLURM jobs: pools of allocations steered from the login node

SLURM controls only the **allocation**. What runs inside it, in which order, and when it
stops is decided by the monitor on the login node, through files on the shared
file system. Two problems go away:

* **queue waits between runs.** Speed ablations, profiling runs and "fix one flag and
  restart" all reuse an allocation that is already up. You can enqueue new variants
  while the current one runs, after looking at its numbers.
* **queue gaps in long training.** A successor allocation waits in the queue with **no
  dependency**, so it builds up age priority (`PriorityWeightAge` on JUPITER; jobs with
  an unsatisfied dependency do not age). When it starts, the run is stopped gracefully
  on the old allocation (a checkpoint is saved) and continues on the new one. The old
  allocation is then released. The only gap is one save plus one startup.

Hands-on introduction: [sub_slurm_jobs_tutorial.md](sub_slurm_jobs_tutorial.md) (ten tasks, nine of them
local against the fake SLURM).

Code: `oellm_autoexp/monitor/pool.py` (monitor side), `scripts/pool/supervisor.sh`
(allocation side), `scripts/pool_ctl.py` (control), `templates/pool_host.sbatch`,
`config/{backend,job}/pool_host.yaml`, `config/slurm/*_pool_host.yaml`.

## Concepts

| | what | identity |
|---|---|---|
| **pool** | a directory plus one monitor session | `<pool>/pool.json` |
| **host** | one SLURM allocation whose batch shell *is* the supervisor | `host001`, ... (record `<pool>.host001`) |
| **job** | an ordinary autoexp job config: its own slurm settings, policy and log | its record in the pool's session |
| **run** | one attempt of a job inside some host: one `srun` step | `<pool>.0007` |

A job keeps **its own complete config**: nodes, time, env, launcher, srun options,
log events, conditions and restart policy. The only thing it shares is the
allocation. When it goes into a pool, its own sbatch template is rendered into a
**step script**:

* the `#SBATCH` header is kept as a comment, documenting what the job asked for;
* node-list variables in the env exports (`SLURM_JOB_NODELIST`,
  `SLURM_JOB_NUM_NODES`, `SLURM_NNODES`) point at the step's nodes, so `MASTER_ADDR`
  and similar settings describe the step, not the allocation;
* `MASTER_PORT` moves with the step index, so a rendezvous never meets the
  TCPStore of the step before it;
* `srun` is pinned to the nodes the supervisor picked
  (`--nodes/--ntasks/--nodelist`) and named after the run (`--job-name=<run id>`).

Each run's status is derived from its directory (`rc` reason, `started`). The
monitor sees it through the normal `JobClientProtocol` (`PoolManager`), so all of
these work unchanged for pool runs:
* `sigterm_rollover`, inactivity and progress events;
* restart hooks, `MaxAttempts` / `MaxActionFires`;
* start, finish and cancel conditions;
* `RestartAction`.

A restart becomes a *new run* in the same pool rather than a new `sbatch`.

## Usage

```bash
# 1. create (or update) a pool from a HOST config; this process becomes its monitor
python scripts/run_autoexp.py --config-name experiments/korbi/pool/jupiter_pool_n2

# 2. put jobs into it, now or later, from any shell on the login node; the live
#    monitor schedules them on its next poll. Every config is resolved, checked
#    against the host (nodes, account, partition) and recorded as usual.
python scripts/run_autoexp.py --pool <pool_dir> --config-name experiments/... [overrides]
#    or set `job.pool_dir: <pool_dir>` in the job's own config

# 3. look and steer
python scripts/pool_ctl.py <pool_dir> status
python scripts/pool_ctl.py <pool_dir> cancel <job|run|name>   # stop a job
python scripts/pool_ctl.py <pool_dir> preempt <run>           # graceful stop + requeue (resume)
python scripts/pool_ctl.py <pool_dir> set keep_pending=1 handover=true
python scripts/pool_ctl.py <pool_dir> exclude <node> ...      # never launch on these again
python scripts/pool_ctl.py <pool_dir> drain | close | open | exit-host <host>
```

`cancel`, `drain`, `close` and `open` are executed by the monitor. `pool_ctl` waits for the result
and prints it (`--wait <s>`, default 120; `--no-wait`), or says right away if no monitor is live.
`cancel` checks first: a target that only matches jobs that already ended, or matches nothing, is
refused with the reason, and nothing is written. A name that matches several active jobs needs
`--all`. Any run id of a job works, including an earlier attempt's. `preempt` accepts only a
RUNNING run.

With `--no-monitor` in steps 1 and 2 nothing waits in the foreground. Attach a monitor
later with `python scripts/monitor_autoexp.py --session-dir <session>`; the session is
in `pool.json`. If no monitor is running, `run_autoexp.py --pool` becomes the monitor.

### The two typical shapes

```yaml
# speed ablations / an interactive queue: one allocation works through the queue and
# stays up while you enqueue more (it is released after idle_exit_s with nothing to do)
job:
  pool_host: {max_hosts_running: 1, keep_pending: 0, min_hosts: 1, idle_exit_s: 1800}

# continuous training: a successor always waits in the queue; when it starts, the run
# moves over (checkpoint) and the old allocation is released
job:
  pool_host: {max_hosts_running: 1, keep_pending: 1, handover: true}
```

Handover has two guards, both added after the first JUPITER run, where the queue was so short
that successors started every 3–4 min and the run ping-ponged between fresh allocations:

* `handover_min_age_s` (default 1800): a host younger than this is not retired. The newer
  host waits for it and is not released as idle in the meantime.
* `successor_delay_s` (default 0): the `keep_pending` successor is queued only once the running
  host is this old. 0 is right for long queues (maximum age priority). For short queues, set it to
  about `wall time - expected queue wait`.

Hosts still waiting in the queue that are no longer needed (`keep_pending` lowered, work gone)
are cancelled.

`max_hosts_running: 2` runs two hosts side by side, each taking the next queued
job. All `job.pool_host` settings are in `PoolHostConfig` (`pool.py`). They are
live: the monitor re-reads `pool.json` every poll and the supervisors re-read
`pool.env` every poll.

## Files (everything is `ls` + `cat`)

```
<pool>/pool.json, pool.env, host_definition.json, state.json, monitor.alive, exclude_nodes
<pool>/ledger.tsv                     one line per run start/end and host start/exit
<pool>/slurm-<jobid>.log              a host's MAIN output (supervisor lines)
<pool>/slurm-<jobid>.<k>-<name>.log   -> the k-th run of that host (symlink to the job's own log)
<pool>/manifests/plan_*.json          provenance of every enqueue (git, args, overrides, plan)
<pool>/requests/                      pool_ctl -> monitor
<pool>/hosts/<hid>/                   info.env alive state exited host.sbatch assign/ control/
<pool>/runs/<rid>/                    step.sh meta.env run.json assigned started rc stop preempt
<job base_output_dir>/slurm-<rid>.log, config-<rid>.yaml   the run's own log and resolved config
```

## Provenance

* **At enqueue** (once per `run_autoexp` invocation, about 0.1 s here), the code state goes into
  the manifest, into every record, and so into each run's `config-<rid>.yaml` as
  `config.metadata.provenance`. `RootConfig.metadata` is free-form, so the file still parses and
  renders the same launch command (tested). The provenance holds the commit and branch, plus a
  `dirty` map of every modified, deleted, renamed or untracked file except gitignored ones,
  each mapped to its git blob hash (`git hash-object <file>`). The same is recorded for every
  git submodule, and a pointer to the manifest, which has the full diff. It is not part of the
  config the job id is hashed from. Normal (non-pool) jobs get the same block.
* **At launch**, the supervisor records the code the run is about to use in
  `<pool>/runs/<rid>/launch_provenance.json` and compares it with the enqueue state. A job can
  wait in the pool queue for hours, and a sync or an edit in between would otherwise go unseen.
  The result (`code unchanged since enqueue` or `CODE CHANGED SINCE ENQUEUE (n): ...`) goes into
  the ledger, the host log and the run log, and `pool_ctl status` flags changed runs. It never
  blocks a run: no python or a timeout only writes `provenance: unavailable`.

## Lifecycle details

* **Stopping a run gracefully** means SIGTERM to the *rank* processes, the direct
  children of the torchrun / ft_launcher agents, on the step's nodes, sent through an
  `--overlap` step. This is the same relay as `templates/jupiter_relay.sbatch`.
  Megatron's `--exit-signal-handler` then saves and exits 0. A run without launcher
  processes (a bash job) gets the signal through `scancel --signal=TERM <step>`.
  After the grace period (`stop_grace_s` for a cancel, `preempt_grace_s` for a
  preemption) the step is killed.
* **PREEMPTED** (handover, `pool_ctl preempt`, the host's wall clock, a lost host)
  is requeued automatically, up to `max_attempts`. Megatron's
  "exiting program after receiving SIGTERM." usually also fires the job's own
  `sigterm_rollover`. That RestartAction is deferred until the run has ended and
  then resumed in place of the built-in requeue, so a run is never restarted
  twice (tested).
* **The host's wall clock.** Use `#SBATCH --signal=B:USR1@<margin>`, which the
  `*_pool_host` slurm configs set. The supervisor is `exec`'d, so it *is* the batch
  shell, sees the signal, preempts the run, and releases the allocation. The run
  goes to the next host.
* **A job's own `sbatch.time`** is enforced per run: the run is stopped and reads
  TIMEOUT, which is final.
* **Lost host** (node failure, `scancel`, a stale supervisor heartbeat): the run
  reads `host_lost` (PREEMPTED) and is requeued. It resumes from the last persistent
  checkpoint.
* **A bad node inside a live host.** Exclude it (`pool_ctl exclude <node>`, or the job's restart
  hook appends it to its `restart_exclude_file`). The next run of the job skips it, so a host
  with a spare node keeps working. A host that can no longer fit ANY queued job, because too
  few usable nodes or too little time is left, is released AT ONCE rather than after
  `idle_exit_s`, and replaced. Every new host's `--exclude` is the union of the pool's
  `exclude_nodes` and the host config's own exclusion file, read when the host is created. A
  queued job bigger than every host of the pool, which only happens if the host definition
  changed after enqueue, triggers a warning instead of a loop of short-lived allocations.
* **Fleet-wide exclusion is opt-in.** `jupiter_autoexclude_pool_host` (the
  `jupiter_autoexclude` group: `$JUPITER_EXCLUDE_NODES`) is for big or long pools. The plain
  `jupiter_pool_host` (small eval pools) only avoids the pool's own `exclude_nodes`. Choose it
  with `slurm=jupiter_autoexclude_pool_host` or in the host config's defaults.
* **A monitor that dies costs no GPU-hours.** An idle host whose `monitor.alive` is
  older than `monitor_dead_exit_s` releases itself. Busy hosts finish their run. A
  new monitor adopts the running pool.

## Compute accounting

A pool host costs its full allocation time, whether busy or idle. The submit gate
prices the HOST config. Enqueued jobs are not priced again, because they consume
the host's allocation. Idle time is bounded by `idle_exit_s` (monitor) and
`monitor_dead_exit_s` (supervisor).

## Testing locally

`tests/fakeslurm` is a small local SLURM (`sbatch`, `squeue`, `sacct`, `scancel`,
`srun`, `scontrol`). It models:
* `--signal=B:`, time limits, held jobs, and step node contention;
* killing every step's processes at the end of a job, like proctrack/cgroup.

The real `SlurmClient`, supervisor and step scripts run against it unchanged:

```bash
PATH=$PWD/tests/fakeslurm/bin:$PATH FAKESLURM_DIR=$PWD/dump/fs POOL_TEST_ROOT=$PWD/dump/pooltest \
  python scripts/run_autoexp.py --config-name experiments/tests/pool_local_host --no-monitor
.venv/bin/python -m pytest tests/pool   # unit (fast) + end-to-end scenarios (~7 min)
```

The end-to-end tests run `scripts/sigterm_dummy_trainer.py` (Megatron's
SIGTERM-save-exit and resume semantics) under a fake torchrun agent. Like the real
agent, it kills its workers when *it* receives the signal, so a graceful stop
passes only if the supervisor signals the ranks.

## Validating on a cluster

1. **Phase 0 probe** (`scripts/pool/phase0_probe.sbatch`, bash only, 2 nodes, at most 6 min).
   It checks the SLURM mechanics the supervisor relies on:
   * the same rendezvous port across consecutive steps;
   * step-relative `SLURM_NODEID` on a node subset;
   * finding a step by name, an `--overlap` relay step next to a full step, and
     `scancel --signal` on a step;
   * a prompt next step;
   * `B:USR1` reaching the batch shell.

   ```bash
   ssh <cluster> 'mkdir -p <dir> && cd <dir> && sbatch --account=<acc> --partition=<part>' < scripts/pool/phase0_probe.sbatch
   grep PROBE <dir>/pool_phase0_probe-<jobid>.log
   ```
   JUPITER, 2026-10-26 (job 2518446, Slurm 25.05):
   * every mechanic PASSED: port reuse across steps, step-relative node ids on a subset,
     named-step lookup, an overlap step starting next to a full-CPU step (1 s), the next step
     starting 1 s after a signalled one, and `B:USR1` reaching the batch shell;
   * `scancel --signal=TERM <step>` reaches every process of the step, so a task can die of the
     signal before its trap runs, but the step ends cleanly (rc 0).
   Real pool, JUPITER 2026-10-26 (130M Megatron, 2-node hosts, mcore 0.19 c9cf7e84c, te218-fa3):
   * three jobs ran back to back in one allocation (2 nodes, then 1 node);
   * `pool_ctl preempt`: SIGTERM went to all 8 ranks, Megatron saved at iteration 190 and exited
     rc 0; the job was requeued in the SAME allocation 10 s later and loaded iteration 190;
   * handover with a training run moved twice across allocations (570 -> 572, 635 -> 637),
     and each old allocation released itself (COMPLETED, not TIMEOUT);
   * found the ping-pong described above, which is why the two guards exist.
2. **Real pool** with the 130M Megatron smoke (2 nodes, 300 iterations, a checkpoint
   every 50):
   ```bash
   python scripts/run_autoexp.py --no-monitor --config-name experiments/korbi/pool/jupiter_pool_n2
   POOL=$OUTPUT_DIR/pools/jupiter_pool_n2
   python scripts/run_autoexp.py --no-monitor --pool $POOL --config-name experiments/korbi/pool/megatron_130M_pool_smoke aux.additional_tag=a
   python scripts/run_autoexp.py --no-monitor --pool $POOL --config-name experiments/korbi/pool/megatron_130M_pool_smoke aux.additional_tag=b slurm.sbatch.nodes=1
   python scripts/monitor_autoexp.py --session-dir <session from pool.json>   # in tmux
   # while run a trains:  python scripts/pool_ctl.py $POOL preempt <run of a>
   #   expect: 'exiting program after receiving SIGTERM' + a checkpoint, then a requeued run of
   #   job a in the SAME allocation that loads that checkpoint; then b on 1 node.
   # handover:            python scripts/pool_ctl.py $POOL set keep_pending=1 handover=true
   #   expect: a second allocation queued; when it starts, the run moves over and the first
   #   allocation is released.
   ```

## Known limits (deliberately not in v1)

* One run per host at a time. No packing of small jobs side by side into one
  allocation.
* A job's allocation-level settings (account, partition, qos, reservation) must
  match the host's, and `--exclusive`/`--signal` are the host's. The enqueue check
  rejects a mismatch.
* `$SLURM_JOB_ID` inside a run is the HOST's id. A wandb name built from it is shared
  by every run of the same job name in one host; `$AUTOEXP_RUN_ID` is per run.
* The handover does not overlap the successor's startup with the predecessor's
  training.
* Megatron needs no change: the stop goes through the existing exit signal handler.
