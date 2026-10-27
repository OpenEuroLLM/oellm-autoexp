# Operating pools: continuous large runs

The manual for whoever runs pools in production: a person, or an agent. The goal is that
training **keeps running**:
* allocations queue ahead and build up age priority;
* runs move from allocation to allocation;
* a job is restarted with a fix without a new allocation;
* node failures are handled automatically, by a restart in the same allocation or by a switch to
  the next one.

Short overlaps of two allocations are accepted in exchange for continuity.

* Concepts and reference: [sub_slurm_jobs.md](sub_slurm_jobs.md)
* Hands-on tasks (local, fake SLURM): [sub_slurm_jobs_tutorial.md](sub_slurm_jobs_tutorial.md)

```
pool ─┬─ hosts (allocations)   one SLURM job each; its batch shell IS the supervisor
      └─ jobs  (your configs)  one monitor record each, with their own log policy
                └─ runs        one attempt of a job on one host = one srun step
```

## 1. Choosing a setup

| Use case | Host config | Key settings |
|---|---|---|
| Continuous training, fast restarts (JUPITER) | `experiments/korbi/pool/jupiter_pool_continuous` | `keep_pending: 1`, `handover: true`, autoexclude, no spare node |
| Continuous training, long queues / slow restarts (for example MareNostrum) | the same, on that cluster's `*_pool_host` slurm config | **one spare node**: `slurm.sbatch.nodes=<job nodes + 1>` and `+slurm.sbatch.no_kill=true` |
| Speed ablations, interactive queue | `experiments/korbi/pool/jupiter_pool_n2` | `keep_pending: 0`, `min_hosts: 1`, longer `idle_exit_s` |
| Small eval jobs | the plain `jupiter_pool_host` slurm config | no autoexclude needed; the pool's own `exclude_nodes` still applies |

**Why one spare node.** Without `--no-kill`, SLURM ends the whole allocation when one node fails.
The run then continues only once the next allocation starts, which with long queues can take
hours. With `--no-kill` and one spare node:
1. the allocation survives the failure;
2. the job's restart policy restarts the run;
3. the supervisor launches the new run on the remaining nodes. It skips any node SLURM reports
   down, drained, failing or not responding, plus every excluded node.

Restart time is then one startup, not a queue wait. The price is one idle node per allocation.
A second failure in the same allocation leaves too few nodes. The host is then released at once
and the run continues on the successor.

**`jupiter_pool_continuous` at a glance** (all live-editable with `pool_ctl set`):

| Setting | Value | Why |
|---|---|---|
| `keep_pending` | 1 | a successor always waits in the queue, without a dependency, so it ages |
| `handover` | true | a newer live allocation takes over the run, and the old one is released |
| `handover_min_age_s` | 3600 | no switch before the old allocation has run 1 h. With short queues, successors start within minutes; without this the run would ping-pong. |
| `successor_delay_s` | 0 | queue the successor at once: maximum age. Raise it only if the queue is short. |
| `preempt_grace_s` | 1800 | must exceed the checkpoint save time, with margin. Measure it. |
| `min_remaining_s` | 1800 | no run starts on an allocation with less left: startup plus one save interval |
| `host_stale_s` | 1800 | a RUNNING host whose supervisor heartbeat is older than this is cancelled |
| `max_attempts` | 20 | UNHEALTHY requeues only (lost host, too few nodes). Handovers and wall-clock stops are unlimited. |
| `sbatch.signal` | `B:USR1@1200` | the supervisor stops the run gracefully, with a checkpoint, 20 min before the wall |

## 2. The job config

Any training experiment works unchanged. It keeps its own resources, log policy and restart
hooks. For long pool runs:

* **Policy:** use `auto_restart_ckptreset` or `auto_restart_ckptreset_faultscan`. **Do not use a
  policy with `MaxAttemptsCondition`** (for example `auto_restart_ft`). It counts the job's TOTAL
  attempts, and every handover adds one. Budgets must be `MaxActionFiresCondition`, which counts
  per event.
* **Fault scan:** `nccl_peer_scan` works for pool runs. It scans the failed RUN: its host's SLURM
  job id, its step's nodes in task order, and its own time window. Nodes it attributes are
  appended to `restart_exclude_file` (the fleet list); the next run and every new or queued
  host avoid them. In `pre_command`, `{runtime_job_id}` is the HOST's SLURM job id;
  `{run_id}` and `{step_nodes}` are also available.
* **Megatron:** `exit_signal_handler: true`, so SIGTERM saves a checkpoint. `save`/`load` point
  at the job's own `base_output_dir`.
* **`sbatch.time`** of the job is its per-RUN limit (enforced by the supervisor; TIMEOUT is final).
  Set it at or above the host's wall. `exit_duration_in_mins` works as usual.
* **The job's nodes must fit the host.** This is checked at enqueue. Account, partition, QOS and
  reservation must match the host's.
* `$SLURM_JOB_ID` inside a run is the HOST's id; `$AUTOEXP_RUN_ID` is per run. Do not build
  unique names, such as a wandb name, from `$SLURM_JOB_ID` alone.

## 3. Procedures

All commands run from the repository root, with the cluster environment loaded. On JUPITER:
`source ~/work/venv/bin/activate; export PYTHONPATH=. SLURM_ACCOUNT=<acc> OUTPUT_DIR=<dir>`.
`HF_HOME` must hold a complete tokenizer cache.

**Start a pool.**
```bash
python scripts/run_autoexp.py --no-monitor --monitor-state-dir $MS \
    --config-name experiments/korbi/pool/jupiter_pool_continuous job.name=<pool> slurm.sbatch.nodes=<N>
python scripts/run_autoexp.py --no-monitor --monitor-state-dir $MS --pool $OUTPUT_DIR/pools/<pool> \
    --config-name <training experiment> [overrides]
tmux new -d -s <pool> "python scripts/monitor_autoexp.py --session-dir $MS/pool_<pool tag> --poll-interval 60"
```
The session dir is in `$OUTPUT_DIR/pools/<pool>/pool.json` (`session_dir`). Run ONE monitor per
pool. A second one refuses to start while the first one's heartbeat is fresh.

**Restart a job with a fix, without a new allocation.**
1. `pool_ctl cancel <job>`. The run gets SIGTERM and saves. Wait for `monitor: cancelled [...]`.
2. Enqueue the fixed config with the SAME `job.name`, so it resumes that checkpoint. Add
   `+job.metadata.pool_priority=10` so that it runs before older queued work.

Cancel FIRST. With two live allocations, which happens during a handover wait, an idle one could
otherwise start the fix while the old run still writes into the same checkpoint directory. A
cancel of a job that has already ended is refused, and nothing is written.

**Move a run off an allocation (for example a slow node).** `pool_ctl exclude <node>`, then
`pool_ctl preempt <run>`. The job resumes on the remaining nodes, or on the next allocation.

**Add capacity or pause.**
* `pool_ctl set keep_pending=2` adds a second waiting successor.
* `pool_ctl close` cancels the queued hosts, drains the live ones and stops new submissions.
  `open` undoes it.
* `pool_ctl drain` finishes the current runs, then releases the hosts.

**Change pool settings live.** `pool_ctl set key=value ...`: the monitor and the supervisors
re-read them every poll. A new host DEFINITION (nodes, time, account) means re-running the host
config with `run_autoexp.py`. It applies to hosts created from then on.

**Update code while the pool runs.** Allocations survive code changes. Each piece takes effect at
a different point:

| What changed | Takes effect | How |
|---|---|---|
| monitor code (`oellm_autoexp/monitor/*.py`, `orchestrator.py`) | when the monitor restarts | stop it (Ctrl-C, or `kill` the monitor PID only), start it again; it adopts the running pool. Hosts and runs do not notice. |
| scripts the supervisor calls per run (`oellm_autoexp/utils/provenance.py`) and the jobs' own code and configs (Megatron, templates) | at the host's next run | nothing to do; the provenance line of the next run reports the change |
| `scripts/pool/supervisor.sh` | for hosts STARTED after the change | a running host keeps the version it started with |

**Replace `supervisor.sh` on disk, never edit it in place.** Bash reads a running script from its
file while executing it, so an in-place edit (an editor saving in place, `scp` or `cp` over the
file, `sed -i`...) can corrupt every host currently running it. Replacing the file gives the new
version a new inode while running hosts keep the old one. `rsync` (the default, without
`--inplace`) and `git checkout` both replace the file. Validated on JUPITER, 2026-10-27: one
host (job 2537762) ran three runs of one job, across a provenance fix synced between runs and a
monitor swap to new code, without an interruption.

When killing a monitor by pattern, do not use a pattern that also matches your own shell's
command line. `pkill -f "monitor_autoexp.py --session-dir ..."` sent over ssh matches the remote
shell running it. Use the PID from `ps`, or `pgrep -f` with a pattern that is not on your own
command line.

## 4. What is handled automatically

| Event | What happens | Where you see it |
|---|---|---|
| Successor allocation starts | after `handover_min_age_s`, the old host preempts its run (checkpoint) and releases itself; the job resumes on the new host; a new successor is queued | ledger `host exit: exit requested: handover ...`; monitor log `handover: retiring` |
| Host reaches its wall clock | USR1 → graceful stop → requeue → the job continues on the successor | run `rc` `0 preempted`; host COMPLETED (not TIMEOUT) |
| Run crashes (CUDA/NCCL error, exception) | the job's own policy: restart hooks (fault scan → exclusions), then a new run, normally in the SAME allocation | monitor log `Action executed ... RestartAction`; `NCCL peer scan of pool run ...` |
| Node fails, no `--no-kill` | SLURM ends the allocation → `host_lost` → requeue on the successor or a new host, from the last periodic checkpoint | run `rc` `-1 host_lost`; host NODE_FAIL |
| Node fails, `--no-kill` + spare | allocation survives → the policy restarts the run → the supervisor skips the down node → same allocation | new run's `started` `nodes=` without that node |
| Too few usable nodes / too little time left | the host is released AT ONCE and replaced | host `exited`: `cannot run any queued job (...)` |
| Bad node found after a successor was queued | the queued host's `ExcNodeList` is updated (`scontrol update`) | monitor log `queued host ... more node(s) excluded` |
| Supervisor hangs | heartbeat older than `host_stale_s` → host cancelled → run requeued | host `exited`: `stale heartbeat` |
| Host never starts (sbatch fails) | backoff 30 s, doubling to 10 min; one loud error per attempt | monitor log `ended without ever starting` |
| Monitor dies | runs continue; nothing is requeued or handed over until it is back. Idle hosts release themselves after `monitor_dead_exit_s`. | `pool_ctl status`: `monitor tick N s ago` |

## 5. Watching a pool

* `pool_ctl <pool> status`: hosts (SLURM id, state, heartbeat, busy or idle), runs (status, host,
  rc), unprocessed requests, and `CODE CHANGED SINCE ENQUEUE` flags.
* `<pool>/ledger.tsv`: one line per host start and exit, run start and end, and the provenance
  check.
* `<pool>/slurm-<jobid>.log`: a host's supervisor log. `<pool>/slurm-<jobid>.<k>-<name>.log` is
  the k-th run's own log.
* `<session>/monitor.log`: every decision, including handover, release, requeue, scans and
  exclusions.

Act on these:

| Sign | Meaning | Action |
|---|---|---|
| `monitor tick` older than a few polls | monitor dead | restart it (same command; it adopts the pool) |
| repeated `ended without ever starting` | sbatch broken (account, QOS, quota, PATH) | fix the cause; the backoff keeps retrying |
| several `host_lost` in a row | fabric or partition trouble | check the nodes, `pool_ctl exclude`, maybe `close` |
| runs `FAILED` again and again on the same host | node fault not attributed | `pool_ctl exclude <node>` + `preempt` |
| `CODE CHANGED SINCE ENQUEUE` | the repository changed while the job was queued | expected after a deliberate sync; otherwise find out why |
| `cannot run any queued job (... s left ...)` | allocations too short for `min_remaining_s` | lower `min_remaining_s` or raise the wall |

## 6. Pitfalls

* **Same `job.name` = same checkpoint directory.** Reusing a name resumes. Reusing the name of a
  FINISHED job "runs" in seconds.
* **Cancel first, then enqueue the fix.** See section 3.
* **No `MaxAttemptsCondition` in pool job policies.** See section 2.
* **Every new terminal needs the environment** (venv, `PYTHONPATH`, `SLURM_ACCOUNT`). On JUPITER,
  the login `SLURM_ACCOUNT` is `reformo`.
* **Quotas:** JUPITER fails writes above the SOFT limit. Measure with `df` on a path INSIDE the
  project directory; `jutil` can show 0 when it is not.
* **Overlap cost:** a switch holds two allocations for one save plus one startup, roughly 10-15
  min at 512 nodes. That is accepted here; `handover_min_age_s` bounds how often it happens.
* **Never `scancel` a host by hand.** Use `pool_ctl exit-host <hid>`, or `pool_ctl preempt <run>`
  and then `exit-host`. A scancel can put SLURM's `*** JOB <id> ... CANCELLED AT ***` line into
  the run's log. The job's `operator_cancel` rule then cancels the JOB for good, instead of the
  run being requeued. To end a job, use `pool_ctl cancel <job>`.
* **Restarts come from the log.** The recommended policies have no `state_events` (see
  `auto_restart_ckptreset.yaml`). A run's log contains its `srun`'s stderr, so `Exited with
  exit code`, `DUE TO NODE FAILURE`, NCCL timeouts and inactivity reach the job's restart rules.
* **One run per host at a time.** Pools do not pack jobs side by side.

## 7. Validation status (2026-10-27)

| | |
|---|---|
| fake SLURM, end to end | everything above, including handover, wall clock, host loss, bad node with and without a spare, the fault scan on a pool run, queued-host exclusion updates, failed-start backoff (`tests/pool`) |
| JUPITER, 130M Megatron, 2-node hosts (2026-10-26/27) | back-to-back runs in one allocation; preempt + requeue in the same allocation, resuming at the exact iteration; handover of a training run across allocations; a bad node excluded, with the next run on the spare node of the same allocation; a host that cannot fit the queue released at once; its replacement's `--exclude` (fleet list + bad node, also in SLURM's `ExcNodeList`); the wall-clock stop with `B:USR1` (graceful, host COMPLETED, resumed on the next host at iteration 1446 after 1445); `pool_ctl` cancel refusal and result reporting; `close` → drain → monitor exit; provenance with and without git; a monitor swap to new code with no host interruption. Every host ended COMPLETED, none TIMEOUT. |
| not yet run | anything above 2 nodes; a real node failure with `--no-kill`; MareNostrum. Run a 32-64-node pilot with a forced handover, a deliberate `scancel` of a host and a crashing run before going to 512 nodes. |

## 8. MareNostrum 5 (validated 2026-10-27, ehpc533)

**Production pools:** `mn5_pool_p33` (33-node hosts, acc_ehpc) and `mn5_pool_p129` (129-node
hosts, acc_xehpc), `/gpfs/scratch/ehpc533/hybrid_nucleus/pools/`.
- Both are run from snapshot `autoexp_pools1`; their sessions are in `$R/pool_state`.
- `min_hosts 1`, `keep_pending 1`, handover after 6 h, one spare node with `--no-kill`,
  `B:USR1@2700`.
- One monitor per pool, started with `scripts/korbi/mn5_autoexp_bg.sh pool_p<N>_mon
  scripts/monitor_autoexp.py --session-dir $R/pool_state/pool_mn5-pool-p<N> --poll-interval 30`.

**What differs from JUPITER (all fixed in the code):**
- **Stop signal:** the production runs use `--exit-signal SIGUSR2`, and SIGTERM kills them without
  a save. A run is stopped with the job's own exit signal (`meta.env` STOP_SIGNAL).
- **Working directory:** runs execute in the snapshot they were enqueued from
  (`job.metadata.pool_workdir`), not in the host's.
- **Relay steps need `--mem=0`:** a full step holds all node memory, and an `--overlap` step with
  any explicit memory request is refused ("Memory required by task is not available").
- **Probe CPUs:** MN5 has 160 CPUs per node (`--threads-per-core=2`). The phase-0 probe takes them
  from the allocation.
- **acc_debug** allows ONE submitted job per user, shared by every agent (same unix user), so it
  is not usable for pools that hand over.

**Phase-0 probe:** all mechanics pass (48115049, 48115374).

**2-node smoke** (s1 final recipe, real Megatron), all passed:
- Back-to-back runs, and preempt + requeue in the same allocation. SIGUSR2 reached 4 ranks per
  node; the run saved and resumed at the exact iteration.
- A fix enqueued from a new snapshot ran in it.
- The wall-clock stop: graceful, host COMPLETED.
- Six handovers in a row, resuming each time.

**Adopting queued jobs** (`pool_ctl <pool> adopt <jobid>`): turns an already-queued ordinary job,
such as the cns144 128-node jobs, into a pool host.
- A fail-closed hook (`scripts/pool/adopt_sitecustomize.py`, installed as `<WorkDir>/sitecustomize.py`)
  holds the job's own torchrun from the first instant.
- The monitor starts the supervisor into the allocation (`srun --jobid --overlap --mem=0`); runs
  are overlapping steps.
- Second fail-safe: the monitor cancels the job if the hook has not engaged 90 s after it starts.
- Validated on MN5 (48117110): 0 iterations of the held job, pool work at full speed, COMPLETED 0:0
  on release. The supervisor must drop the step's SLURM_MEM_PER_CPU and similar variables (done).
- Never resync a snapshot that holds an installed hook without the current sync script (it now
  excludes `/sitecustomize.py`).

**One run per job across pools** (`+job.metadata.pool_singleton=<key>`): enqueue the same job,
with the same `job.name`, into both pools.
- Only one run holds the key.
- A larger pool's host supersedes a smaller pool's run: its job is cancelled after a save, and the
  run resumes in the larger pool from that checkpoint.
