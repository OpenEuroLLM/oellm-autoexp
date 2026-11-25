# Pool allocation: improvement plan

**Status:** proposal, 2026-11-24. Written after a day in which a 256-node training run could not be
restarted for ~9 hours while every individual component behaved "as designed".
**2026-11-25:** reviewed against the code (§9; still nothing implemented). Added multi-user pools
(§10: foreign submitters, the hold-foreign-jobs flag, `permit`) and one overall implementation
plan (§11) that replaces the earlier phase table.
**2026-11-25 (later): P0–P2 implemented** (§11, "Implemented" below the table); P3–P6 are open.

Companion documents:

- [`docs/pool_allocation.md`](pool_allocation.md) — the node / GPU / GRES accounting this plan builds
  on (terms, failure modes, capacity probe, invariants). Read it first.
- [`docs/hiccups/20261124_marenstrum_v4b_ambitious.md`](hiccups/20261124_marenstrum_v4b_ambitious.md) —
  the fifteen concrete blockers of that day, the happy path, and the first version of the semantics
  proposed here.

---

## 0. Vocabulary

Three levels, one word each. Everything else in this document uses them.

| term | what it is | identifier today | example |
|---|---|---|---|
| **job** | the thing you submit by name: a definition plus a policy. One live instance at a time, with its restarts. This is what `cancel`, `replace`, `preempt`, `hold` and `protect` act on. | definition name + its job records | `ambitious_v4b_learned_fix_256` |
| **pjob** | one pool job = one attempt to run the job inside a pool. Created by the pool monitor, has `rc`, `started`, `assigned`, `meta.env`, `step.sh`. | `<pool>.NNNN` | `mn5-pool-p264.2406` |
| **sjob** | one Slurm job = one host allocation owned by a pool. Holds nodes and GPUs; hosts pjobs. | Slurm job id / `<pool>_hostNNN` | `48867560` / `mn5_pool_p264_host010` |

So: a **job** runs as **pjobs**, and those run on **sjobs**.

```
job  ambitious_v4b_learned_fix_256
 ├── pjob  .2403   rc=-1 insufficient_nodes   (placement failure, not an attempt)
 ├── pjob  .2406   started 21:54  iter 57110  (the live one)
 └── sjob  48867560  host010  261 nodes  gres/gpu=1044
```

Earlier drafts of this plan called the top level a *line*; that word is retired because it reads as a
text line next to `meta.env` and log output. Commands therefore read `cancel --job <name>`,
`status --job <name>`, `preempt --job <name>`.

---

## 1. What actually went wrong (compressed)

Every failure of that day belongs to one of five families. The plan is organised around them.

| family | today's symptom | root cause |
|---|---|---|
| **A. Cancel is not final** | 3 competing `ambitious` records; cancelled runs restarted on the next poll | `pool_ctl cancel` cancels one *record*; a live record re-submits on the next poll; the monitor has no durable "this job is cancelled" state |
| **B. Placement failure is treated as run failure** | `rc=-1 insufficient_nodes` every 34 s, each creating a new run dir | the monitor's `retry_failed` action counts a *capacity* rejection as a run failure |
| **C. Capacity is computed from stale state** | `needs 256 node(s), only 2 usable`; `srun: Requested node configuration is not available` | duplicated `EXCLUDE=` lines in `meta.env` (35 then 292); `ALL_NODES` from `info.env` (264) while Slurm had 261 |
| **D. Priority alone does not schedule** | 1-node evals took the host a 256-node run needed | assignments launched in rid order; no reservation; no eviction policy |
| **E. Operators (and agents) fight the system** | two supervisors, two resets, a preempt of the main run by its own guard | no single-instance lock, no "never evict the run you are placing" rule |

---

## 2. The model: a *job* is the unit

A **job** is `(pool, definition-name)`. It owns its records, its runs and its restart policy. Today
every command acts on a record or a run, which is why a job could not be stopped.

```
job
 ├── control/                  cancel | hold | protect | replace-request   (durable markers)
 ├── policy                    restart, preempt, exclusion, reservation
 ├── records  (must be 1 live)  <job_id>.job.json
 └── runs                      <pool>.NNNN  (rc, started, assigned, meta.env, step.sh)
```

### 2.1 Job state machine

| state | meaning | allowed transitions |
|---|---|---|
| `QUEUED` | one live record, no run placed | → `PLACING` (capacity ok), → `CANCELLED` |
| `PLACING` | run assigned, no `started` marker | → `RUNNING`, → `PLACEMENT_FAILED`, → `CANCELLED` |
| `PLACEMENT_FAILED` | capacity could not satisfy the request | → `PLACING` (retry on capacity change), **not** counted as an attempt |
| `RUNNING` | `started` marker present | → `FAILED`, → `STOPPED`, → `CANCELLED` |
| `FAILED` | run ended non-zero with a fault | → `QUEUED` (policy), → `CANCELLED` (policy exhausted) |
| `STOPPED` | graceful stop (preempt, wall clock, rotation) | → `QUEUED` (resume from checkpoint) |
| `CANCELLED` | **terminal** | none |

A foreign job (§10) starts one step earlier: `HELD(foreign)` → `QUEUED` via `permit`, → `CANCELLED`
via `deny` / `cancel`. While held it is not in the monitor's session at all (`inbox/held/`).

`CANCELLED` is terminal for the *job*, not for one record. Any monitor, after any restart, must
refuse to submit a job in `CANCELLED` unless an explicit `replace` clears the tombstone.

---

## 3. Monitoring

### 3.1 One supervisor per pool, with a lock

```bash
flock -n "$POOL/control/supervise.lock"  scripts/korbi/recall_surgery/supervise_main.sh
```

- Second instance exits with a clear message instead of competing (today two resets ran at once and
  produced duplicate records).
- The supervisor writes `$POOL/control/supervise.pid` (`pid host start_ts`) and refreshes
  `$POOL/monitor.alive`; `pool_ctl status` shows its age and refuses to act when it is stale.

### 3.2 What is monitored (and what is *not*)

| watch | signal | action |
|---|---|---|
| run `started` within `placement_grace` (default 90 s) | run dir | → `PLACEMENT_FAILED`, evict or wait, **no attempt consumed** |
| iteration line age | log | stall > `stall_s` (default 1800) → restart with a reason |
| `rc` after `started` | `rc` file | fault classification (below) |
| `srun: Requested node configuration is not available` | log | capacity/state bug → `PLACEMENT_FAILED` + `doctor` (§4.4) |
| `cudaErrorIllegalAddress`, `found NaN`, watchdog | log | node attribution (§5) → `FAILED` |
| tracker age | `ckpt/latest_checkpointed_iteration.txt` | no save in `save_interval × 2.5` steps → warn |
| disk free | `df` | below `disk_floor` → stop archiving, alert (§7.3) |

### 3.3 Health endpoint

`pool_ctl doctor` (new) prints a single screen, no side effects:

```
job ambitious_v4b_learned_fix_256   RUNNING
  record   a5e85a  submitted 21:53   attempts 0/4
  run      .2406   started 21:54:09  iter 57110  step 18.9 s  tracker 56950
  capacity alloc 261  exclusions 4  available 257  need 256  slack +1
  exclusions  pool 1 / host 3 / run 4  (union 4)   meta.env EXCLUDE lines 1  ok
  info.env    264 nodes  live 261   STALE (+3)      -> exclude dropped nodes or refresh
  monitor     alive 4 s   supervisor pid 1234 on alogin1
  evals       records 0  running 0  queued 0
```

Every field above maps to a failure we had today; if a field cannot be filled, that is itself a bug.

---

## 4. Autorestart

### 4.1 Policy object (per job)

```yaml
restart:
  on: [failed, preempted, timeout, wallclock]
  max_attempts: 4
  backoff_s: [60, 180, 600, 1800]
  stall_s: 1800
  placement_grace_s: 90
  placement_is_not_attempt: true      # key rule
  on_exhausted: cancel                # or hold
```

### 4.2 Fault classification (deterministic)

| class | signature | counts as attempt | node attribution |
|---|---|---|---|
| `run_fault` | traceback, assertion, trap (`tokens_per_expert`), NaN, illegal access | **yes** | yes, if a single node is named |
| `placement` | `insufficient_nodes`, `Requested node configuration is not available`, no `started` within grace | **no** | no |
| `infra` | NET/IB CQE, keep-alive, TCPStore reset, node health failure | **no** | yes |
| `operator` | `stop`/`preempt` marker, `pool_ctl preempt` | no | no |

Today `placement` was treated as `run_fault`, so every poll created a run dir and bumped attempts.

### 4.3 Restart must reuse the job's capacity check

Before creating a run, evaluate `feasible()` (§6). If not feasible: state `PLACEMENT_FAILED`, log once
per change, and wait for a capacity event (host resize, exclusion change, eval completion). Never spin.

### 4.4 `doctor` fixes the two capacity bugs found today

1. **`meta.env` duplicate keys** — the writer appended, so a second `EXCLUDE=` (292 entries) overrode
   the first (35) when the supervisor sourced the file. Writer must *replace* keys; `doctor` reports
   `meta.env EXCLUDE lines: N` and fails when N ≠ 1.
2. **Stale `info.env`** — `NUM_NODES`/`NODELIST` must follow `squeue -j <job> -o %N` after any resize.
   `doctor` reports `info.env 264 / live 261 STALE` and names the dropped nodes, which must then be
   excluded so the supervisor cannot pick them (§5.4).

---

## 5. Autoexclude

### 5.1 Attribution rules (what may exclude hardware)

| evidence | may exclude? | reason |
|---|---|---|
| `Rank N, node X, device D, iteration I: found NaN` | **yes** | single-device fault names its node |
| `cudaErrorIllegalAddress` with a watchdog naming a rank/node | **yes** | |
| `NET/IB ... port error(10)` (not `port active(9)`) | **yes** | HCA port fault |
| keep-alive failure naming a host | **yes** | |
| MoonEP trap (`tokens_per_expert`) | **no** | code/data condition |
| `FileNotFoundError`, `insufficient_nodes`, `configuration not available` | **no** | configuration, not hardware |
| generic `rc=1` with no named node | **no** | unknown until a probe says otherwise |

Today `port active(9)` (the recovery job) was read as a fault, and a `FileNotFoundError` was read as
a node failure. Both are attribution bugs.

### 5.2 Exclusion hygiene

1. **One merged set per host.** `union(pool, host, run)` intersected with the *live* allocation,
   computed once per event, written to one file. No consumer computes its own.
2. **Never inline a blob.** A record stores `exclude_file`, never an expanded list. Today a record
   carried 326 entries and every restart inherited them.
3. **Cap against slack.** After a merge, assert `available ≥ need + 1`; if not, refuse the submit and
   say which side is short. Today every "off by one" cost a full 16-minute startup.
4. **Dropped nodes are exclusions.** A node Slurm removed from the job must be added to the host
   exclusion set, or the supervisor will pick it and `srun` will refuse the whole step.

### 5.3 Re-admission (not only exclusion)

`pool_ctl readmit <node>`: probe the node individually (1-node `srun --gres=gpu:4`, `nvidia-smi`,
IB port states), and only then remove it from the exclusion sets. Today two nodes were re-admitted by
hand-editing the file, and one exclusion removal (needed to reach 256) was made without a probe.

### 5.4 Individual probe, never chunked

A chunked step fails as a whole and names no node. The probe must be **one node per step**:

```bash
srun --overlap --jobid="$JOB" --nodelist="$node" --nodes=1 --gres=gpu:4 \
     bash -c 'nvidia-smi -L | wc -l; nvidia-smi --query-gpu=memory.total --format=csv,noheader; \
              cat /sys/class/infiniband/*/ports/*/state'
```

Judge IB **against the fleet norm** (`mlx5_3` is DOWN on every MN5 node; 5 of 6 ACTIVE is normal),
otherwise every node looks broken.

---

## 6. Preemption and priority

### 6.1 Priority scheme

| tier | value | examples |
|---|---|---|
| `protected` | 1000 | the main training job; never auto-preempted once `RUNNING` |
| `main` | 150 | other training jobs |
| `control` | 90–120 | handover/continuation segments |
| `eval` | 55–60 | 1-node evaluations |
| `batch` | 30–50 | exports, archiving, smoke tests |

`preempt_margin: 100` (configurable). **A preemption is allowed only if `priority(R) − priority(V) ≥
preempt_margin`.** With the tiers above: main (150) may preempt evals (55/60), 90 points is *not*
enough to preempt control, and nothing preempts `protected`.

### 6.2 Preemption predicate

```python
def may_preempt(R, V, state):
    return (
        state.job(R).priority - state.job(V).priority >= PREEMPT_MARGIN
        and not state.job(V).protected
        and state.capacity_short(R)                       # R cannot be placed otherwise
        and state.job(R).feasible_after_evict(V)          # eviction actually helps
        and state.lost_work(V) <= PREEMPT_LOST_WORK_S      # e.g. 300 s since last save
        and state.job(V).evictable                        # not already held/evicted
    )
```

`feasible_after_evict` is the important one: today a preempt of one 1-node eval could never make a
256-node run fit, so the eviction was pure loss.

### 6.3 Eviction order and anti-thrash

1. Lowest priority first, then **least progress**, then newest run.
2. On eviction write `V/control/hold` — the job may not be re-placed until `R` has its `started`
   marker (or `hold_timeout`). This is what prevents the 2-second refill we observed.
3. Cooldown per job (`preempt_cooldown_s`, default 600): a job evicted twice in a window is held,
   not evicted a third time.
4. Graceful: `preempt` (save + requeue) for tier ≥ `control`; `cancel` only for tier `batch`, and only
   with `--force`.

### 6.4 Reservation (fixes "evals steal the host")

When a run needs N nodes and `available − N < reserve_floor` (default 1), the scheduler **reserves**
N nodes and refuses to place smaller runs into them. Launch order inside a host is **largest first**,
not rid order. Today evals launched first because their rids were lower.

---

## 7. `pool_ctl` changes

| command | today | proposed |
|---|---|---|
| `cancel <x>` | cancels one record; runs restart | **`cancel --job <name>`**: writes `control/cancel` (durable), marks every live record `final_state=cancelled`, cancels runs, writes a tombstone. Honoured after monitor restart. |
| `replace <job> <spec>` | — | submit the new record **and** cancel every older live record of the job in one transaction; at most one live record per job |
| `status` | hosts/runs dump | **`status --job <name>`**: state machine, record, run, capacity, slack, exclusions, ETA |
| `doctor` | — | read-only consistency check (§3.3): duplicate `meta.env` keys, stale `info.env`, exclusion union vs slack, monitor age |
| `preempt` | preempt one run | `preempt --job <name> [--force]`, subject to §6.2 |
| `hold` / `release` | — | hold a job or a run out of placement (used by eviction and by operators) |
| `protect <job>` | — | mark `protected`; refuses automatic preemption |
| `exclude` / `readmit` | `exclude` only | `readmit` requires a passing individual probe (§5.3) |
| `reconcile` | — | one poll that brings records, runs and markers back into a consistent state (used after a monitor restart) |
| `signal` | — | `SIGUSR2` to the monitor to force an immediate poll after a state change |

### 7.1 Signals (the point raised in review)

A monitor restart must not resurrect work. Concretely:

- **`control/cancel`** is a *file*, so it survives any monitor restart. The monitor's first poll after
  start calls `reconcile`, which (a) cancels records whose job has `control/cancel`, (b) drops runs
  with `rc` present, (c) holds runs whose job is held, (d) refuses to submit a job in `CANCELLED`.
- **`SIGUSR2`** makes the running monitor poll now instead of in 34 s; `pool_ctl` sends it after every
  state change so one command costs one poll, not one poll per step.
- **`control/protect`** is read by the evictor before it acts, so a protected job cannot be preempted
  by a stale in-memory priority.

---

## 8. Invariants to assert (fail loudly)

1. `1 live record per job`. A second live record is a bug; `reconcile` cancels the older one.
2. `meta.env` has exactly one line per key; the writer replaces, never appends.
3. `info.env` matches `squeue -j <job> -o %N`; after a resize it is rewritten.
4. `available = |live allocation| − |union(exclusions) ∩ live allocation| ≥ need + reserve_floor`.
5. A `srun` nodelist never contains a node outside the live allocation.
6. `CANCELLED` is terminal; only `replace` clears it.
7. `placement` never increments `attempts`.
8. One supervisor per pool (lock); an evictor never names the run it is placing.
9. `protected` jobs are never auto-preempted.
10. Every state change is executed in one poll (signal after write).

---

## 9. Review of §0–§8 against the code (2026-11-25)

Nothing in §0–§8 is implemented. Some of it overlaps with code that already exists, and a few points
are wrong about the code. Corrections:

| § | proposal | what the code does today | correction |
|---|---|---|---|
| 0, 2 | job = `(pool, definition-name)` | Records are keyed by `job_id`. `enqueue_into_pool` (`oellm_autoexp/orchestrator.py:1071`) makes `<name>`, `<name>_q2`, … for each enqueue. The only identity shared across records is `metadata.pool_singleton` (`PoolManager._claim_singleton`, `<pools>/_singletons/<key>`), and it **spans pools**. | Job key = `pool_singleton` when set, else `definition.name`, **qualified by the submitter** (§10): `(submitter, key)`. The job's `control/` lives next to the singletons (`<pools>/_jobs/<key>/`), so a cancel also applies to a copy of the job in another pool, as the singleton guard does. |
| 3.1 | `flock` around `scripts/korbi/recall_surgery/supervise_main.sh` | That script is an agent helper, not the pool monitor. The monitor already refuses a second instance through its heartbeat (`ensure_no_live_monitor`: pid, host, age). | The lock belongs in the monitor (`run_autoexp.py` / `monitor_autoexp.py`). Use a `mkdir` lock plus the heartbeat, like `_claim_singleton`; do not rely on `flock` across GPFS login nodes. Helper scripts get their own lock. |
| 3.2, 3.3 | "disk floor (§7.3)" | §7.3 does not exist. | Turn it into a `doctor` warning; no automatic action. |
| 4.4.1 | `meta.env` writer must replace keys | `write_env_file` already rewrites the whole file. `add_run_exclude` (`pool.py:440`) appends a second `EXCLUDE=` **on purpose** (the last line wins). The 292-entry line came from re-reading the fleet `exclude_file` (`_exclude_list`) at render / refresh time. | The run's `EXCLUDE` holds only its own nodes, intersected with the host's allocation. Fleet, pool and host lists are merged by the supervisor at launch (it already reads all three). `add_run_exclude` rewrites the key. |
| 4.4.2 | `info.env` must follow `squeue` | `supervisor.sh refresh_alloc` (since 11-05) rewrites `info.env` from `squeue -o %N` every `POOL_ALLOC_REFRESH_S`. The monitor has `live_job_nodes` / `removed_nodes`. Only supervisors started before that code keep their start-time list. | No new writer is needed. `doctor` reports stale `info.env` (old supervisors), and P0 is narrowed accordingly. |
| 4.2 | placement ≠ attempt | The supervisor writes `rc -1 insufficient_nodes` → `PREEMPTED`. `loop.py:440` counts it as *unhealthy* against `max_attempts`. Every requeue goes through `PoolManager.submit` → `allocate_run_id` → a **new run dir**. | Correct as written. The fix spans **both** halves: the supervisor leaves an unplaceable run PENDING with a `placement` note (no `rc`, so no new rid), and the monitor stops counting it. |
| 6 | one preemption predicate | Three paths preempt today: `pool_ctl preempt`, the cross-pool singleton supersede (`_claim_singleton`), and handover. | The predicate has to gate all three, otherwise invariants 8 and 9 do not hold. |
| 6.4 | reservation | The same starvation has happened at least three times (memory: p132 10-31, p264 11-24). | Highest value in the list: do it before preemption (§11 P4 before P5). |
| 7 | `hold` / `release` for jobs | `pool_ctl hold <host>` already exists (the host is never released). | Job-level commands always take `--job <key>` (`hold --job`, `release --job`). The positional `hold <host>` keeps its meaning. |
| 7.1, 8.10 | `SIGUSR2` to the monitor | The monitor runs on one login node and `pool_ctl` often on another, so a signal cannot reach it. | `pool_ctl` writes `requests/*.json` (as today), and the monitor checks `requests/`'s mtime every ~2 s while it sleeps. The effect is the same and it works across login nodes. |

---

## 10. Multi-user pools

**Goal (user, 2026-11-25):** a colleague in the same group can submit jobs into "my" pool. Runtimes
(containers, environments) are group-accessible, and output dirs are group-writable, so results
written by the pool owner's user are fine. Pools get a **hold-foreign-jobs flag** plus a **`permit`
command**, and the flag can be switched off later.

### 10.1 What happens today when user B enqueues into A's pool

Traced on `mn5_pool_p264`; `mn5_pool_c136` (cns143) has the same layout. Permissions were read
on MN5 on 2026-11-25:

```
/gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p264        drwxrws---+  group ehpc533 rwx (default ACL)
  runs/ hosts/ requests/                                        drwxrws---+  group rwx
/gpfs/projects/ehpc533/users/korbip/hybrid_v019/pool_state/pool_mn5-pool-p264   drwxr-sr-x   group r-x  (= session dir)
```

1. `run_autoexp.py --pool <A's pool>` → `enqueue_into_pool` reads `pool.json` and
   `host_definition.json` and runs `check_fits_host`. **This works**: the pool dir is group-readable.
2. It writes `<session>/<job_id>.job.json` with `JobFileStore.upsert`. The session dir is `r-x` for
   the group, so B gets **PermissionError. This is the only hard blocker.**
3. If the session dir were writable, A's monitor would pick the record up on its next poll
   (`load_all` globs the session dir). `PoolManager.submit` would render `step.sh` from B's template
   in B's workdir, and A's supervisor would run it **as A**. It would "work", with no gate at all.
4. After the enqueue, `run_autoexp` checks `ensure_no_live_monitor`. If A's monitor is down, it
   would **start a monitor as B**. That monitor would `sbatch` hosts under B, write host records into
   A's pool, and race A's monitor when A's comes back. This must be refused.
5. `pool_ctl status` / `cancel` as B: B can read the session (`r-x`) and write `requests/` (`rwx`), so
   **B can already cancel any job in the pool**. Nothing checks who is asking.
6. Everything else in the pool dir is group-writable today: `runs/*/step.sh` of queued runs,
   `pool.env` (sourced by every supervisor), `exclude_nodes`, `hosts/*/control/`. Any group member can
   already run code as A, or stop the main run. Pool integrity is cooperative today.

### 10.2 Trust boundary

- A foreign run executes **B's code** (B's workdir, template and container) **as A**, on A's
  allocation, charged to the host's account. By default B must be a member of that account
  (`foreign_account_check`, §10.4 step 2b); the owner can switch the check off per pool. B can
  change the code in their workdir after a permit. `permit` therefore means "I accept this job onto my
  allocation" (resources, priority, slot). It is **not a sandbox**.
- What the pool can enforce:
  - who may put jobs in;
  - who may steer whose jobs;
  - that B cannot change the pool's own state (other runs' step scripts, settings, hosts);
  - that B cannot start hosts or monitors for A's pool;
  - that foreign metadata cannot escalate (priority, singleton, host role, login-node commands).
- **Authentication = file owner.** The monitor reads `st_uid` of an inbox or request file. B cannot
  create a file owned by A. A user name written *inside* a file is never trusted.

### 10.3 Layout and permissions

```
<pool>/pool.json        + "owner": {"user", "uid"} (first creation; a non-owner never rewrites pool.json)
<pool>/inbox/           group rwx, setgid: every enqueue lands here (owner too, see P1)
<pool>/inbox/held/      monitor-owned, group r-x: foreign jobs that passed validation and await `permit`
<pool>/inbox/done/      monitor-owned, group r-x: <file>.result.json = queued <job_id> | held | rejected: <why>
<pool>/requests/        group rwx (as today); the monitor authorizes each request by its st_uid
everything else         group r-x: runs/, hosts/, pool.json, pool.env, exclude_nodes, ledger.tsv
<pools>/_singletons/    group r-x (today group rwx through the hybrid_nucleus default ACL)
<session>/              group r-x (as today): status and cancel resolution read it, only the monitor writes
```

**Write access is not what keeps this ordered** (user, 2026-11-25). A group member with write access
could in principle do anything by editing the `.sh` scripts. Order comes from the flow itself: the
inbox, the hold/permit, requests authorized by the monitor, and `pool_ctl` refusing owner commands.
That flow works the same with or without write access. Tightening the ACL is an **optional**
hardening step:

`hybrid_nucleus` hands out `group:ehpc533:rwx` as its default ACL, so a tighter ACL has to be set
explicitly:

- **any pool, when wanted:** `pool_ctl secure` (plan; `--apply`). New pools are **not** tightened
  automatically. It changes filesystem ACLs only (no Slurm job), so
  it is compatible with AGENTS.md.

Admins (`pool_admins`) get `u:<admin>:rwx`. Inbox and requests must **not** be sticky: the monitor
(A) moves B's files out of them.

### 10.4 Ingest (monitor, every poll, before scheduling)

For each `inbox/*.job.json` (dotfiles are in-flight writes and are skipped):

1. **Who:** `uid = st_uid` → user name.
   - Owner or admin → normal path.
   - Anyone else must match `foreign_submitters` (`""` = nobody, `@group` = any member of the pool
     dir's group, or a list of users). Otherwise the job is rejected.
2. **Fits:** parse the record and run `check_fits_host` again, as A.
2b. **Account membership** (`foreign_account_check`, default on): B must be a member of every
   Slurm account the job could run on (the host definition's account plus `host_accounts`).
   - Methods, tried in order: `sacctmgr show assoc user=B account=X`, when the cluster lets A see it;
     then the Unix group named after the account (`getent group X`).
   - MN5 (checked 2026-11-25): `PrivateData=accounts,...,users` hides other users' associations
     from sacctmgr. The accounts are Unix groups, though (`cns143` lists its 4 members, `cns144`
     10), so the group method decides there.
   - If no method can decide, the job is rejected with a message naming
     `pool_ctl set foreign_account_check=false`.
   - In a pool with several host accounts, a foreign job is placed only on hosts whose account B
     belongs to.
3. **Sanitize** foreign metadata:

   | key | foreign value |
   |---|---|
   | `pool_role`, `pool_host_id`, `pool_host_account`, `pool_host_adopted`, `pool_adopt_workdir` | rejected (B could otherwise make A's monitor `sbatch` a host) |
   | `pool_preflight_template` | rejected (it would replace the pool's node check) |
   | `pool_job_id`, `pool_enqueued_ts`, `pool_submitted_by` | overwritten by the monitor |
   | `pool_priority` | clamped to `foreign_max_priority` (default 60 = eval tier, §6.1); `permit --priority N` overrides |
   | `pool_singleton` | dropped (it would let B supersede A's main run across pools); `permit --singleton` restores it |
   | `pool_use_excluded_nodes`, `pool_stop_signal`, `pool_workdir` | kept |
   | monitor events with `CommandCondition` / `ShellCommandCondition` / `RunCommandAction` / `NewJobAction` / `AppendToFileAction` / `RestartAction.pre_command` | **kept and listed** in the held entry (`pool_ctl status`) so the owner sees them before `permit`. Gating them would add no protection, because once permitted the run's `step.sh` executes the submitter's code as A anyway (decided 2026-11-25 during implementation) |

4. **Writable as A:** the job's log dir, config-path dir and `base_output_dir` must be writable by A.
   A real probe file is used, because `os.access` does not evaluate ACLs reliably on GPFS. A failure
   is reported with the dir that failed.
5. **Name collision:** a foreign job whose name or singleton key matches an active job of another
   submitter is rejected.
6. **Stamp and id:** `metadata.pool_submitted_by = {user, uid, ts, inbox_file}`. The job id is
   assigned here, which moves the `_qN` dedupe into the monitor (serialized, with no enqueue race).
7. **Place:**
   - foreign and `hold_foreign_jobs` → `inbox/held/<job_id>.job.json`, result `held`;
   - otherwise → `store.upsert` into the session, result `queued <job_id>`.

A held job is **not in the session**, so it is inert by construction: no events are evaluated, no
run dir is created and no `step.sh` is rendered.

### 10.5 `permit`, `deny`, and the flag

| command | who | effect |
|---|---|---|
| `pool_ctl <pool> permit <job\|name> [--priority N] [--singleton]` | owner, admins | request op `permit`. The monitor checks the request's `st_uid`, moves held → session and writes a ledger line `permit <job> by <user>`. |
| `pool_ctl <pool> permit --user B` / `--all` | owner, admins | every currently held job of B, or every held job |
| `pool_ctl <pool> deny <job> [--reason ...]` | owner, admins, submitter | held → `inbox/done` with the reason; B sees it in the enqueue result |
| `pool_ctl <pool> set hold_foreign_jobs=0` | owner, admins | new foreign jobs go straight to the queue (still sanitized). **Already-held jobs stay held**; `status` lists them with a `permit --all` hint, so switching the flag off never launches a backlog by surprise. |
| `pool_ctl <pool> set foreign_auto_permit=B,C` | owner, admins | standing permit for named users: the flag stays on for everyone else |

### 10.6 Who may steer what

Request ops are authorized by the monitor (`st_uid`). Commands that write files directly become
owner-only through the ACL, and `pool_ctl` refuses them up front with a clear message.

| op | owner / admin | submitter (own jobs) | other group member |
|---|---|---|---|
| `status`, `queue`, `doctor` | yes | yes | yes |
| `cancel`, `replace`, `deny` | any job | yes | no |
| `preempt` (becomes a request op; today a direct `runs/<rid>/preempt` marker) | any job | yes | no |
| `permit`, `set`, `exclude`, `readmit`, `pin`, `hold`, `protect`, `freeze`, `shrink`, `park`, `promote`, `drain`, `close`, `open`, `adopt`, `attach`, `secure` | yes | no | no |

### 10.7 `run_autoexp.py --pool` as a non-owner

- **Never becomes the pool's monitor** (`pool.json owner.uid != os.getuid()`): it prints "queued in
  A's pool; scheduled when A's monitor runs". It also never calls `prepare_pool_host` on someone
  else's pool.
- **Client-side preflight** before writing to the inbox, so B learns about problems before A's
  monitor rejects:
  - workdir, template and container image are group-readable;
  - log and output dirs are group-writable (or have a group ACL);
  - warnings for `~`, `$HOME`, `$USER` and `/home/...` paths, and for B's own `users/<B>/` dirs
    without group write.
- Writes the inbox file atomically (dot-prefixed tmp, then rename) and waits for
  `inbox/done/<file>.result.json`, as `_await_result` does for requests (`--no-wait` skips the wait).
  It then prints `queued <job_id>`, `held (awaiting permit by <owner>)` or `rejected: <why>`.

### 10.8 A foreign run's environment

It runs as A. Things to be aware of:

- `$USER`, `$HOME`, `~/.cache`, HF / Triton caches and W&B credentials are **A's**.
- `squeue --me` and `scancel -u $USER` inside the job act on **A's** jobs. Watchdog-style helpers
  that cancel by user must not run inside foreign jobs.
- The monitor renders `export AUTOEXP_SUBMITTED_BY=<user>` into the run's `step.sh`. This is not
  a supervisor change, so it also works on hosts that are already running (§11 P1).
- Files are created as A. With the `hybrid_nucleus` setgid + default ACL they stay group-writable, so
  B can use and delete them. A user-level quota would count them against A; MN5 scratch quota is
  per group.
- Slurm accounting and fair-share see A's user on the host's account.
- The submitter is recorded in `run.json` (`submitted_by`) and in a monitor-written ledger line
  (`queued <rid> by <user>`), so GPU-h per submitter can be summed from the ledger. There is no
  GPU-h cap for foreign users (user, 2026-11-25).

### 10.9 Config

```yaml
pool_host:
  foreign_submitters: ""      # "" = owner only (today's behaviour); "@group"; or "userb,userc"
  hold_foreign_jobs: true     # foreign jobs wait in inbox/held until `pool_ctl permit`
  foreign_auto_permit: ""     # users whose jobs skip the hold
  foreign_max_priority: 60    # clamp for foreign pool_priority; permit --priority overrides
  foreign_account_check: true # submitter must be a member of the host account(s); false = allow spending
                              # in accounts they are not a member of (explicit owner decision)
  pool_admins: ""             # users treated as the owner for steering (plus ACL rwx)
```

All of these are `PoolHostConfig` fields, live through `pool_ctl set` (as today). `owner` is written
into `pool.json` at creation, and `create_or_update_pool` refuses an update from a different uid.

---

## 11. Overall implementation plan

Order:

1. read-only diagnostics;
2. the ingest path, which multi-user and the job model both build on;
3. job identity and durable control;
4. the scheduler changes, in order of value (placement, then reservation, then preemption).

Every phase ships with a regression test in `tests/pool/` (fakeslurm harness) that replays the log
signature of a real failure. Multi-user tests inject the file owner through a `file_uid(path)` seam,
because tests cannot change uid.

| phase | scope | main files | acceptance test |
|---|---|---|---|
| **P0** diagnostics + writer fix | `pool_ctl doctor` (read-only, §3.3); the run-level `EXCLUDE` holds only the run's own nodes ∩ allocation, and `add_run_exclude` rewrites the key (§9 row 4.4.1); `status` shows monitor age and record count | `scripts/pool_ctl.py`, `pool.py` (`add_run_exclude`, `_exclude_list` use in `submit`) | `doctor` on a fixture with two `EXCLUDE=` lines and a stale `info.env` reports both; a re-rendered run never carries fleet-list nodes outside its host |
| **P1** inbox ingest + multi-user | `inbox/` for **other users'** enqueues. The owner's enqueues stay direct, which is compatible with old snapshots and gives immediate job ids; the monitor ingests an owner's inbox file the same way, and P2 can route everything through it; `owner` in `pool.json`; §10.4 ingest with sanitizer; `hold_foreign_jobs`, `permit`, `deny`, `foreign_*` settings; `foreign_account_check`; request authorization by `st_uid`; non-owner `run_autoexp` path (§10.7); `pool_ctl secure`; submitter in `run.json`, `status` and a monitor ledger line; `AUTOEXP_SUBMITTED_BY` rendered into `step.sh`. **No `supervisor.sh` change**, so running pools can be repurposed with a monitor restart. | `orchestrator.py` (`enqueue_into_pool` → inbox), `pool.py` (`PoolHostConfig`, `create_or_update_pool`, new `_ingest`, `_process_requests` authorization, `render_step_script`), `scripts/run_autoexp.py`, `scripts/pool_ctl.py` | a foreign job is held with no run dir and no events evaluated; `permit` by the owner queues it; a `permit` request written by a foreign uid is refused; with `hold_foreign_jobs=0` it queues directly; the sanitizer rejects `pool_role` and clamps the priority; a submitter outside the host's account group is rejected, and accepted with `foreign_account_check=false`; a non-owner `run_autoexp` does not start a monitor; an unwritable log dir → `rejected: <dir>`; a foreign `cancel` of the owner's job is refused |
| **P2** job identity + durable control | job key `(submitter, singleton-or-name)`; `<pools>/_jobs/<key>/control/{cancel,hold,protect}`; `cancel --job` with tombstone, `replace`, `status --job`; `reconcile` on monitor start; `mkdir` monitor lock; fast request poll (§9 row 7.1) | `pool.py`, `loop.py` (submit refuses a tombstoned job), `scripts/pool_ctl.py`, `orchestrator.py` (lock) | killing and restarting the monitor does not resurrect a cancelled job; a submit leaves exactly one live record (11-24: `_q2 … _q18`); a cancel in p264 also stops the job's copy in another pool |
| **P3** placement ≠ attempt | the supervisor keeps an unplaceable run PENDING with `placement` (no `rc`, no new rid); the monitor stops counting `insufficient_nodes`; fault classification §4.2; backoff | `scripts/pool/supervisor.sh`, `loop.py:425-460`, `pool.py` | a 256-node job against 255 usable nodes logs once and creates **no** run dir (11-24: one every 34 s) |
| **P4** reservation | reserve N nodes for the highest-priority waiting run; largest-first launch within a host; `hold` on eviction against the 2-second refill | `pool.py` (assignment), `supervisor.sh` (launch order) | an eval cannot take a node the main run needs (11-24, 10-31) |
| **P5** preemption | §6.2 predicate (`margin ≥ 100`, `feasible_after_evict`, lost-work budget) applied to **all three** paths: `pool_ctl preempt`, singleton supersede, handover; `protect` | `pool.py` (`_claim_singleton`, `_handover_check`), `pool_ctl.py` | the main run preempts an eval and starts; an evictor never preempts the run it is placing; a protected job is never preempted |
| **P6** autoexclude | §5.1 attribution table; `readmit` with a single-node probe (§5.4, fleet-normal IB) | `pool.py` (`scan_pool_run`), `pool_ctl.py` | `port active(9)` and `FileNotFoundError` exclude nothing; a NaN excludes exactly its node |

**Implemented 2026-11-25 (P0–P2), where it differs from the table:**

- **P0 follow-ups (user review, 2026-11-25):**
  - **Health check.** The monitor runs the cheap part of `doctor` (no SLURM calls) every
    `health_check_s` (900 s; 0 = off). It writes `<pool>/health.txt`, logs each new PROBLEM
    once, and `pool_ctl status` shows the result.
  - **`doctor --probe [--hosts …] [--gpus N]`.** One `srun --overlap --mem=0` step per node (no
    `--gres`, so busy nodes answer too). It checks the GPU count and the ACTIVE InfiniBand ports
    against the host's most common count. It writes `<host>/doctor_probe.txt`, excludes nothing,
    and skips frozen hosts.
  - **Dead-node copy kept.** Dead and removed nodes stay copied into every run's `EXCLUDE`.
    Health exclusions (bad GPU/NIC) are not copied, so `USE_EXCLUDED_NODES` runs (evals) keep
    using those nodes.
- **P0:** the run's `EXCLUDE` is recomputed at every assignment as `JOB_EXCLUDE` (the job's own list,
  written at submit) ∪ the host's dead nodes ∪ its removed nodes (the latter only for
  `USE_EXCLUDED_NODES` runs). It is written as one line, and older doubled lines are repaired
  (`set_run_exclude`). `pool_ctl doctor [--no-slurm]` is read-only and exits 1 on a problem.
- **P1 follow-ups (user review, 2026-11-25):** a foreign job held longer than `held_warn_s` (1 h)
  is a PROBLEM in the health check and in `doctor`, logged once. Switching the hold off leaves held
  jobs held; `permit --all` / `--user` releases them whenever the owner chooses.
- **P1:** as in §10, except that `secure` is optional, login-node commands are listed rather than
  gated, and the owner's own enqueues stay direct.
- **P2:**
  - **Job key and markers.** The key is `<submitter>:<pool_singleton or name>`. Markers live in
    `<pools>/_jobs/<key>/control/{cancel,hold}`.
  - **Tombstones.** The monitor closes the records of a tombstoned job on **every poll**, not only
    at start. That covers a monitor restart and a cancel given in another pool, so no separate
    `reconcile` command exists.
  - **`--replace`.** `run_autoexp --replace` clears the tombstone and closes the job's older live
    records in this pool (`supersede`). A plain enqueue next to a live record only warns; it does
    not auto-cancel.
  - **Commands.** `pool_ctl cancel --job`, `hold --job` / `unhold --job` (`unhold` mirrors the
    existing host command; no `release`), and `status --job`.
  - **Not yet.** `protect` waits for P5, which has the evictor that reads it.
  - **Monitor lock.** A `mkdir` lock in `run_loop` (`<session>/.monitor.lock`): a live holder
    refuses; a dead or stale holder, or `--force`, is taken over; the lock is released at exit,
    which includes the CPU-budget restarts.
  - **Fast pickup.** Instead of `SIGUSR2`, the monitor's sleep ends within ~2 s when a request or
    inbox file is waiting.

**Rollout notes per phase:**

- **P0** is read-only plus one writer change, and is safe on live pools.
- **P1 and P2** are monitor-side. They take effect at the next **monitor** restart, which is pool
  state, not Slurm.
- **P3 and P4 change `supervisor.sh`.** A supervisor runs the code of the snapshot its host started
  from, so held hosts keep the old behaviour for up to their 3-day wall clock. The monitor has to work
  with both supervisor versions until every host has rotated.
- **P1's inbox switch has to be compatible.** For one release the monitor also accepts records
  written directly into the session (an old snapshot's `run_autoexp`), and treats them as the owner's.

---

## 12. Migration without breaking running jobs

- Everything above is **pool state** (and filesystem ACLs for `pool_ctl secure`), not Slurm. No job
  is edited, requeued or resized (AGENTS.md).
- New commands land behind `pool_ctl --experimental` first; `doctor` is read-only and safe from day one.
- `reconcile` runs only at monitor start, so a rollout needs a monitor restart. That is exactly the
  case the `control/cancel` tombstones must survive.
- Keep the current scripts (`supervise_main.sh`, `reset_pool_and_start.sh`, `dedupe_meta.sh`) as
  documented workarounds until P3/P4 land. Each has a single-instance lock and a "never evict the run
  you are placing" guard.
- **Multi-user rollout:** a general oellm-autoexp feature, repurposing running pools. The first
  target is the cns143 pool `mn5_pool_c136` (user, 2026-11-25). Its layout is the same as p264's:
  pool dir group rwx through the ACL, session dir group `r-x`.
  1. **Code.** `sync_to_marenostrum.sh` to a new snapshot, then restart c136's monitor from it. That
     is pool state only: the hosts, their Slurm jobs and running runs stay untouched. P1 changes no
     `supervisor.sh` code, so running hosts need no rotation.
  2. **ACLs (optional).** `pool_ctl mn5_pool_c136 secure`, dry-run first; it prints the ACL
     changes. The flow does not need it (§10.3); nobody but `eit847786` relies on write access in
     the MN5 pool dirs (user, 2026-11-25).
  3. **Allow the colleague.** `pool_ctl mn5_pool_c136 set foreign_submitters=<colleague>`, with
     `hold_foreign_jobs=true`. Leave `foreign_account_check` on: the colleague has to be in
     `getent group cns143`, unless you switch the check off for this pool.
  4. **Smoke test.** The colleague enqueues a 1-node smoke **from a snapshot with the new
     `run_autoexp`**. An old one fails harmlessly with PermissionError on the session dir. Then
     `status` → `permit` → check that it runs and that its outputs are group-writable.
  5. **Widen.** Only then add more submitters or switch the hold off.

---

## 13. Open questions

1. **Value, not only priority.** Priority says who *should* win; progress says who *should not* be
   killed. The eval harness has per-item progress; exposing it would make `feasible_after_evict` and
   the lost-work budget far better than a fixed 300 s.
2. **Who owns the exclusion decision?** A node excluded for one pool changes capacity for another.
   Today three lists exist (global, pool, host) and their union is what matters.
3. **GRES blindness.** `sinfo` / `scontrol show nodes` are denied for this account, so the pool cannot
   see node state or GRES at all. Either get the permission or make the srun probe the official
   capacity oracle (`doctor` should run it on demand).
4. **Monitor placement.** The monitor runs on a login node that can disappear; `control/*` markers make
   that survivable, but the *supervisor* (host side) also needs to honour them.
5. **Which colleagues, and admins?** `foreign_submitters` is per pool. Should anyone be a
   `pool_admin` (steer and permit), or only a submitter?

Decided 2026-11-25 (user):

- **First pools:** the cns143 pools first (`mn5_pool_c136`), but the feature is general to
  oellm-autoexp, not MN5-specific.
- **No GPU-h cap** for foreign submitters. Usage is only recorded, per submitter, in the ledger.
- **Spending in other accounts** is allowed when the owner switches `foreign_account_check` off for
  the pool. The default checks membership (§10.4 step 2b).
- **ACL tightening is safe:** no other user relies on write access to the MN5 pool dirs.

---

## 14. One-paragraph summary

Treat a training job as a first-class object with a durable `cancel`/`protect`/`hold` marker, one live
record, a state machine, and a policy; compute capacity from the **live** allocation minus a **single
merged** exclusion set and refuse to act when it does not fit; never count a placement failure as a
restart; never launch small runs into capacity a higher-priority run has reserved; only preempt when
the priority margin is at least 100 **and** the eviction actually makes the request feasible **and** the
victim loses little work; attribute node faults only to faults that name a node; and make every state
change one poll long. Every enqueue goes through a pool **inbox** that the owner's monitor ingests: it
authenticates the submitter by file owner, sanitizes foreign metadata, holds foreign jobs until
`permit` (a flag that can be switched off), and never lets a non-owner start a monitor or change pool
state. A foreign run still runs the colleague's code as the owner, so `permit` gates resources, not
trust.
