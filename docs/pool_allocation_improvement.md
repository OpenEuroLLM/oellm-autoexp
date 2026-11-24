# Pool allocation: improvement plan

**Status:** proposal, 2026-11-24. Written after a day in which a 256-node training run could not be
restarted for ~9 hours while every individual component behaved "as designed".

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

## 9. Phased implementation

| phase | scope | acceptance test (regression of a real failure) |
|---|---|---|
| **P0** | writer fixes: `meta.env` replace-keys, `info.env` follows the job, `doctor` | `.2403`'s `Requested node configuration is not available` cannot occur; `doctor` reports `EXCLUDE jobs 1` and `info.env stale` |
| **P1** | `cancel --job`, `replace`, `reconcile`, `SIGUSR2` | killing and restarting the monitor does **not** resurrect a cancelled job; a submit leaves exactly one live record (today: `_q2 … _q18`) |
| **P2** | fault classification; `placement` ≠ attempt; backoff | a 256-node request against a 255-node capacity logs one job and creates **no** run dir (today: one per 34 s) |
| **P3** | reservation + largest-first launch + `hold` on eviction | an eval cannot take a node the main run needs; no 2-second refill |
| **P4** | preemption predicate (`margin ≥ 100`, `feasible_after_evict`, lost-work budget), `protect` | main preempts an eval and starts; an evictor never preempts the main run it is placing |
| **P5** | autoexclude attribution table + `readmit` with probe | `port active(9)` and `FileNotFoundError` exclude nothing; a NaN excludes exactly its node |

Each phase ships with a test that replays the corresponding log signature from today.

---

## 10. Migration without breaking running jobs

- Everything above is **pool state**, not Slurm. No job is edited, requeued or resized.
- New commands land behind `pool_ctl --experimental` first; `doctor` is read-only and safe from day one.
- `reconcile` runs on monitor start only, so a rollout needs a monitor restart — which is exactly the
  case the `control/cancel` tombstones must survive.
- Keep the current scripts (`supervise_main.sh`, `reset_pool_and_start.sh`, `dedupe_meta.sh`) as
  documented workarounds until P2/P3 land; each has a single-instance lock and a "never evict the run
  you are placing" guard.

---

## 11. Open questions

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

---

## 12. One-paragraph summary

Treat a training job as a first-class object with a durable `cancel`/`protect`/`hold` marker, one live
record, a state machine, and a policy; compute capacity from the **live** allocation minus a **single
merged** exclusion set and refuse to act when it does not fit; never count a placement failure as a
restart; never launch small runs into capacity a higher-priority run has reserved; only preempt when
the priority margin is at least 100 **and** the eviction actually makes the request feasible **and** the
victim loses little work; attribute node faults only to faults that name a node; and make every state
change one poll long.
