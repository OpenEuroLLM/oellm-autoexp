# 2026-11-24 — MareNostrum v4b "ambitious" main run: what blocked us

Session record of the failures between 05:24 and 21:16 UTC+1 on 2026-11-24, in the order they bit us,
with the fix for each. Everything here is on `mn5_pool_p264` / `host010` (job 48867560, 264 nodes,
`gres/gpu=1056`, later 261 nodes / `gres/gpu=1044`).

Read this before touching the pool again: most of the day was spent re-learning these.

---

## 1. The MoonEP fix was not in the live run

**Symptom.** The main line restarted from a frozen submission definition that pinned
`pylibs_moonep_ec3eb8b`, i.e. the build with the illegal-access path.

**Fix.** `e9e8238` (fence + guards) prebuilt as `pylibs_moonep_e9e8238`, plus the Megatron patch
`baa4c0db1` (`dense_topk_from_routing`) in a new snapshot `autoexp_v4bfix2`.

**Trap.** The exclusion list and `PYTHONPATH` are read at **render** time. A fix to a config file
does not reach an already-submitted run, and a fix to a *file* the run reads (exclude list) does not
reach a run that was already rendered.

---

## 2. The guarded build traps at ~56472, deterministically

**Symptom.**

```
MoonEP planning: tokens_per_expert disagrees with topk, trapping:
  src_rank=0 expert=1 tokens_per_expert=0 topk_count=640
```

**Cause.** `dispatch_preprocess` built the dense `[S, k]` indices with
`probs.masked_fill(~routing_map, -1.0).topk(k)`, which fills a short routing row with the lowest
expert indices, while `tokens_per_expert` counted the map's real entries. The planner then wrote past
an expert's allocation: `cudaErrorIllegalAddress` on `ec3eb8b`, the trap on `e9e8238`.

**Fix.** `dense_topk_from_routing()`: pad a short row at weight 0 on one of its routed experts, take
the counts from the padded indices. Bit-identical on k-hot rows; MoonEP experts are SwiGLU without
bias, so a zero-weight slot is an exact +0.

**Cost.** ~**3 %** slower per step (18.84 s vs 18.35 s on the *same* iterations, `.2152` vs `.2226`).
Not the node, not the data window: measured on identical step numbers.

---

## 3. Config recipe lost in the successor submissions

**Symptom.** Every run from `.2171` (06:17) trained **dense attention with no packing plans**: 20.7 s/step
instead of 18.3 s, and a different data recipe.

**Cause.** The live run carried enqueue overrides the config file never had:
`page_sparse_backend=rowmask`, `page_sparse_per_group`, `page_sparse_selector=page_maxpool`,
`packing_plan_boundaries=group`, four `packing_plan_path` entries, `dataloader_index_bundle`,
`dataloader_inter_document_masking`, `CUDA_ENABLE_*COREDUMP*`, `non_persistent_save_interval=50`.

**Fix.** Copied verbatim from the frozen definition of `.2152` into
`config/experiments/korbi/recall_surgery/v4b_learned_fix_mn5.yaml` (commits `02665a20`, `2614f164`).

**Lesson.** A successor config must be diffed argument-by-argument against the run it replaces.
`diff` the rendered `step.sh` lists; do not trust the config file to be complete.

---

## 4. Rollback left a dangling rolling-checkpoint pointer

**Symptom.** Every rank exited 1, then a rendezvous keep-alive failure (`rc=143`):

```
FileNotFoundError: .../runs/v4b_learned/main/ckpt_rolling/iter_0056000
  at checkpointing.py:2281 _get_checkpoint_format
```

**Cause.** The rollback moved the wrong-recipe saves away **and** wrote
`ckpt_rolling/latest_checkpointed_iteration.txt = 56000`, pointing at a directory that no longer
existed. The loader prefers the rolling checkpoint over the persistent one.

**Fix.** Delete the rolling tracker when the rolling directory is empty; the loader then prints
"will not load any non-persistent checkpoint" and uses the persistent save.

---

## 5. NaN in the forward loss — hardware, not instability

**Symptom.**

```
RuntimeError: Rank 726, node as05r4b24, device 2, iteration 56469:
  Unexpected result nan (found NaN in local forward loss calculation)
```

and earlier `Rank 713, node as05r4b20, device 1, iteration 56721`.

**Fix.** Excluded `as05r4b24` and `as05r4b20` (same rack `as05r4b`). The run then crossed 56469 and
56472 with zero NaN, zero trap, zero illegal access.

**Lesson.** A single-device NaN names its node in the message. A *trap* names no node — do not exclude
hardware for a code-level failure.

---

## 6. `as02r1b08` mlx5_1:1 port flap

**Symptom.** `NCCL WARN NET/IB : mlx5_1:1 Got non-fatal async event: port error(10)`, then
`client reregistration(17)`, then `port active(9)` 67 minutes later. 16 events in `.2178`, 8 within
minutes in `.2186`. The pool's fabric rule restarted `.2178` at 10:38.

**Note.** `port active(9)` is the port coming **back**; the fault is `port error(10)`. Do not read the
recovery line as the failure.

**Fix.** Excluded `as02r1b08`.

---

## 7. `sinfo` / `scontrol show nodes` are denied — the pool is blind

**Symptom.**

```
slurm_load_node: Access/permission denied
```

The pool monitor logs `node scan without scheduler state` for the same reason, and `sick_nodes()` in
`scripts/pool/supervisor.sh` returns **nothing**.

**Consequence.** Every capacity decision the pool makes is blind to node state and GRES.

**Workaround.** Probe through `srun --overlap` inside the allocation: `nvidia-smi -L`, GPU memory,
and `/sys/class/infiniband/*/ports/*/state`. See `scripts/korbi/recall_surgery/probe_nodes_individual.sh`
(1-node steps, because a chunked step fails as a whole and names no node).

**Result of the 1-node probe (261 nodes).** 261/261 answered, all with 4 GPUs, 65247 MiB, IB norm
5 of 6 ports ACTIVE (`mlx5_3` is DOWN fleet-wide). **No GRES-limited node, no GPU loss, no bad IB.**

---

## 8. The exclusion lists snowballed and starved the host

**Symptom.** The host reported `idle ... 2` usable nodes, and the supervisor rejected every attempt:

```
run mn5-pool-p264.2337 needs 256 node(s), only 2 usable
→ rc=-1 (insufficient_nodes)
```

**Cause.** Two separate accumulations:

1. `hosts/host010/exclude_nodes` grew to **262 entries** (nearly the whole allocation) — the launch-time
   node check wrote its failures there. Reset to the 3 fault nodes.
2. The **run-level** `EXCLUDE` in `meta.env` grew to **326–327 entries**, of which 262 fell inside the
   allocation. The global file has only **35** entries (5 inside the allocation) and the resolver
   `oc_exclude_nodes()` returns exactly those 35, so the inflation happens in the submission path,
   not in the file.

**Fix applied.** Patched every live main job record to `slurm.exclude_file = v4b_exclude_5.txt` with
`sbatch.exclude` = the 5 nodes inside the allocation. A fresh submission then renders `EXCLUDE` with
35 entries (5 inside) → 259 usable → the run fits with 3 spare.

**Open.** The exact code path that inflates `meta.env`'s `EXCLUDE` to ~327 entries is **not yet
identified**. Until it is, any run created by a *restart* of an old record can inherit the inflated
list, and the fix is to re-render from a record whose `exclude_file` points at a short list.

---

## 9. Capacity arithmetic: 264 nodes, 256 needed, zero slack

```
264 allocation nodes
 −3 dropped by Slurm (as01r4b22, as01r4b31, as05r1b31) → 261
 −5 excluded inside the allocation (as02r1b08, as05r3b23, as05r3b24, as05r4b20, as05r4b24)
 = 256  → exactly the run's requirement
```

Zero spare means **one eval on any usable node makes the 256-node step impossible**, and one more
exclusion does too.

Also note: `as01r2b04` was removed from the exclude list (no documented reason, and the probe showed
4 healthy GPUs). That was worth exactly one node — the difference between 255 and 256.

---

## 10. Evals compete for the same host

**Symptom.** 18 eval runs (`ev_ambitious_v4b_learned_fp8_56000_*`, priority 55/60) queued into `p264`;
the 256-node main run (priority 150) could not be placed.

**Why priority alone does not save it.** The supervisor launches assignments in **rid order**, and the
evals have lower rids, so 1-node evals launch first and consume the slack the big run needs. The
monitor's assignment pass does sort by priority, but once a run is *assigned* it is never
re-evaluated.

**Fix used.** `max_runs_per_host=1` while bringing the main run up (only the main can be placed),
then raise it to 6 (main + 5 evals on the excluded nodes). Evals have `USE_EXCLUDED_NODES=1`, so they
can sit on the excluded nodes without hurting the run.

---

## 11. Duplicate main records (`_q2`, `_q3`, …)

**Symptom.** Three or more live records for `ambitious_v4b_learned_fix_256`, each spawning 256-node
runs; the queue showed several "ambitious" jobs.

**Cause.** Each `run_autoexp.py` submission with the same name creates a new record; the pool renames
the duplicates `_q2`, `_q3`, … but they all stay live. Cancelling a *run* does not cancel its record.

**Fix.** Cancel the **records**, not the runs: `pool_ctl <pool> cancel <record-name>`. Then submit once.

---

## 12. The restart churn loop (what "assigned, waiting, cancelled" actually was)

Per monitor poll (~34 s):

1. the record is restarted → a new run dir with a 256-node request;
2. the supervisor computes `usable_nodes` and finds fewer than 256 (snowballed exclude, §8);
3. it writes `rc=-1 insufficient_nodes`;
4. the monitor treats that as a failure and restarts the record → back to 1.

That is why the queue looked like a conveyor belt of `.2347 … .2352`, each dying within 35 s.

---

## Operational constants (for planning)

| constant | value |
|---|---|
| monitor poll | ~34 s (every `pool_ctl` request waits for it) |
| cancel grace / preempt grace | 120 s / 900 s |
| run startup | ~16 min (15.8–16.3 min measured) |
| per-step time | 18.84 s median (3 % slower than the pre-fix 18.35 s) |
| wall rate incl. saves | ~19.2 s/step in a clean window; a 50-step rolling save adds ~130 s |

---

## What is running at the end of this session

One clean main submission `ambitious_v4b_learned_fix_256_a001a9_q8` with:

- `pylibs_moonep_e9e8238` + `dense_topk_from_routing` (snapshot `autoexp_v4bfix2`),
- the full recipe (page-sparse `rowmask`/`page_maxpool`, four `v4b_p2` packing plans, index bundle,
  inter-document masking, core dumps, 50-step rolling saves),
- `--exclude` = 35 entries (5 inside the allocation),
- `max_runs_per_host=1` so no eval can take the host while it comes up,
- resumes from rolling **56950**.

## Rules that would have saved the day

1. **Never re-render from memory.** Diff the rendered `step.sh` of the successor against the run it
   replaces, argument by argument.
2. **Fix the source, not the instance.** Exclude lists, `PYTHONPATH` and configs are read at render
   time; editing a file that an already-rendered run reads does nothing.
3. **One record per run name.** Cancel records, not runs.
4. **Check the arithmetic before preempting anything**: `allocation − union(exclusions) ≥ nodes`.
5. **A failure that names no node is not a hardware fault.** A NaN or a watchdog line names its node;
   a trap, a `FileNotFoundError` or `insufficient_nodes` does not.
6. **Probe nodes one at a time** when GRES is suspect; a chunked step hides the node that matters.
7. **Do not preempt the main run** once it is placed — only if nodes are genuinely insufficient or
   failing.

---

# Addendum — the root causes found last, and the happy path

## 13. Two of my own scripts were fighting each other

`supervise_main.sh` (started 19:53) evicts "runs holding a usable node" — and once the main run was
placed it held 256 usable nodes, so the evictor preempted **the main run itself**. At the same time
two copies of `reset_pool_and_start.sh` ran concurrently, each cancelling everything and submitting
one main run, which is where the duplicate `ambitious` records came from.

**Rule.** One supervisor at a time, and an evictor must never name the run it is trying to place.
Every long-running script needs a single-instance lock (`flock` on a lock file in the pool dir).

## 14. `meta.env` carried TWO `EXCLUDE=` lines — the supervisor read the wrong one

This was the real cause of `needs 256 node(s), only 2 usable`:

```
EXCLUDE=<35 entries>    <- the run's own list (correct)
EXCLUDE=<292 entries>   <- appended by the restart hook's exclusion refresh
```

The pool writes `meta.env` more than once (submit, then the exclusion refresh) and the writer
**appends** instead of replacing. The supervisor sources the file with `. meta.env`, so the **last**
assignment wins: 292 entries, ~259 of them inside the allocation → `usable_nodes` = 2.

**Fix applied.** `scripts/korbi/recall_surgery/dedupe_meta.sh` keeps the first occurrence of every key
(run as a background task while the pool is live). **Proper fix:** `write_env_file` must replace keys,
never append.

## 15. `srun: Requested node configuration is not available` — a stale `ALL_NODES`

The supervisor's `ALL_NODES` came from `info.env` (264 nodes). Slurm has since dropped three
(`as01r4b22`, `as01r4b31`, `as05r1b31`), leaving 261. When the supervisor picks 256 nodes from a
264-node list it can include a dropped node, and `srun` refuses the whole step:

```
srun: error: Unable to create step for job 48867560: Requested node configuration is not available
```

**Fix.** `info.env` must be rewritten from the job's live node list (`squeue -j <job> -o %N`) after any
resize, and `refresh_alloc` must actually shrink `ALL_NODES` (it keeps the live subset only).

---

# The happy path (what worked, and it took one command)

This is the sequence that had the main run training within seconds, and it is the only sequence that
worked all day:

```
1. cancel the LINE, not the runs        # cancel_line.sh <pool> <name>
   — every live record is cancelled, so the monitor has nothing to restart
2. max_runs_per_host = 1                # nothing else can take the host while it comes up
3. submit exactly ONE job               # one record, one run, fresh render
4. host idle + 261 usable − 5 excluded = 256  → assigned → started in seconds
5. first iteration ~16 min later        # 15.8–16.3 min, 18.84 s/step
6. max_runs_per_host = 6                # main + 5 evals on the excluded nodes
```

Every failure today came from violating one of those six lines.

---

# Proposed system

## Semantics

**A *line* is the unit.** `(pool, definition-name)` owns its records, its runs and its restart policy.
No command should ever act on a single record or a single run of a line without saying so.

| command | semantics |
|---|---|
| `cancel <line>` | **finally stop**: every live record of the line becomes `final_state=cancelled`, its runs get a `cancel` marker, the monitor never restarts it. Cascading and atomic (one monitor poll). |
| `replace <line> <spec>` | submit the new record and **auto-cancel every older live record of the line in the same transaction** — at most one record per line is ever live. This is what `_q2`, `_q3` should have been. |
| `restart <line>` | allowed only per the record's policy; a **placement failure is not a run failure** (see below). |
| `preempt <line>` | explicit user action only; refused for a `protected` line once it holds its nodes. |
| `status <line>` | one line: live record, live run, nodes, capacity, exclusions, last iteration, ETA. |

## Rules that must be enforced by the tool, not by the operator

1. **Cancel means cancel, at both levels.** A cancelled run must not be restartable; a cancelled
   record must not be re-submitted. Today a cancelled *run* was restarted on the next poll because
   its record stayed live.

2. **One live record per line.** Any submit replaces the predecessor. The pool's dedupe renaming
   (`_q2`, `_q3`) is a bug, not a feature: it turns one logical job into N competing jobs.

3. **Placement failure ≠ run failure.** `insufficient_nodes` and `configuration is not available` mean
   *the scheduler could not place the run*. They must not consume a restart attempt, must not create a
   new run dir on every poll, and must not trigger the `retry_failed` action. The record stays queued
   and the scheduler waits for capacity, logging one line per change.

4. **Reserve before filling.** When a run of priority P needs N nodes, the scheduler reserves N nodes
   before placing lower-priority runs, and launches **largest-first**. Today the supervisor launched
   assignments in rid order, so 1-node evals took the slack a 256-node run needed.

5. **Capacity is a precondition, checked before any action.**
   `feasible = |allocation| − |union(exclusions) ∩ allocation| ≥ nodes`. If false, refuse the submit
   and say which side is short. No preempting, no churn.

6. **One exclusion set per host.** Merge pool + host + run exclusions once, intersect with the live
   allocation, write one file, and have every consumer read that file. Never store an inline
   exclusion blob in a record, and never append a second `EXCLUDE=` line.

7. **`meta.env` is written once and replaced, never appended.** One line per key. A reader that
   sources the file must never be able to see a stale duplicate.

8. **`info.env` follows the job.** After any resize, rewrite `NUM_NODES`/`NODELIST` from
   `squeue -j <job> -o %N`; `ALL_NODES` shrinks with it.

9. **One supervisor, one lock.** Every long-running control script takes a `flock`; an evictor never
   names the run it is placing. Killing the monitor or a supervisor is a first-class command with a
   pid/host file, not a `pkill` guess.

10. **Protected lines.** The main training line is `protected: true`: no automatic preemption once it
    has its `started` marker. Only an explicit user action or a named node fault may stop it.

11. **Every state change is one poll.** `freeze + evict + place` must be a single monitor action, not
    a chain of requests that each cost 34 s. Today a manual restart cost minutes of pure poll latency.

## What this would have changed today

| today | with the system |
|---|---|
| 3 competing ambitious records | 1 (replace semantics) |
| cancel left a restartable record | cancel is final |
| `insufficient_nodes` restarted every 34 s | one log line, no new run |
| evals stole the host | nodes reserved, largest-first |
| supervisor preempted the main run | protected line, evictor skips itself |
| `meta.env` read a 292-entry EXCLUDE | one line per key |
| `srun: configuration not available` | `info.env` follows the job |
| manual freeze/clear/unfreeze dance | one `replace` call |
