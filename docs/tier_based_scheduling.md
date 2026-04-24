# Tier-based scheduling across multiple invocations

How to split a single sweep config into **prioritized, independently-launched
subsets** without duplicating work or breaking `${sibling.*}` references.

Worked example used throughout:
`config/experiments/swagatam/multilingual_scaling/dense_130M_lr_gbsz_tokens_grid.yaml`.

---

## 1. Problem statement

A multi-stage sweep (stable → decay) produces one big plan. When you launch
it, every job is submitted at once — SLURM eventually orders them, but you
have no lever to say "please run the 9 **center** decays first, then the 36
**cross** decays, then the 36 **diagonals**."

Goals:

1. **Prioritize** some decays over others (center → cross → diagonal).
2. Do this with **separate invocations** so each is a standalone script you
   can queue or cancel independently.
3. **Never duplicate the shared stables** (20 of them, shared across all
   81 decays). Each stable is submitted by exactly one invocation, chosen
   so the highest-priority tier doesn't end up queuing stables whose decays
   are all lower-priority.
4. Keep decays' `${sibling.stable.*}` references working even when stables
   are launched by a *different* invocation.

---

## 2. The key insight: siblings resolve from the full plan, always

```python
# oellm_autoexp/orchestrator.py
jobs = resolve_sweep_with_dag(
    root,
    points_by_idx,
    config_setup=config_setup,
    config_class=RootConfig,
    full_points_by_idx=all_points_by_idx,   # <- always the full set
)
```

And inside the resolver:

```python
# oellm_autoexp/hydra_staged_sweep/dag_resolver.py
lookup_points = full_points_by_idx if full_points_by_idx is not None else points_dict
sibling_index = _build_sibling_index(lookup_points)
```

So the sibling DAG is built over **every expanded sweep point**, regardless of
which subset you intend to submit. The `sweep.filter` only runs *after* each
point has been fully resolved (see `docs/sweep_resolution_ordering.md`).
Consequently:

- A decay point's `${sibling.stable.*}` interpolation resolves to the
  stable's real, fully-materialised config even when the stable will be
  filtered out of the submission list.
- Filesystem paths (`job.base_output_dir`, `job.log_path_current`) are
  deterministic functions of `job.name`, so they agree across invocations.

Together these mean: if invocation A submits the stables and invocation B
submits some decays, B's decays produce *exactly the same* `load=...`,
`start_condition.path=...`, and `cancel_condition.log_path=...` strings they
would have produced in a single combined run. Coordination is entirely via
the shared filesystem.

---

## 3. The pattern: one YAML, one `backend.megatron.aux.priority_tier` variable

### The knob

The tier knob and the per-combo tier metadata both live under
`backend.megatron.aux.*` — a free-form `dict[str, Any]` that OmegaConf is
happy to resolve against, and the same namespace that already holds
`aux.tokens`, `aux.start_iter`, etc. Keeping everything together avoids the
awkward "top-level `aux.priority_tier` vs nested `backend.megatron.aux.tokens`"
split.

> Earlier drafts of this pattern put `priority_tier` under the top-level
> `aux: dict[str, Any]` field on `RootConfig`, because `compoconf.parse_config`
> is strict about unknown top-level keys. That still works, but putting
> `priority_tier` next to the rest of the sweep-overridable aux fields under
> `backend.megatron.aux` is strictly more consistent and has no downside:
> Hydra overrides accept the longer dotted path, and the filter expression
> simply uses `${backend.megatron.aux.priority_tier}` throughout.

```yaml
# In the base config, as a sibling of backend.megatron.aux.tokens et al.
backend:
  megatron:
    aux:
      tokens: 300_000_000_000
      priority_tier: all   # overridden from the CLI
      # ...plus stable_launch_tier, *_tokens_set, save_step_*, ...
```

Valid values enumerate the sets you want to launch independently. For the
`dense_130M_lr_gbsz_tokens_grid.yaml` example: `all`, `center`, `cross`,
`diagonal`.

From the CLI it is overridden via Hydra dotted syntax:
`backend.megatron.aux.priority_tier=center`.

### Per-combo tier membership and stable ownership

Each Group-1 hyperparameter combo carries **one Python-set-literal string
<<<<<<< HEAD
per tier** *and* a `stable_launch_tier` that names the tier responsible for
submitting this combo's stable:
=======
per tier**:
>>>>>>> exp_diana

```yaml
- backend.megatron.lr: 1.e-3
  backend.megatron.global_batch_size: 128
  backend.megatron.aux.tokens: 300_000_000_000              # max_decay_tokens
  backend.megatron.aux.center_tokens_set:   "set()"
  backend.megatron.aux.cross_tokens_set:    "{12_000_000_000, 20_000_000_000, 30_000_000_000, 50_000_000_000}"
  backend.megatron.aux.diagonal_tokens_set: "{80_000_000_000, 120_000_000_000, 200_000_000_000, 300_000_000_000}"
<<<<<<< HEAD
  backend.megatron.aux.stable_launch_tier:  cross
=======
>>>>>>> exp_diana
```

The three sets' union is the combo's full allowed decay set. `"set()"` is
used for empty — *not* `"{}"`, which is a dict literal in Python and breaks
`set | dict` unions.

`stable_launch_tier` is, by convention, the **earliest** tier (in the order
center → cross → diagonal) whose `*_tokens_set` is non-empty for that combo.
Why not just always launch stables with `center`? The 130M grid has stables
whose decays live *only* in `cross` and/or `diagonal` — e.g. the
`gbsz=512, 300BT` stables feed only diagonal decays. Launching all 20
stables eagerly under `priority_tier=center` would queue those large,
low-priority jobs ahead of actual center work. Assigning each stable to
exactly one tier keeps `center` lean (4 stables + 9 decays) and defers
non-critical stable capacity to the cross/diagonal invocations.

### Tier-aware filter

For brevity, the block below elides the `backend.megatron.` prefix; every
`aux.*` reference in the actual filter is the full
`${backend.megatron.aux.*}` interpolation.

```yaml
filter: "\\${oc.eval:'(
    (\"\\${stage}\" == \"stable\" and
        (\"\\${aux.priority_tier}\" == \"all\"
         or \"\\${aux.priority_tier}\" == \"\\${aux.stable_launch_tier}\"))
    or
    (\"\\${stage}\" != \"stable\" and (
        (\"\\${aux.priority_tier}\" == \"all\"
            and \\${aux.tokens} in
                \\${aux.center_tokens_set}
                | \\${aux.cross_tokens_set}
                | \\${aux.diagonal_tokens_set})
        or (\"\\${aux.priority_tier}\" == \"center\"
            and \\${aux.tokens} in \\${aux.center_tokens_set})
        or (\"\\${aux.priority_tier}\" == \"cross\"
            and \\${aux.tokens} in \\${aux.cross_tokens_set})
        or (\"\\${aux.priority_tier}\" == \"diagonal\"
            and \\${aux.tokens} in \\${aux.diagonal_tokens_set})
    ))
)'}"
```

(Written on one line in the actual YAML; broken here for readability.)

Reading it:

<<<<<<< HEAD
- **Stables** survive the filter iff
  `aux.priority_tier == "all"` OR `aux.priority_tier == aux.stable_launch_tier`.
  Each stable is therefore launched by exactly one non-`all` invocation
  (the tier it is owned by).
=======
- **Stables** survive the filter iff `priority_tier ∈ {"all", "center"}`.
  This guarantees the 20 stables are launched by exactly one invocation.
>>>>>>> exp_diana
- **Decays** survive iff their `aux.tokens` sits in the tier's per-combo
  set (for `tier=all`, the union of all three).

All five fields (`priority_tier`, `stable_launch_tier`, `tokens`, and the
three `*_tokens_set` sets) live in the same `backend.megatron.aux` dict, so
there is no "two different `aux`es" gotcha — the whole tier surface area
is one consistent namespace.

Python precedence detail: `in` is a comparison (precedence lower than `|`),
<<<<<<< HEAD
so `x in A | B | C` parses as `x in (A | B | C)`. `set() | {…}` is a no-op
union, so empty-tier combos contribute nothing.
=======
so `x in A | B | C` parses as `x in (A | B | C)`. That's what we want.
`set() | {…}` is a no-op union, so empty-tier combos contribute nothing to
either the per-tier check or the `all`-tier union.
>>>>>>> exp_diana

---

## 4. Launch workflow

```bash
# 1) 4 center stables + 9 center decays
python scripts/run_autoexp.py \
    --config-name=experiments/.../dense_130M_lr_gbsz_tokens_grid \
    backend.megatron.aux.priority_tier=center

# 2) 10 cross stables + 36 cross decays (decays wait on sibling stables'
#    branching checkpoints)
python scripts/run_autoexp.py \
    --config-name=experiments/.../dense_130M_lr_gbsz_tokens_grid \
    backend.megatron.aux.priority_tier=cross

# 3) 6 diagonal stables + 36 diagonal decays
python scripts/run_autoexp.py \
    --config-name=experiments/.../dense_130M_lr_gbsz_tokens_grid \
    backend.megatron.aux.priority_tier=diagonal
```

Each invocation:

- Expands the full sweep plan (200 points, always).
<<<<<<< HEAD
- Applies the filter to pick its tier's subset (13 / 46 / 42 jobs for
  center / cross / diagonal).
- Spawns its own orchestrator/monitor process.
- Submits its jobs to SLURM (stables and their tier's decays go immediately;
  decays gated by `FileExistsCondition` wait until the sibling stable writes
  its branching checkpoint).
=======
- Applies the filter to pick its tier's subset (29 / 36 / 36 jobs).
- Spawns its own orchestrator/monitor process.
- Submits its jobs to SLURM (either immediately for stables and centers, or
  waits on `FileExistsCondition` for cross/diagonal decays).
>>>>>>> exp_diana

You **must** launch the three tiers in order (center → cross → diagonal) and
cannot skip any of them: every tier owns some stables, and skipping a tier
leaves its stables unsubmitted — the decays in *that* tier would then wait
forever on `FileExistsCondition`. Use
`backend.megatron.aux.priority_tier=all` if you want a single-shot
submission of the full 101 jobs.

---

## 5. Partition sanity check

Per-tier counts for the 130M lr × gbsz × tokens sweep:

| `backend.megatron.aux.priority_tier` | stable | decay | total |
|---|---|---|---|
| `all`      | 20 | 81 | 101 |
<<<<<<< HEAD
| `center`   |  4 | 9  | 13  |
| `cross`    | 10 | 36 | 46  |
| `diagonal` |  6 | 36 | 42  |

`13 + 46 + 42 = 101`, matching the default `all`. Stables partition as
`4 + 10 + 6 = 20` and decays as `9 + 36 + 36 = 81`. The three non-`all`
tiers are disjoint, so back-to-back submission of all three is bit-identical
to a single `all` invocation (same jobs, same parameters, same checkpoints).

Which stables belong to which tier:

- **center** (4): `(gbsz=32, lr=1e-3)`, `(gbsz=64, lr=1e-3)`,
  `(gbsz=128, lr=2e-3)`, `(gbsz=256, lr=2e-3)`.
- **cross** (10): every combo whose `center_tokens_set` is empty but
  `cross_tokens_set` is not — e.g. `(gbsz=16, lr=1e-3)`,
  `(gbsz=32, lr=5e-4 / 2e-3)`, `(gbsz=64, lr=5e-4 / 2e-3)`,
  `(gbsz=128, lr=1e-3 / 4e-3)`, `(gbsz=256, lr=1e-3 / 4e-3)`,
  `(gbsz=512, lr=2e-3)`.
- **diagonal** (6): combos where only `diagonal_tokens_set` is non-empty —
  `(gbsz=16, lr=5e-4 / 2e-3)`, `(gbsz=64, lr=4e-3)`, `(gbsz=128, lr=5e-4)`,
  `(gbsz=512, lr=1e-3 / 4e-3)`.

### Verification script

Run it directly:

```bash
python scripts/validate_lr_gbsz_tokens_grid_tiers.py
```

Expected output (and what the script asserts):

```
Pre-filter sweep points: 200
  tier=all       stables= 20  decays= 81  total=101
  tier=center    stables=  4  decays=  9  total= 13
  tier=cross     stables= 10  decays= 36  total= 46
  tier=diagonal  stables=  6  decays= 36  total= 42

All assertions passed.
```

Source: `scripts/validate_lr_gbsz_tokens_grid_tiers.py`. It expands the
sweep, simulates the filter in Python for each tier, and asserts the
partition invariants (disjoint, `center ∪ cross ∪ diagonal == all`).

=======
| `center`   | 20 | 9  | 29  |
| `cross`    | 0  | 36 | 36  |
| `diagonal` | 0  | 36 | 36  |

`29 + 36 + 36 = 101`, matching the default `all`. The three partitioned
tiers are disjoint, so back-to-back submission of all three is bit-identical
to a single `all` invocation (same jobs, same parameters, same checkpoints).

Verification script:

```python
from types import SimpleNamespace
import yaml
from oellm_autoexp.hydra_staged_sweep.expander import expand_sweep

with open("config/experiments/swagatam/multilingual_scaling/dense_130M_lr_gbsz_tokens_grid.yaml") as f:
    cfg = yaml.safe_load(f)
sw = cfg["sweep"]
sw_obj = SimpleNamespace(type=sw["type"], groups=sw["groups"], filter=sw.get("filter"),
                         base_values={}, list_composition=[])
points = expand_sweep(sw_obj)  # -> 200

def keep(params, tier):
    stage = params["stage"]
    tokens = int(params["backend.megatron.aux.tokens"])
    c = eval(params["backend.megatron.aux.center_tokens_set"])
    x = eval(params["backend.megatron.aux.cross_tokens_set"])
    d = eval(params["backend.megatron.aux.diagonal_tokens_set"])
    if stage == "stable":
        return tier in {"all", "center"}
    return tokens in {"all": c | x | d, "center": c, "cross": x, "diagonal": d}[tier]

for tier in ("all", "center", "cross", "diagonal"):
    n = sum(keep(p.parameters, tier) for p in points)
    print(tier, n)
```

>>>>>>> exp_diana
---

## 6. How decay-stable coordination works across invocations

A decay's config after resolution looks like:

```yaml
backend.megatron.load: "<base_output_dir_of_sibling_stable>/checkpoints"
backend.megatron.ckpt_step: <start_iter>

job.start_condition:
  class_name: FileExistsCondition
  path: "<base_output_dir_of_sibling_stable>/checkpoints/iter_NNNNNNN"

job.cancel_condition:
  class_name: LogPatternCondition
  log_path: "<base_output_dir_of_sibling_stable>/current.log"   # log_path_current
  pattern: "FATAL ERROR|OutOfMemoryError|Traceback"
```

Each of those values is computed from the stable's *config*, not its runtime
state. So the cross and diagonal orchestrators produce the same strings the
center orchestrator would have.

Runtime sequence:

1. Center orchestrator submits stable → Megatron writes
   `.../checkpoints/iter_NNNNNNN` once it hits `save_extra_steps[N]`.
2. Center orchestrator also maintains `current.log` as a symlink to the
   active `slurm-<jobid>-....log` (see `oellm_autoexp/monitor/slurm_client.py`
   and `docs/log_symlink_usage.md`).
3. Cross/diagonal orchestrators' monitor loops tick, evaluate each decay's
   `FileExistsCondition` against the filesystem. No IPC with the center
   orchestrator is needed — they're independent processes.
4. If the stable fails, `LogPatternCondition` watches `current.log` for
   error patterns and cancels the downstream decays — this also works
   cross-process because the log path is filesystem-resident.

---

## 7. Design alternatives considered

| Option | Why not chosen |
|---|---|
| **Three separate YAMLs** (center/cross/diagonal) each duplicating the base | Heavy duplication. Any change to model/data config has to be mirrored three times. No upside over `defaults`-based sharing, which itself is awkward because overriding specific sweep-group entries via `defaults` is painful. |
| **Single YAML + SLURM `--nice` per grid_position** | Still a single invocation, so no way to gate "don't even queue the diagonals until I say so." Doesn't satisfy the "three scripts" requirement. Works fine as an *addition* on top of tiering, e.g. `slurm.sbatch.nice=-50` on the center invocation. |
| **Single YAML + `subset_indices` CLI override** | Requires the user to translate `(lr, gbsz, tokens, tier)` tuples into sweep-point indices — brittle and needs recomputing whenever the sweep shape changes. |
| **Hardcoded `load:` paths + no stables in cross/diagonal YAMLs** | The sibling mechanism already does this for us, *because `full_points_by_idx` means stables are always resolvable*. Hardcoding the path template adds maintenance burden (if you rename `job.name` you have to fix both the stable and the hardcoded string). |

The chosen approach (single YAML +
`backend.megatron.aux.priority_tier`) wins because:

- Stables, their decays, and all sibling references live together in one
  place. No duplication.
- Adding a new tier is one new `*_tokens_set` field per combo and one new
  clause in the filter.
- `backend.megatron.aux.priority_tier=all` remains a valid,
  single-invocation mode — identical to pre-tiering behavior.

---

## 8. Caveats and operational notes

- **Tier launch order is mandatory, and skipping is not allowed.** Because
  each stable is owned by exactly one tier, invoking `cross` without
  `center` submits the 10 cross-owned stables but leaves the 9 center decays
  without their parents (center stables are never launched). Invoking only
  `diagonal` leaves 14 of 20 stables unsubmitted entirely. The safe
  sequences are `center → cross → diagonal` (in that order) or a single
  `backend.megatron.aux.priority_tier=all`. You can overlap the three
  invocations in wall time (cross orchestrator can be started as soon as
  center is underway; see next bullet), but you cannot drop any of them.
- **The per-tier orchestrator must stay alive** until every decay it owns
  has been submitted — each decay waits on the branching checkpoint of its
  sibling stable, which is polled by whichever orchestrator owns the decay.
  Ctrl-C'ing early strands pending decays (stables already dispatched to
  SLURM keep running). Re-launching the same
  `backend.megatron.aux.priority_tier=<tier>` picks up the remaining work
  because already-submitted jobs are recognised via the monitor state store.
- **Safe to parallelise the three invocations.** They work on disjoint
  job-name sets and can each run their own orchestrator concurrently.
<<<<<<< HEAD
  Because stables are now distributed across tiers, cross and diagonal can
  be started as soon as you're happy for their stables to start filling
  the queue — they don't need to wait for the center decays to finish.
=======
>>>>>>> exp_diana
- **Monitor state.** Each invocation writes its own
  `monitor_state/<run_id>/` folder. They don't share state, so if you want
  to inspect "one dashboard of 101 jobs" you have to aggregate across the
  three folders.
- **Combining with SLURM-level nice.** If you want to bias the queue when
  invocations overlap, stack `slurm.sbatch.nice` on top of the tier:
  `backend.megatron.aux.priority_tier=center slurm.sbatch.nice=-50` makes
  centers jump queue ahead of a later
  `backend.megatron.aux.priority_tier=diagonal slurm.sbatch.nice=+100`
  submission.
- **Strict serialisation.** If you *must* prevent lower-priority tiers from
  starting until higher-priority ones finish (not just "from being
  submitted"), add a sentinel file written by the last high-priority job
  and an extra `FileExistsCondition` gating the low-priority ones. In
  practice this wastes cluster capacity and is rarely worth it.

---

## 9. Recipe: adapting this to a new sweep

1. **Enumerate tiers.** Decide the subsets you want to submit independently.
   Tiers don't have to be symmetric (e.g. one tier could be "smoke-test
   combos" and another "full grid").
2. **Tag each decay point with exactly one tier.** For the 130M grid this
   came from the CSV's `grid_position` column. For a new sweep, enumerate
   `(lr, gbsz, tokens) → tier` however makes sense.
3. **Compute per-combo per-tier token sets.** Each Group-1 entry gets
   `aux.<tier>_tokens_set` string literals; their union covers the combo's
   full allowed decay set.
4. **Pick each combo's `stable_launch_tier`.** Set it to the earliest tier
   (in your priority ordering) whose `*_tokens_set` is non-empty for that
   combo. That tier's invocation will own launching the stable. This
   ensures lower-priority stables don't sneak into a higher-priority tier's
   queue.
5. **Add `priority_tier: <default>`** to the `backend.megatron.aux:`
   mapping (alongside the other sweep-overridable aux fields like `tokens`
   and `stable_launch_tier`). `all` is a reasonable default that keeps the
   old non-tiered behaviour. Avoid inventing a bare top-level key —
   `RootConfig` is a strict dataclass and will reject unknown top-level
   fields. The top-level `aux: dict[str, Any]` field on `RootConfig` works
   too, but putting the knob under `backend.megatron.aux` keeps all
   tier-related fields in a single consistent namespace.
6. **Write the tier filter.** Template:
   ```
<<<<<<< HEAD
   (stage == stable AND (aux.priority_tier == "all"
                         OR aux.priority_tier == aux.stable_launch_tier))
   OR
   (stage != stable AND (
       (aux.priority_tier == "all"   AND tokens in union(all tier sets))
       OR (aux.priority_tier == T    AND tokens in <T>_tokens_set)    for each T
   ))
   ```
7. **Verify partition.** Adapt `scripts/validate_lr_gbsz_tokens_grid_tiers.py`
   and confirm:
   - stables partition: `sum(stables per non-all tier) == len(all stables)`,
   - decays partition: `sum(decays per non-all tier) == len(all decays)`,
   - union invariant: `center ∪ cross ∪ ... == all` point-for-point.
   Any mismatch means a combo is missing a `stable_launch_tier` or your
   per-tier sets overlap/miss rows.
8. **Document the mandatory launch order** (and the "don't skip tiers"
   invariant) in the config's trailing comment block. Future-you will
   thank you.
=======
   (stage == stable AND priority_tier ∈ stable_tiers)
   OR
   (stage != stable AND (
       (priority_tier == "all"   AND tokens in union(all tier sets))
       OR (priority_tier == T    AND tokens in <T>_tokens_set)    for each T
   ))
   ```
6. **Verify partition.** Run the script in §5 and confirm
   `sum(non-all tiers) == len(all)`. Any mismatch means your per-tier sets
   overlap or miss rows.
7. **Document the invocation commands** in the config's trailing comment
   block. Future-you will thank you.
>>>>>>> exp_diana

---

## 10. Related reading

- `docs/lr_gbsz_tokens_grid_sweep.md` — the worked example this doc refers
  to throughout.
- `docs/multi_stage_design.md` — pull-based stable→decay linkage.
- `docs/sweep_resolution_ordering.md` — DAG resolution order and sibling
  lookup internals.
- `docs/escaped_interpolation_design.md` — why `\\${…}` is needed inside
  sweep filters and group configs.
- `docs/log_symlink_usage.md` — how `log_path_current` stays valid across
  SLURM restarts; relevant for `cancel_condition` across invocations.
