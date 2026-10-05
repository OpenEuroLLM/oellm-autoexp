# Multi-budget `(lr × gbsz × tokens)` grid sweep

Design notes for `config/experiments/swagatam/multilingual_scaling/dense_130M_lr_gbsz_tokens_grid.yaml`.

The goal: run a 130M dense model across **9 token budgets** (6B … 300B) with a
**different 3×3 `(lr, gbsz)` sub-grid per budget**, while reusing each stable
training run for as many decay jobs as possible. Grid membership is defined by
`mls/selected_ml_grid.csv`.

---

## 1. The grid, at a glance

Each token budget picks 3 learning rates × 3 batch sizes, shifting diagonally
with scale (larger budgets → larger batches, optionally larger LR):

| budget       | lrs                      | gbsz              |
|--------------|--------------------------|-------------------|
| 6B           | `{5e-4, 1e-3, 2e-3}`     | `{16, 32, 64}`    |
| 12B          | `{5e-4, 1e-3, 2e-3}`     | `{32, 64, 128}`   |
| 20B/30B/50B  | `{1e-3, 2e-3, 4e-3}`     | `{64, 128, 256}`  |
| 80B/120B/200B/300B | `{1e-3, 2e-3, 4e-3}` | `{128, 256, 512}` |

9 budgets × 9 cells = **81 decay runs** (= 81 CSV rows). Across those
rows, **20 distinct `(lr, gbsz)` pairs** appear. Each pair's *max* decay
budget determines how long its stable-LR training has to run.

| gbsz | lrs | max decay tokens |
|---|---|---|
| 16  | `{5e-4, 1e-3, 2e-3}` | 6B  |
| 32  | `{5e-4, 1e-3, 2e-3}` | 12B |
| 64  | `{5e-4}`             | 12B |
| 64  | `{1e-3, 2e-3, 4e-3}` | 50B |
| 128 | `{5e-4}`             | 12B |
| 128 | `{1e-3, 2e-3, 4e-3}` | 300B |
| 256 | `{1e-3, 2e-3, 4e-3}` | 300B |
| 512 | `{1e-3, 2e-3, 4e-3}` | 300B |

**Target:** 20 stable + 81 decay = **101 jobs**.

---

## 2. Why Approach A ("enumerate 20 valid combos")

Two designs were possible:

- **(A)** Enumerate the 20 valid `(lr, gbsz, max_decay_tokens)` as a `list`
  group; product with 10 stages (1 stable + 9 decay); **filter** drops the
  invalid decays only. Expansion: 200 points → 101 kept.
- **(B)** Product over the full outer space `all_lrs=[2.5e-4…8e-3] × all_gbsz=[16…1024]`
  = 42 combos × 10 stages = 420 points, then filter ~76 % away.

We chose **(A)**. Benefits:

1. Half the sweep points to materialise.
2. `max_decay_tokens` can be a plain per-combo scalar — no brittle big
   `oc.eval` lookup table.
3. The explicit list is the source of truth; reading the YAML is easier than
   reverse-engineering a filter.
4. No "phantom" points with `max_decay_tokens = 0` that could cause
   interpolation errors before the filter runs (see §6 for why this
   matters — the filter is evaluated **after** full OmegaConf resolution).

---

## 3. Sweep structure

```
sweep:
  type: product
  groups:
    - type: list    # Group 0: dynamic job.name / wandb_exp_name
    - type: list    # Group 1: 20 (lr, gbsz, max_decay_tokens, allowed_set) combos
    - type: list    # Group 2: 10 stages (1 stable + nested list of 9 decays)
  filter: "<stage-aware CSV membership check>"
```

The outer `product` multiplies Groups 1 × 2 → 200 sweep points. Group 0
contains a single config, so it just contributes job-naming overrides.

### Group 1: per-combo parameters

Each entry sets four fields:

```yaml
- backend.megatron.lr: 1.e-3
  backend.megatron.global_batch_size: 64
  backend.megatron.aux.tokens: 50_000_000_000                       # max decay tokens
  backend.megatron.aux.allowed_decay_tokens_set: "{6_000_000_000, 12_000_000_000, 20_000_000_000, 30_000_000_000, 50_000_000_000}"
```

Two subtle points:

- `aux.tokens` holds the **combo's max decay tokens**. Stable stage leaves
  it alone (so stable trains to exactly `max_decay_tokens`); decay stages
  **override** it to their specific budget.
- `allowed_decay_tokens_set` is a **string** whose content is a Python set
  literal. When interpolated into an `oc.eval` expression it's substituted
  verbatim — Python then parses the braces as a `set`. We use a string
  (not a YAML list) to guarantee literal insertion without any OmegaConf
  container → string coercion surprise.

### Group 2: stages

```
- stage: stable
  <computed train_iters, start_iter, save_step_br_*, save_step_end_*>
- type: list
  defaults: <shared decay config: load path, start_condition, cancel_condition, ...>
  configs:
    - stage: decay6BT   -> aux.tokens = 6_000_000_000
    - stage: decay12BT  -> aux.tokens = 12_000_000_000
    …
    - stage: decay300BT -> aux.tokens = 300_000_000_000
```

All 9 decay stages share a `defaults` block (checkpoint loading, start
gating, cancel logic); each config line only specifies `stage` and
`aux.tokens`.

---

## 4. Escaped OmegaConf interpolations

Throughout the sweep, formulae are written as *escaped* interpolations
— i.e. **double-backslash-dollar** (`\\$`):

```yaml
backend.megatron.train_iters: "\\${oc.eval:'(\\${backend.megatron.aux.tokens}+…)//…'}"
```

Why: Hydra/OmegaConf resolves `${…}` **eagerly** while expanding the base
config. The sweep mechanism must preserve these expressions until **after**
sweep point values (like `aux.tokens`) have been injected. Writing `\\${…}`
in YAML survives parsing as `\${…}`, which OmegaConf treats as a **literal**
`${…}` string. It gets resolved only when the fully-merged per-point
config is rendered.

Rule of thumb: any formula referencing sweep-varying values must be
escaped; anything referencing only base-config constants can use `${…}`
directly.

---

## 5. Core formulae (seq_length = 4096, `decay_fraction_cooldown` = 0.2)

For every combo and every target-budget `B` tokens:

```
budget_iters(B)      = ceil(B / (seq_length × gbsz))
train_iters          = ceil(aux.tokens / (seq_length × gbsz))
lr_wsd_decay_iters   = int(train_iters × decay_fraction)        # 0 in stable, 20% in decay
start_iter           = int(train_iters × (1 - decay_fraction))  # stable exits here; decay loads here
save_step_br_N       = int(budget_N_iters × 0.8)                # iter where decay branches off
save_step_end_N      = budget_N_iters                           # iter where stable matches that budget
```

Putting it together for stable runs (rounded to integers):

| gbsz | max_decay | train_iters |
|------|-----------|-------------|
| 16   | 6B        |  91,553     |
| 32   | 12B       |  91,553     |
| 64   | 12B       |  45,777     |
| 64   | 50B       | 190,735     |
| 128  | 12B       |  22,889     |
| 128  | 300B      | 572,205     |
| 256  | 300B      | 286,103     |
| 512  | 300B      | 143,052     |

Branching checkpoints for a given budget are at 80 % of `budget_iters(B)`.
E.g. at gbsz = 128 the 50B branching iter is
`int(ceil(50e9 / 524_288) × 0.8) = int(95_367 × 0.8) = 76_293`.

---

## 6. Filtering

```yaml
filter: "\\${oc.eval:'\"\\${stage}\" == \"stable\" or
        \\${backend.megatron.aux.tokens}
        in \\${backend.megatron.aux.allowed_decay_tokens_set}'}"
```

Semantics:

- **Stable** points always pass (the LHS disjunct short-circuits).
- **Decay** points pass iff their `aux.tokens` is in the per-combo allowed
  set — i.e. if the `(lr, gbsz, tokens)` triple is actually a row in the CSV.

Important: filter evaluation happens **after** the full per-point config
is rendered (see `dag_resolver.py:_resolve_filter_from_context`). All
interpolations must succeed first, **then** the filter trims the survivors.
That's why Approach A is safer than Approach B — we never materialise a
phantom point with nonsensical values.

Result: 200 points → 20 stable + 81 decay = **101 jobs**. (A symmetry
check: every stable's `aux.tokens == max(allowed_decay_tokens_set)`, so
stable points would pass the RHS too — but the explicit `stage == stable`
short-circuit makes intent obvious and is robust if the set semantics
ever change.)

---

## 7. Checkpoint saving (`save_extra_steps`)

We want each stable to drop checkpoints at **branching** and **end**
points for every token budget it might feed. With 9 budgets that's up to
18 extra saves per stable job.

### The scalar-override trick

`save_extra_steps` is a YAML **list**. Hydra's list-override parser
rejects `oc.eval` expressions inside list elements, so we can't per-stable
append computed iters directly. Instead:

```yaml
save_extra_steps:
  - "${backend.megatron.aux.save_step_br_0}"
  - "${backend.megatron.aux.save_step_br_1}"
  …
  - "${backend.megatron.aux.save_step_end_8}"

aux:
  save_step_br_0: 999_999_999   # sentinel, overridden by stable stage
  save_step_br_1: 999_999_999
  …
  save_step_end_8: 999_999_999
```

The list is fixed-length (18 slots) and just interpolates scalar keys.
**Stable stage** overrides all 18 slots with real `oc.eval` expressions
hard-coded to 6B…300B budgets — always valid regardless of the combo,
because each expression only references `seq_length` and
`global_batch_size` (both available from Group 1). Slots whose iter
exceeds `train_iters` are simply never reached — harmless.

**Decay stages** don't override these slots, so they remain at the
sentinel `999_999_999` and decay jobs emit no `save_extra_steps` saves.

### Why branching AND end points?

- **Branching** (`0.8 × budget_iters`): lets a decay job load this iter
  via `load + ckpt_step = start_iter`.
- **End** (`1.0 × budget_iters`): baseline "what would loss be if we kept
  training stably to this budget", useful as a no-decay reference when
  comparing against the decay endpoint.

For combos with small `max_decay_tokens` (e.g. gbsz=16 → 6B), only
`save_step_br_0` / `save_step_end_0` actually fire during stable training;
the other 16 slots point past `train_iters`.

---

## 8. Decay → stable linkage (pull-based)

Every decay stage, via the shared `defaults` block, sets:

```yaml
backend.megatron.aux.decay_fraction: "${backend.megatron.aux.cooldown_decay_fraction}"
backend.megatron.load:      "${sibling.stable.job.base_output_dir}/checkpoints"
backend.megatron.ckpt_step: "${backend.megatron.aux.start_iter}"
backend.megatron.override_opt_param_scheduler: true

job.start_condition:
  class_name: FileExistsCondition
  path: "${sibling.stable.job.base_output_dir}/checkpoints/iter_<padded start_iter>"

job.cancel_condition:
  class_name: LogPatternCondition
  log_path: "${sibling.stable.job.log_path_current}"
  pattern: "FATAL ERROR|OutOfMemoryError|Traceback"
```

### How `${sibling.*}` works

`dag_resolver.py:extract_sibling_patterns` scans each sweep point's
parameters for escaped `${sibling.<stage>.…}` references and builds a
dependency DAG. For every decay point, it finds the sibling point with
the **same Group 1 position** (i.e. matching `(lr, gbsz)`) whose `stage`
matches the requested pattern — in our case the stage literally named
`stable`. That sibling's fully-resolved config is then exposed under
`sibling.stable.*` when the decay point is rendered, making
`${sibling.stable.job.base_output_dir}` a valid interpolation.

### How gating works at runtime

The monitor loop (`monitor/loop.py:166-197`) iterates over every
unsubmitted job on each tick:

1. Evaluate `start_condition`. If it passes → `sbatch`.
2. Otherwise the job stays pending.

For a decay job the start condition is a `FileExistsCondition` on the
exact `iter_<branching_iter>` directory inside its stable sibling's
checkpoint dir. Until the stable has saved that branching checkpoint
(via `save_extra_steps`), the decay just sits in the store.

`cancel_condition` catches sibling failure: if the stable log contains
`FATAL ERROR|OutOfMemoryError|Traceback`, the decay is marked cancelled
before it's ever submitted.

---

## 9. Job counts & resource estimates

- **Pre-filter sweep expansion**: 20 combos × 10 stages = 200 points.
- **Post-filter**: 20 stable + 81 decay = 101 jobs (verified by running
  `expand_sweep` on the YAML — 10 points per stage, 20 per combo).

Node count is auto-scaled:

```yaml
slurm.sbatch.nodes:
  "${oc.eval:'max(1, int(global_batch_size // (micro_batch_size * gpus_per_node)))'}"
```

With `micro_batch_size = 16` and MN5's 4 GPUs/node: gbsz = 16 → 1 node,
gbsz = 512 → 8 nodes. The `max(1, …)` guard catches small gbsz that
would otherwise round down to 0.

**Warmup filter was dropped intentionally.** With
`lr_warmup_iters = 2000`, the worst-case ratio is the 6B decay at
gbsz=64: `train_iters ≈ 22_889`, ratio ≈ `0.087 ≪ 0.3`. No combo is
ever at risk of spending more than 30 % of its iters warming up.
(The `_tiered.yaml` variant still AND-s the `< 0.3` guard into
`sweep.filter` as a defensive future-proofing check.)

### Job-name format (at a glance)

Jobs surface in `monitor_state/` as
`dense_130M_lr<LR>_gbsz<GBSZ>_<stage><horizon_suffix>.job.json`:

- Decays already carry their budget in `stage`, so the suffix is empty:
  - `dense_130M_lr0.001_gbsz64_decay12BT.job.json`
- Stables now also carry a `<N>BT` suffix matching the combo's
  `max_decay_tokens` (overridden by Group 2 via
  `backend.megatron.aux.job_horizon_suffix`):
  - `dense_130M_lr0.001_gbsz32_stable12BT.job.json`  (gbsz=32 stops at 12B)
  - `dense_130M_lr0.001_gbsz512_stable300BT.job.json` (gbsz=512 goes to 300B)

This makes it obvious at a glance how long any stable will train without
cross-referencing the CSV, and keeps `${sibling.stable.job.base_output_dir}`
pointing at a deterministic path (since `job.base_output_dir` is a pure
function of `job.name`). The suffix is defined only in the
`_tiered.yaml` variant at the moment.

---

## 10. Controlling execution order — tier-based scheduling

The config ships with a `backend.megatron.aux.priority_tier` knob (default
`all`) that splits the 101 jobs into three disjoint submission tiers. Each
of the 20 stables is owned by **exactly one** tier — the earliest tier (in
`center → cross → diagonal` order) whose decays actually consume it — so
`backend.megatron.aux.priority_tier=center` doesn't eagerly queue the
biggest stables when their decays live in `cross`/`diagonal`.

The tier knob, the per-combo decay sets, and `stable_launch_tier` all live
under the same `backend.megatron.aux.*` namespace (alongside `aux.tokens`
and the `save_step_*` fields), so the whole tier-scheduling surface is one
consistent dotted path.

| `backend.megatron.aux.priority_tier` | stables | decays | total |
|---|---|---|---|
| `all`      | 20 | 81 | 101 |
| `center`   |  4 |  9 |  13 |
| `cross`    | 10 | 36 |  46 |
| `diagonal` |  6 | 36 |  42 |

`4 + 10 + 6 = 20` stables and `9 + 36 + 36 = 81` decays, partitioning the
101-job plan.

Each Group-1 combo carries (all under `backend.megatron.aux.*`):

- three per-tier Python-set-literal strings (`center_tokens_set`,
  `cross_tokens_set`, `diagonal_tokens_set`) whose union equals the combo's
  full allowed decay set;
- `stable_launch_tier` — the tier that will launch this combo's stable.

`sweep.filter` keeps a stable iff
`priority_tier ∈ {"all", stable_launch_tier}` and keeps a decay iff its
`tokens` is in the selected tier's set.

```bash
# MUST launch in this order; no tier may be skipped because every tier
# owns at least some stables.
python scripts/run_autoexp.py --config-name=<this-config> backend.megatron.aux.priority_tier=center
python scripts/run_autoexp.py --config-name=<this-config> backend.megatron.aux.priority_tier=cross
python scripts/run_autoexp.py --config-name=<this-config> backend.megatron.aux.priority_tier=diagonal
```

You can overlap the three in wall time (cross can start once center is
underway), but you cannot skip a tier — its stables would never be
submitted and the dependent decays would wait on `FileExistsCondition`
forever. Use `backend.megatron.aux.priority_tier=all` for a single
one-shot submission.

The sibling mechanism works across invocations because `orchestrator.py`
passes `full_points_by_idx=all_points_by_idx` to the DAG resolver (so
`${sibling.stable.*}` resolves even when the sibling stable is filtered
out of the current submission), and `job.base_output_dir` /
`job.log_path_current` are deterministic filesystem paths.

Verification:

```bash
python scripts/validate_lr_gbsz_tokens_grid_tiers.py
```

See **`docs/tier_based_scheduling.md`** for the full design rationale,
alternatives considered, and a recipe for adapting the pattern to other
sweeps. If you additionally want SLURM-queue-level biasing when tiers
overlap, stack `slurm.sbatch.nice=-50` on the center invocation and
`slurm.sbatch.nice=+100` on the diagonal one.

---

## 11. Gotchas / future-proofing

- **Float equality in the filter.** The allowed-set membership test is a
  Python `in` check on floats. YAML `1.e-3` parses as `0.001`; Python's
  `float('1e-3') == float('0.001')` is true and they hash identically,
  so `0.001 in {0.001, …}` works. If you ever inject a non-trivially-
  rounded LR (e.g. `0.0005000001`) you'll get a silent miss — stick to
  short literals.
- **Default `allowed_decay_tokens_set`.** The base config gives it "all
  9 budgets" so a solo run without a sweep still passes the filter.
- **Per-combo `save_interval`.** We inherit `save_interval = 512_000 // gbsz`,
  so small-gbsz combos checkpoint frequently. If checkpoint overhead
  becomes an issue at gbsz ≤ 32, tighten this.
- **Cluster choice.** Defaults are `marenostrum` container + SLURM with
  `qos: acc_ehpc`. Swap to `leonardo`/`lumi` by changing the `defaults:`
  block.
- **Data/tokenizer.** Currently `gpt-neox-20b` legacy tokenizer on MN5
  paths. To switch to the OELLM-262k multilingual tokenizer, add
  `- /experiments/swagatam/multilingual_scaling/data/oellm_256k` to
  `defaults:` and remove the `vocab_file`/`merge_file`/`legacy_tokenizer`
  lines.
- **Non-contiguous allowed sets matter.** E.g. `(5e-4, 128)` has
  `allowed_decay_tokens_set = {12B}` only — not `{6B, 12B}` — so no 6B
  decay job is spawned even though the combo could mechanically support
  it. Don't simplify to "all budgets ≤ max"; the CSV is authoritative.
- **Symmetry check on stable's aux.tokens.** The YAML is self-consistent
  iff `aux.tokens == max(allowed_decay_tokens_set)` for every Group 1
  entry. This was verified once by script; if you edit the grid, re-verify.

---

## 12. Files touched

- `config/experiments/swagatam/multilingual_scaling/dense_130M_lr_gbsz_tokens_grid.yaml`
  — the sweep config itself.
- `docs/lr_gbsz_tokens_grid_sweep.md` — this document.

## 13. Related reading

- `docs/tier_based_scheduling.md` — how
  `backend.megatron.aux.priority_tier` +
  `backend.megatron.aux.stable_launch_tier` split this sweep into
  independently-launchable subsets without breaking sibling refs.
- `docs/multi_stage_design.md` — general pull-based multi-stage sweep design.
- `docs/escaped_interpolation_design.md` — why `\\${…}` is needed inside
  sweep configs.
- `docs/sweep_resolution_ordering.md` — DAG resolution order and sibling
  lookup details.
- `config/sweep/sweep_example.yaml` — minimal reference for the same
  composable-group pattern.
