# Multilingual scaling sweeps

These sweeps suggest the HP grid per model size and token budget, for both
training and its matching validation run. This document explains how the
sweep files under this directory work; the files themselves should only
carry comments that don't fit this general pattern (for example why one specific
combo's `split` or `checkpoint_reuse_run` differs from the rest).

For the generic sweep grammar (`type: product`/`list`, `defaults`, escaped
`\${...}` interpolations) see `config/sweep/sweep_example.yaml` — this doc
only covers how that grammar is *used* here.

## Layout

- `training/qwen3_dense_<size>.yaml` — the single, complete HP grid for one
  model size: every `(lr, gbsz)` combo ever planned, with what is already
  trained marked so that launching it only submits what is missing (see
  [The merged grid](#the-merged-grid)).
- `validation/qwen3_dense_<size>.yaml` — its validation counterpart; GROUP 1
  mirrors the training file entry-for-entry.
- `experiments/multilingual_scaling/validation/common.yaml` — base val config;
  sets `load` to the matching training run's checkpoint dir by default (see
  GROUP 2 below).

Each file is named `qwen3_dense_<size>.yaml`, where `<size>` is the
**Porian/total** parameter count (`0.21B`, `0.39B`, `0.71B`, `1.26B`) —
*not* the N_NE label the paired `backend/megatron/multilingual_scaling/qwen3_dense_<B>_ne.yaml`
architecture file uses (for example this `0.21B` grid pairs with that dir's `0.1B_ne`
architecture). See
[experiments/multilingual_scaling/README.md](../../experiments/multilingual_scaling/README.md#what-ne-and-porian-mean)
for why the two differ. Consumers select a grid by way of the `defaults:` path, for example
`/sweep/multilingual_scaling/training@sweep: qwen3_dense_0.21B`; the leaf
experiment configs are
`experiments/multilingual_scaling/{training,validation}/<cluster>/qwen3_dense_<size>.yaml`.

## The merged grid

Until 2026-11-03 each size had three grids, `v1` (initial grid from the
English scaling runs, trained on Leonardo/MN5), `v2` and `v3` (re-centred
grids, trained on LUMI). Each version only listed the batch of combos
launched in that round, and they disagreed for many combos (stable horizon,
`split`, `micro_batch_size`, tier). They were merged into the single file per
size with these rules, checked against what exists in
`/scratch/project_465002530/multilingual_scaling/<ne>/training` and the
cross-cluster loss records in `dense_multilingual_models_scaling_results`:

- **One GROUP 1 entry per stable trajectory** `(lr, gbsz, stable horizon)`.
  When a combo has several real stable runs (for example a Leonardo `stable300BT` and
  a LUMI `stable80BT`), each gets its own entry. Sibling lookup goes by GROUP 1
  entry index, so each decay resolves to its own entry's stable. A decay belongs
  to the stable it actually loaded from (its `config-*.yaml`), else to the
  latest version that listed it. Stable horizons that only existed in a config
  and never on disk were dropped.
- **Tier** of each decay budget = the tier in the latest version that listed
  it; `stable_launch_tier` = the first non-empty tier (rule above).
- **`aux.done_tokens_set`** (training only) = decay budgets already trained on
  any cluster (val/train loss recorded, or final checkpoint on LUMI). The
  training filter never submits them. Leonardo/MN5 decays have no logs on
  LUMI, so `finish_condition` alone would resubmit them. The seven 0.1B
  decays trained from hybrid checkpoints (losses removed 2026-11-03) are *not*
  in this set, so they get rerun.
- **Dropped budgets** — planned in v1/v2, never trained, and not re-selected
  by v3's HP selection — are not in any tier set; the per-entry comment lists
  them. Stables with no decay left are kept as a comment block at the end of
  GROUP 1 (not launched, not evaluated).
- **`aux.skip_stable_launch: true`** whenever no pending decay needs the stable
  to train further (all its pending branch checkpoints already exist, or
  nothing is pending).
- **`split`** = what the still-to-run jobs of that entry need. Finished jobs are
  never resubmitted, so their historical split only lives in the comments.
  Seven 0.1B LUMI stables switched from `99,1,0` to `100,0,0` around iter
  22–25k (v2 config change, 2026-07-28); their comments record the switch iter.
- **`micro_batch_size`** = the latest version's value (it only changes nodes /
  gradient accumulation, not the run's identity).

To add work later: append budgets to an entry's tier sets (or add a new entry
for a new stable), and move budgets into `aux.done_tokens_set` once trained.
Never edit `split` or `aux.tokens` of an entry whose stable already has
checkpoints.

## The shape of one sweep: hyperparameters × stages

Every file is `type: product` over two (training) or three (validation)
`groups`:

- **GROUP 0** — one `list` entry that derives `job.name` /
  `wandb_exp_name` from the point's own params.
- **GROUP 1** — a `list` of `(lr, gbsz)` combos: the actual HP grid.
- **GROUP 2** — a `list` of stages (`stable`, `decay<N>BT`, and for
  validation also `branch<N>BT`/`end<N>BT`).

The outer product multiplies combos × stages, then `filter` (bottom of each
file) drops the pairs that don't correspond to a real job — see
[Filtering](#filtering) below.

### GROUP 0 — job naming

```yaml
job.name: "qwen3_dense_<size>_ne_lr${backend.megatron.lr}_gbsz${backend.megatron.global_batch_size}_${stage}${backend.megatron.aux.job_horizon_suffix}"
```

`<size>_ne` here is the **N_NE** label, not the file's own Porian/total name — for example
`qwen3_dense_0.21B.yaml`'s `job.name` starts with `qwen3_dense_0.1B_ne_lr...`.
`job.name` drives `wandb_exp_name` and the run's leaf output directory, and both
of those need to keep matching the historical `0.1B_ne`-labeled runs already
recorded/stored under that identity — only the *file itself* was renamed to the
Porian label.

`job_horizon_suffix` is empty for decay/branch/end stages (they already carry
their budget in `stage`, for example `decay12BT`) and `<N>BT` for `stable` (computed
in GROUP 2, since a stable's horizon is its combo's max decay budget, not a
per-stage constant). So stable jobs surface as `..._stable12BT` /
`..._stable300BT` in `monitor_state/`, and validation job names match their
training counterparts exactly for `stable`/`decay<N>BT` — see
[GROUP 2 (validation)](#group-2-validation-stable-decay-branch-end).

### GROUP 1 — the hyperparameter grid

Each entry is one `(lr, gbsz)` pair from the design CSV, grouped visually by
`# ---- gbsz N ----` comments. Fields:

| field | meaning |
|---|---|
| `aux.tokens` | this combo's max decay token budget (what `stable` trains to) |
| `aux.center_tokens_set` / `cross_tokens_set` / `diagonal_tokens_set` | per-tier Python-set-literal *strings* of decay budgets this combo runs at that grid position; their union is the combo's full allowed decay set. Strings (not YAML lists) so they interpolate verbatim into the `oc.eval` filter, which then parses them as a Python `set` literal |
| `aux.stable_launch_tier` | the tier whose invocation submits this combo's stable — the *first* tier (in `center → cross → diagonal` order) with a non-empty token set for this combo. Keeps `priority_tier=center` from eagerly queuing big stables (for example gbsz=512) whose decays only live in `cross`/`diagonal` |
| `aux.skip_stable_launch` (training only) | `true` when the entry's stable needs no further training (see [The merged grid](#the-merged-grid)) — its decays branch off the existing checkpoints instead of the sweep resubmitting/continuing training on it. Defaults to `false` in the group's `defaults:` block |
| `aux.done_tokens_set` (training only) | Python-set-literal string of decay budgets already trained on any cluster; the filter never submits them. Defaults to `"set()"` |
| `aux.extend_stable_source` (training only) | checkpoints dir of an older, shorter stable to seed this stable from (see below). Defaults to `""` (no-op) |
| `aux.max_tokens` (validation only) | a second, never-overridden copy of `aux.tokens` — GROUP 2's decay/branch/end stages override `aux.tokens` per-stage, so this field lets the filter compare a stage's budget against the combo's own max (see `end` dedup below) |
| `split` | train/val/test split for this combo's stable run and every decay branching off it. Must match whatever split actually produced the existing checkpoint history for this `(lr, gbsz)` pair — Megatron rebuilds the shuffled sample index from `split`, and a mismatch causes a discontinuity at resume. Per-combo overrides are commented inline where they apply |

Decay/branch/end stages in GROUP 2 override `aux.tokens` to their own
budget; `stable` leaves it as the combo's max.

### GROUP 2 (training) — stable + decay

```
- stage: stable          # trains to aux.tokens (combo's max), no decay
- type: list
  defaults: <shared decay config>
  configs:
    - stage: decay6BT    # aux.tokens: 6_000_000_000
    - stage: decay12BT   # aux.tokens: 12_000_000_000
      ...
```

**`stable`** computes (by way of escaped `oc.eval`, see
[Escaped interpolations](#escaped-interpolations-and-siblings)):
`train_iters`, `aux.start_iter`, `aux.job_horizon_suffix`, and 18
`aux.save_step_br_N` / `aux.save_step_end_N` slots — one branching (80% of
budget) and one end (100% of budget) checkpoint iter per token budget
(6B…300B), hardcoded so the same expression is valid for every combo
regardless of its own max. Slots whose iter exceeds this combo's
`train_iters` are simply never reached. These slots feed
`backend.megatron.save_extra_steps` (a fixed 18-element list in the base
config that just interpolates the scalar slots — Hydra's list-override
parser rejects `oc.eval` directly inside list elements, hence the
indirection through named scalars).

**Decay stages** (shared `defaults:`) set `aux.decay_fraction`,
`train_iters`/`lr_wsd_decay_iters`/`aux.start_iter` for their own budget,
and load from the stable sibling's branching checkpoint:

```yaml
backend.megatron.load: "${sibling.stable.job.base_output_dir}/checkpoints"
backend.megatron.ckpt_step: "${backend.megatron.aux.start_iter}"
job.start_condition:    # wait for that branching checkpoint to exist
job.cancel_condition:   # cancel if the stable sibling's log shows a fatal error
```

### Extending a stable run past its original budget (training, rare)

An entry can raise `aux.tokens` past what an older, shorter stable of the
same `(lr, gbsz)` already trained, instead of training a fresh stable from
iteration 0 (currently only 1.26B `lr=5e-4, gbsz=256`, 120BT → 300BT). For those,
`aux.extend_stable_source` (Group 1, per-combo) points at the older, shorter
stable run's checkpoints dir. The Group 2 stable stage's `job.start_condition`
runs `scripts/seed_extend_stable_checkpoint.sh` to symlink every `iter_*` the
source has (not just the resume point — some may still be needed as decay
branch points) into this job's own checkpoints dir before first launch; it
no-ops once that dir is non-empty, so it never rolls back real progress. The
stable stage also sets `override_opt_param_scheduler` conditionally
(`extend_stable_source != ""`), since the seeded checkpoint's LR scheduler
was built for the old, smaller `train_iters`. The bootstrap runs as an actual
script rather than an inline shell command because the sweep's DAG resolver
round-trips every resolved value through Hydra's CLI override grammar, which
doesn't survive shell metacharacters in a YAML string.

### GROUP 2 (validation): stable, decay, branch, end

Validation adds two stage kinds beyond training's `stable`/`decay<N>BT`:

- **`branch<N>BT`** — evaluates the stable run's checkpoint at 80% of budget
  N (the exact checkpoint a `decay<N>BT` training run forked its cooldown
  from). Measures quality right before decay starts.
- **`end<N>BT`** — evaluates the *un-decayed* stable checkpoint at 100% of
  budget N (as opposed to `decay<N>BT`, which is the *decayed* model at
  budget N).

`stable` and `decay<N>BT` don't override `backend.megatron.load`: the base
val config (`common.yaml`) defaults it to
`".../training/${job.name}/checkpoints"`, and GROUP 0's naming formula makes
a val point's `job.name` match its training counterpart exactly — so the
right checkpoint dir is found by job-name matching alone, no sibling
reference needed.

`branch`/`end` points have no training-side directory of their own (training
only ever writes `stable`/`decay<N>BT` dirs), so job-name matching can't find
their checkpoints. Instead they pull `load` from this *same file's* `stable`
sibling point by way of `${sibling.stable.backend.megatron.load}`, and pick the
specific checkpoint with `ckpt_step` (80% or 100% of budget N, computed the
same way as training's `save_step_br_N`/`save_step_end_N`). Sibling
references only resolve within one sweep file's DAG, so there's no way to
reference the training sweep directly — going through the local `stable`
point is the only option.

Every stage still sets `train_iters`/`lr_wsd_decay_iters` even though the val
config runs with `skip_train=True` (no training happens) — this is so the
warmup-ratio filter clause evaluates against realistic values instead of the
base val config's placeholder `train_iters=1`.

All four stage kinds gate submission on `job.start_condition`, which — unlike
training — waits for the *training* run's own final checkpoint
(`iter == train_iters`, always written at end of training regardless of
`save_interval`) rather than a mid-run save, and additionally checks the
job's own log doesn't already contain a completed validation-loss line (so
restarts don't re-evaluate).

## Escaped interpolations and siblings

Every formula that references a sweep-varying value (`aux.tokens`,
`global_batch_size`, sibling paths, …) is written double-backslash-escaped —
`"\\${oc.eval:'...'}"` — because Hydra/OmegaConf would otherwise resolve
`${...}` eagerly while composing the base config, before the sweep has
injected this point's specific values. The escaped form survives as a
literal string through sweep expansion and is only unescaped and resolved
once the full per-point config (including `${sibling.*}` data) is assembled.
Formulas that only reference base-config constants can use plain `${...}`.

`${sibling.stable.*}` resolves to the fully-resolved config of the sibling
sweep point with the same GROUP 1 combo (same `lr`/`gbsz`) but
`stage: stable`. Sibling resolution sees *all* sweep points in every
invocation — including ones a given `priority_tier` filters out of
submission — so `${sibling.stable.*}` still resolves correctly when running
`priority_tier=cross` even though that tier doesn't submit every stable.

## Filtering

Every file's `filter` combines two or three checks, ANDed together:

1. **Tier routing**: a `stable` point survives iff `priority_tier == "all"`
   or `priority_tier == aux.stable_launch_tier`; a non-`stable` point
   survives iff its `aux.tokens` is a member of the tier's own
   `*_tokens_set` (for `tier=all`, the union of all three). This is what
   lets `priority_tier=center|cross|diagonal` launch a disjoint subset of
   the full job list — see [Usage](#usage) below. (Training only) a
   `stable` point additionally requires `not aux.skip_stable_launch`, and a
   non-`stable` point requires its `aux.tokens` not to be in
   `aux.done_tokens_set`.
2. **Warmup-ratio guard** (defensive, currently a no-op):
   `lr_warmup_iters / train_iters < 0.3`. No current combo is close to
   tripping this; it exists so a future short-budget addition doesn't
   silently spend a large fraction of training on warmup.
3. **End dedup** (validation only): drops an `end<N>BT` point whenever N
   equals the combo's own `aux.max_tokens` — that checkpoint is already
   evaluated by the `stable` point itself. Compares against `aux.max_tokens`
   (a plain per-combo field) rather than `sibling.stable`'s resolved values,
   since `stable` can't sibling-reference itself. `branch<N>BT` is never
   affected by this (80% of budget N never equals 100% of any budget).

Because filtering happens **after** the full per-point config is resolved,
every point's interpolations must succeed even for points that get filtered
out — this is why GROUP 1's grid is written as an explicit list of valid
combos rather than a full cartesian product filtered down after the fact:
it avoids ever materializing a "phantom" combo with nonsensical values.

## Formulas

With `seq_length = 4096` and `cooldown_decay_fraction = 0.2` (unless a combo
overrides it), for token budget `B` and this combo's `gbsz`:

```
train_iters        = ceil(aux.tokens / (seq_length * gbsz))   # stable: aux.tokens = combo max; decay/branch/end: per-stage budget
lr_wsd_decay_iters  = int(train_iters * decay_fraction)        # stable: 0; decay: 20%
start_iter          = int(train_iters * (1 - decay_fraction))  # stable exits here; decay loads here
save_step_br_N       = int(budget_N_iters * 0.8)                # stable only — branching checkpoint for budget N
save_step_end_N      = budget_N_iters                            # stable only — end checkpoint for budget N
```

## Job counts

For a file with **C** GROUP-1 combos:

- Training: `C` combos × 10 stages (1 stable + 9 decay) = `10C` points
  pre-filter.
- Validation: `C` combos × 28 stages (1 stable + 9 decay + 9 branch + 9 end)
  = `28C` points pre-filter.

Post-filter, each combo contributes exactly 1 stable + (its number of
allowed decay budgets, that is `|center ∪ cross ∪ diagonal|`) decays; validation
additionally contributes the same count again for `branch`, and for `end`
minus however many of those budgets equal the combo's own max. The
`center`/`cross`/`diagonal` tiers are disjoint and partition both the
stable and decay/branch/end counts, so their per-tier totals sum to the
`priority_tier=all` total.

In training, `aux.skip_stable_launch` and `aux.done_tokens_set` then remove
everything already trained, so a launch only submits the pending work listed
in the per-entry comments ("to train" / "rerun"). Re-derive exact counts with
`python scripts/run_autoexp.py --config-name=<leaf config> --dry-run` rather
than trusting a hardcoded table.

## Usage

Tiers must be launched in this order — every tier owns at least one stable,
so skipping one leaves its decays waiting on `FileExistsCondition` forever:

```bash
python scripts/run_autoexp.py --config-name=<this-config> backend.megatron.aux.priority_tier=center
python scripts/run_autoexp.py --config-name=<this-config> backend.megatron.aux.priority_tier=cross
python scripts/run_autoexp.py --config-name=<this-config> backend.megatron.aux.priority_tier=diagonal
```

They can overlap in wall time (for example start `cross` once `center` is
underway) — each invocation runs its own orchestrator/monitor process, and
coordination between them is purely file system-based (checkpoint dirs + log
symlinks under the same `base_output_dir`). Use
`backend.megatron.aux.priority_tier=all` for a single one-shot submission of
every job instead.

## Resource scaling

`slurm.sbatch.nodes` auto-scales as
`max(1, global_batch_size // (micro_batch_size * gpus_per_node))` — for example on
Marenostrum (4 GPUs/node) with `micro_batch_size=4`, `gbsz=16` uses 1 node
and `gbsz=512` uses 8. The `max(1, ...)` guard catches small `gbsz` that
would otherwise round to 0.

## Gotchas

- **Float equality in the filter.** Tier-set membership is a Python `in`
  check on floats; short literals like `1.e-3` parse and hash consistently,
  but a non-trivially-rounded LR (for example `0.0005000001`) could silently miss
  its set. Stick to short literals.
- **Non-contiguous allowed sets are intentional.** A combo's allowed decay
  set is exactly what the design CSV says, not "all budgets ≤ max" — don't
  "simplify" a combo's `*_tokens_set` to a contiguous range.
- **`split` must match training history**, not just look reasonable — see
  the GROUP 1 table above. Changing it for an existing combo breaks resume.
- **`skip_stable_launch` / `done_tokens_set` only affect training.** The
  validation sweep always evaluates every listed checkpoint regardless of
  which cluster or invocation trained it — job-name resolution finds the
  checkpoint either way, and `finish_condition` skips already-evaluated
  points. Points whose checkpoint is not on this cluster (for example Leonardo-only
  decays) simply wait on their `FileExistsCondition` and are never submitted.
- **`validation/` GROUP 1 must stay in sync with `training/`**: same entries
  in the same order, same `aux.tokens` and `*_tokens_set` values, since those
  gate which decay/branch/end checkpoints get evaluated and `aux.tokens`
  must match the training stable's horizon for job-name checkpoint resolution.
- **A job name can refer to two independent models.** Leonardo/MN5 and LUMI
  runs with the same name are different models (different init/data
  pipeline). Entries whose pending decays branch off a LUMI-trained stable
  while some done decays came from Leonardo/MN5 carry a `WARNING` comment.
