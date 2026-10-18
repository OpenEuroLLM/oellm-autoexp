# optimization_procedure: first implementation

This is the prototype of `docs/optimal_architecture_search_procedure.md`. It uses only numpy,
scipy and pandas, and runs in about 35 s on the data already in `dump/`.

```bash
OMP_NUM_THREADS=1 python -m optimization_procedure.run_first_test          # -> dump/optimization_procedure/report.md
OMP_NUM_THREADS=1 python -m pytest optimization_procedure/tests -q -o addopts=""
```

> `OMP_NUM_THREADS=1` matters. The matrices are small, so multi-threaded BLAS on a busy node
> makes each SVD about 100× slower. `run_first_test` sets it automatically.

## Modules

| module | layer (doc §) | what it does |
|---|---|---|
| `forms.py` | registry | Parametric forms, each a plug-in: `chinchilla_bowl` (Chinchilla envelope plus an asymmetric log-quadratic (lr, batch) bowl whose optimum follows power laws), `chinchilla`, `fixed_ratio_slice`. Add a form with `@register`. |
| `model.py` | B + C (§6, §7) | `HierarchicalModel`: each form parameter is base + Σ tag deltas + arch delta + study delta, with Gaussian shrinkage. Shrinkage scales are fitted by empirical Bayes (Laplace evidence). Uses Student-t reweighting (IRLS), a separate noise level per study, leave-one-out hold-out with re-admission, and flags architectures the form misfits. Uncertainty comes from a Laplace approximation, with `inflate()` for calibration. |
| `curves.py` | A (§5) | Loss-vs-LR-schedule curve fit (Tissue et al.); detectors for spikes (with lasting offset), bumps (CUSUM), stalls (predict ahead, significance-tested), NaN and divergence. Returns a weight in [0, 1] per run. |
| `decision.py` | C/D (§7.3, §8.3) | Iso-loss GPU-hour multiplier M_a at the target regime on posterior samples, including the loss cost of uncertain HPs. Assigns drop / live / promote. `review` means the arm has a stability or misfit flag, which blocks automatic promotion. |
| `design.py` | D (§8.2) | Candidates around the posterior-mean HP optimum, limited to a trust region (at most one factor-2 step beyond observed HPs). Picks runs greedily by information gain about the decision quantities per GPU-hour, accounting for runs still in flight. |
| `scheduler.py` | E (§9) | Decides launch-now vs wait. It simulates outcomes of pending runs, updates the model, re-plans, and computes P(still wanted). It launches iff expected waste < λ·Δt·κ. |
| `validate.py` | §8.5 / §12 | Hold-out and leave-one-scale-out validation. `compare_forms` weights forms by held-out predictive density. |
| `ledger.py` | learning loop | Append-only record of observations, predictions registered before results exist, and decisions. `calibration()` scores past forecasts and turns the result into an inflation factor. `save_priors`/`load_priors` carry base and tag effects into the next study. |
| `data.py` | ingest | Loaders for `dump/eval_val_metrics.csv` (architecture sweep, cooled evals) and the dense (lr × batch) baseline grid. Holds the architecture tag registry and GPU-hours from measured throughput (`scripts/scaling_laws/costs.py`). |

## How it learns from every experiment

1. **Across architectures.** Tags such as `linear`, `gdn`, `nope` and `ratio7` carry shared
   effects, and each arch gets its own residual delta on top. The data sets how strongly archs are
   pulled together. An arm with no data (e.g. `pda7_nope`, `skda7_nope`) gets a prior prediction
   from its tags, with wide uncertainty. That makes it the first thing the design proposes to
   measure, cheaply and at small scale.
2. **Points that don't fit.** A point whose leave-one-out z exceeds 4 is held out and the model is
   refit. Held-out points stay in the report and are re-admitted once a later fit explains them.
   Runs with QA weight 0 (diverged, stalled) never enter the fit but show up as stability flags.
3. **Its own reliability.** Leave-one-scale-out validation and ledger calibration yield an
   inflation factor applied to every downstream decision. Predictions are registered before the
   results exist, so the planner is graded on its own forecasts.
4. **Priors for the next study.** `FitResult.to_prior()` passes the base and tag posteriors,
   widened, to the next campaign. Do not reload them on the same data, which would count it
   twice.

## What the synthetic tests pin down

Ground truth is known in `tests/test_procedure.py`.

- A 5% outlier is held out.
- A shared tag effect is recovered as a *difference between architectures* to within 0.5% at
  7.5× beyond the largest trained size.
- Absolute extrapolation at that range is weakly identified (errors up to 5%), but it stays
  within the posterior's 3σ. The model knows what it does not know.
- An arm with no data is predicted from its tags and is the first run the design picks.
- Batches are diverse and later picks gain less information than earlier ones.
- The scheduler behaves correctly at the extremes: never launch early when wall-clock is free,
  always launch when it is priceless.
- Curve QA passes a clean curve and catches a spike with a lasting offset, a bump, a stall, NaN
  and divergence.
- Ledger calibration detects a 3× over-confident forecaster.

## First results on repo data (`dump/optimization_procedure/report.md`)

These come from a prototype, so read them as statements about what the data and model can
support, not as final numbers.

- **Divergence found.** `mamba2_nope` 0.9B @ 300BT diverged (ppl 4.9e8, empty loss in both evals).
  Mamba2 is also the only arm the form misfits (LOO RMS z 2.2), and it has the largest error when
  the 0.9B size is held out (+1.2%). Its apparent 1.3× multiplier is therefore status `review`,
  not `promote`.
- **Held-out-size check.** When 0.9B (or 0.4B) is held out, mean error is +0.27% (+0.16%), but
  90% coverage is only 0.80 (0.68). The calibration inflation of about 2.0 is applied to all
  decisions.
- **Architecture multipliers** (vs fullattn at 3B, D/N 100–1000, GPU-hours, calibrated): gdn3_qwen35
  about 1.22–1.24×, gdn3 about 1.19–1.22×, gdn7 about 1.11×, mlstm7 about 1.06–1.09×, swa7 about
  1.01–1.06×. PDA and SKDA have no local data and are predicted from their tags alone (wide
  intervals, 0.45–3×). Nothing is dropped yet: at δ = 5%, even swa7 cannot be excluded.
- **LR optimum is poorly identified.** The loss surface is flat in lr (about 0.5% loss per 4×
  change in lr), so the lr-law exponents need literature-informed priors. These priors are chosen
  because they extrapolate better to a held-out size, even though in-sample evidence prefers wide
  ones. The design responds by proposing LR probes.
- **Proposed next batch.** Cheap screening probes (6BT, 0.1B/0.4B) for the two unmeasured FLA arms,
  plus LR probes on swa7/mlstm7.
- **Scheduler.** With the first 3 proposals pending (about 0.5 h), the next ones are wanted in only
  33–58% of the simulated outcomes. They launch in parallel only if wall-clock is worth ≳ 10–40
  GPU-h per hour.

## Known limitations / next steps

- The architecture sweep has **one (lr, batch) per cell**. Arch-specific LR offsets are identified
  only through shrinkage. The cheapest fix is 3-LR probes at 0.1B/0.2B per arm, which is what the
  design proposes.
- Candidate runs are costed as independent. Shared WSD stable runs and cooldown branches (§8.1)
  are not yet priced as multi-fidelity actions, and the "extend" / "branch" actions are missing.
- Curve QA runs on sparse W&B val curves. It should ingest the dense Megatron train-loss logs and
  feed the per-run weights into the loss-law fit (`weight` column).
- Model-form uncertainty covers a single form. Use `validate.compare_forms` to average over
  `chinchilla_bowl`, Farseer-type forms, etc.
- Not yet wired into the monitor: an `AnomalyCondition` and a `PlannerTick` action (doc §11).
  The design emits DataFrames, not sweep YAML yet.
