# Scaling-law pipeline (multilingual architecture sweep)

Extracts the held-out **validation losses** produced by the eval sweep
(`eval_all_scaling_all.yaml`) and answers, in this order:

1. **Is each hybrid better than full attention everywhere?** — matched-pair loss ratio
2. **Does the advantage survive overtraining?** — does the ratio move with ln N / ln D.
   A **constant ratio** is the scale-free claim: a fixed percentage gain at any
   `(N, D)` transfers to arbitrary scale without knowing where the loss curve ends up.
3. **Which architecture is compute-optimal?** — iso-loss GPU-hour multiplier, then
   the per-arch compute-optimal frontier `L(C)`
4. **Kaplan / Chinchilla scaling law** — per-arch `L(N,D) = E + A·N^-α + B·D^-β`
   and the compute-optimal frontier derived from it (`chinchilla.py`)

## Workflow

```bash
# 1. (re)pull metrics from the eval logs on the cluster  ->  dump/eval_val_metrics.csv
scripts/korbi/scaling_laws/refresh_metrics.sh                 # ssh host defaults to "jupiter"

# 2. select lr/bs, compare, fit + plot
python scripts/korbi/scaling_laws/run.py --input dump/eval_val_metrics.csv --out dump/scaling_laws

# or the one-shot wrapper (pull + everything):
scripts/korbi/make_scaling_curves.sh
```

`run.py` also accepts an eval **output directory** directly (`--input .../_main_7to1_eval`)
if you have the logs locally.

## The lr/bs selection step (`hpselect.py`)

A scaling law is a statement about the **envelope** `L*(N, D) = min over lr, gbs`.
Fitting raw sweep points instead regresses through a mix of well- and badly-tuned
runs, and the fitted exponent silently absorbs the tuning error. So selection runs
**before** anything is fitted:

- `--hp-select per-arch` (default) — argmin loss per `(arch, size, budget)`; each
  architecture may pick its own optimum. This is the envelope the fit wants.
- `--hp-select shared` — one `(gbs, lr)` per `(size, budget)` used by **all**
  architectures. Costs envelope tightness, buys exactly matched pairs. The
  head-to-head comparison always uses this, whatever `--hp-select` says.
- `--hp-select none` — passthrough.

> **State of the current sweep:** `main_scaling_all.yaml` runs exactly ONE
> `(gbs, lr)` per `(size, budget)` — the tiers are pre-picked "near-winner" combos
> from an earlier baseline sweep — so the selection is a **no-op today** and the
> run prints so. The fit is therefore only as good as those pre-picked tiers; it
> is not yet a true envelope. Add lr/bs points to the sweep and it becomes one
> with no code change.

`hpselect.report_tier_switches` warns about the related trap that **is** live: within
a size the sweep changes tier as the budget grows (0.1B gbs128→256 at 80BT, 0.2B
gbs128/lr1e-3 → gbs512/lr2e-3 at 50BT). Never fit `L(D)` across a switch — it
splices two hyperparameter regimes and mis-reads the slope, which then propagates
straight into the compute multiplier. `compare.py` fits per tier for this reason.

## The head-to-head (`compare.py`)

- `hybrid_vs_fullattn.csv` — per matched `(size, budget, gbs, lr)` cell, both the
  loss **ratio** (`ratio_pct`, the scale-free reading) and the absolute
  **difference** (`delta_mn`). No model is fitted; the architectures share data
  order, batch and schedule, so every nuisance cancels.
- `hybrid_vs_fullattn_trend.csv` — both readings regressed on centred ln N and
  ln D with standard errors. **`dRatio_dlnD > 0` means the advantage erodes as
  you overtrain** — the one outcome that would kill extrapolation.
  `better_invariant` names whichever reading has the flatter ln D trend; the two
  are not compatible (over this sweep L falls ~23%, so a fixed 25 mn gap is a
  ratio drifting ~0.2 pp, and a fixed ratio is a gap shrinking 25 → 20 mn).
- `iso_loss_multiplier.csv` — the factor by which fullattn compute must grow to
  reach the hybrid's loss, in tokens and in GPU-hours. **`gpuh_mult > 1` is the
  decision metric.** `extrapolated` flags rows whose required baseline `D` falls
  outside the tier's fitted range.
- `ratio_vs_fullattn.png` — the ratio against `D/N` (the overtraining view).
- `ratio_vs_gpu_hours.png`, `ratio_vs_flops.png` — the **same y-axis** against
  total training cost (the scaling view). This puts all four size ladders on one
  common axis, so a flat band reads directly as "the same relative gain at every
  compute budget" — the statement you extrapolate with. The two x-axes are not
  equivalent: `flops` (6ND) is *identical* for both architectures at a matched
  cell, so the x position is unambiguous; `gpu_hours` is the **hybrid's own**
  cost, since throughput differs, and full attention would sit slightly to the
  left of every hybrid except SWA (`base_gpu_hours` in the CSV).
- `delta_vs_fullattn.png`, `compute_multiplier.png`.

Conversion to a multiplier uses the local slope `dL/dln(D)` of the fullattn curve
**inside the same tier**. That slope flattens as `D/N` grows, which is why a
constant Δ buys a *larger* compute saving the deeper into overtraining you go.

## Kaplan / Chinchilla scaling law (`chinchilla.py`)

Fits `L(N, D) = E + A·N^-α + B·D^-β` **per architecture** by Hoffmann et al.
Approach 3 — Huber loss (δ=1e-3) on log-space residuals, LogSumExp
parametrisation, multistart over their initialisation grid — then derives the
compute-optimal frontier under two cost accountings:

| accounting | cost | `N_opt ∝` |
|---|---|---|
| FLOPs | `C = 6ND` | `C^(β/(α+β))` |
| GPU-hours | `C_h ∝ D·N^σ`, σ≈0.68 | `C_h^(β/(α+σβ))` |

σ comes from `costs.throughput_powerlaw`: measured throughput goes as `N^-0.68`,
not `N^-1`, because larger models use the GPU better. So an extra parameter costs
less real machine time than 6ND implies, and **the GPU-hour-optimal model is
larger than the FLOP-optimal one** (`N_exp_gpuh` ≈ 0.65–0.78 vs `N_exp_flops` ≈
0.54–0.62 here — the FLOP figure sits between Chinchilla's 0.50 and Kaplan's 0.73,
the GPU-hour figure lands near Kaplan).

`chinchilla_fits.csv` carries the parameters, both frontier exponents, the
tokens-per-parameter ratio at a stated budget, and **size-stratified bootstrap
90% CIs**. The CIs are not decoration: this sweep spans 12× in N against
Hoffmann et al.'s ~1000×, so α and β are weakly identified and the intervals
overlap across architectures. Treat the exponents as indicative and use the
matched-pair ratio for architecture claims. Stratifying by size matters because
with only four distinct N an unstratified resample regularly drops a whole size
and degenerates the fit.

**IsoFLOP profiles (Approach 2) are not run.** They need several model sizes at a
shared FLOP budget; the best cell here has 3, and a 3-point parabola has no
residual degrees of freedom, so its minimum would look like a measurement without
being one. `isoflop_feasibility` reports coverage each run and it switches on once
≥4 sizes share a budget.

## Outputs (`dump/scaling_laws/`)
- `scaling_data.csv` — tidy points: `arch,size,gbs,lr,type,budget_bt,iter,D,N,d_over_n,tier,flops,gpu_hours,val_loss,val_ppl`
- `fits.csv` — per-arch `L(C)=E+A·C^(-alpha)` fitted on the **compute-optimal
  frontier**, over the cost range every arch covers. `low_conf` marks fits with
  too few frontier points or a runaway exponent — do not quote their `alpha`.
- `hypotheses.csv` — how the arch advantage enters `L(N,D)`: constant loss offset
  vs effective-size vs effective-data factor (they agree in-range and diverge
  outside it, so this is secondary evidence to the matched-pair trend).
- `chinchilla_fits.csv` — **per-arch** `L(N,D)` + compute-optimal exponents + bootstrap CIs.
  `L_opt_ref` is evaluated at a **shared** budget (the top of the range all archs
  cover); evaluating each at its own max FLOPs would rank them by who owns the
  biggest run. `A` and `B` are NOT comparable across archs — their units are
  `loss·params^α` and `loss·tokens^β`, so they only line up at equal α, β.
- `chinchilla_frontier.png` — `N_opt(C)` and the compute-optimal `L(C)` per arch
- `loglog_gpu_hours.png`, `loglog_flops.png` — Kaplan-style log-log. Left panel is
  log cost vs log final loss; it is nearly flat because loss is floored at `E≈1.6`
  and spans only ~0.09 decades. The right panel subtracts the fitted `E` to give
  the **reducible** loss, which spans ~0.26 decades and is straight — its frontier
  slope is the compute exponent. Those slopes are conditional on `E`: a larger `E`
  mechanically steepens the slope, and `E` is weakly identified, so rank
  architectures with `pairwise_ranking.csv`, never with the slopes.
- `param_adjusted_losses.csv`, `hybrid_vs_fullattn_nadj{,_trend}.csv`,
  `pairwise_ranking_nadj.csv`, `ranking_by_tier_nadj.csv` — the head-to-head with
  each arch's **extra parameters priced out** (`--n-mode measured` only). Every
  loss is moved to fullattn's N of its size along fullattn's own fitted N-term,
  `L_adj = L + A·(N_fa^-α − N^-α)`; `n_credit_mn` is the amount removed, with a
  90% band from a size-stratified bootstrap of that fit (`fit_lo/hi_pct` in the
  pairwise table). Null assumption: a mixer parameter is worth what a dense
  parameter is to fullattn. Cross-check: the H1 offsets in `hypotheses.csv`
  (a joint fit, no per-cell adjustment) land within ~1 mn of these means.
- `param_adjusted_vs_fullattn.png` — the architecture comparison in one figure.
  Top: per-cell loss ratio vs full attention against training FLOPs, **raw**
  (left) and **param-adjusted** (right, error bars = the fit's 90% bootstrap
  band); lines join one `(size, tier)` ladder only. Bottom: mean adjusted ratio
  ± SE per `(arch, size)` with the raw mean hollow, so the size of each
  parameter correction and the per-size winner read at a glance.
- `pairwise_ranking.csv` — every arch vs every other, paired on the cells both ran.
  This is the ranking instrument: the pairing is exact, so the paired SE resolves
  margins the parametric fit cannot.
- `ranking_by_tier.csv` — the same ranking split by `(gbs, lr)` tier. A margin that
  swings between tiers is hyperparameter sensitivity, not architecture; averaging
  over it produces a ranking that is an artifact of which tiers are populated.
- `chinchilla.json` — pooled (all-arch) `L(N,D)=E+A/N^a+B/D^b`, kept as a sanity check
- `scaling_flops.png`, `scaling_gpu_hours.png` — log-x scatter, **colour+dash =
  architecture, marker = model size**; faded points are dominated (off-frontier).

## Options
- `--type cooldown|stable|both` — cooldown = fully-decayed finals (default, the endpoints);
  stable = pre-decay branch checkpoints.
- `--flops-mode 6nd|full` — `6nd` = Chinchilla `6ND` (attention-agnostic, so archs are
  compared at matched model-compute); `full` adds self-attention FLOPs (distinguishes
  full vs windowed vs linear attention). The **GPU-hour** axis always reflects the
  measured per-arch throughput, so it captures the real efficiency gap regardless.
- `--keep-incomplete` — keep evals whose checkpoint iteration disagrees with their
  budget. By default these are dropped and listed: an eval that picked up the
  `firstcd` seed (`start_iter+1`, one step into the decay) instead of the completed
  `contcd` reports a loss that is not the cooldown endpoint at all.

## Cost model (`costs.py`)
- `N` — **measured** non-embedding params per `(arch, size)`: Megatron's
  `number of parameters` log line (`N_TOTAL_MEASURED`) minus the tied embedding.
  The architectures are **not** param-matched: at the same size label the
  hybrids carry +7–12% N_ne over fullattn (0.1B≈71M, 0.2B≈186M, 0.4B≈440M,
  0.9B≈859M), Mamba2 +22%/+23% at 0.2B/0.9B (expand-2 `d_inner = 2·hidden` vs
  GDN's `heads·128`), mLSTM +24%/+18% at 0.4B/0.9B (no GQA mode). An arch
  missing from the table raises. `--n-mode nominal` restores the old shared
  dense-geometry N (it reproduces the pre-2026-09-29 fits exactly) — it credits
  the heavier hybrids with too few params, which read up to ~35 mn of pure size
  gain as architecture gain and inflated Mamba2's alpha to 0.36.
- `D` — tokens seen = `iter × gbs × 4096`.
- `GPU-hours` = `D / (TOK_S[arch][size] × 3600)` using the measured tok/s/GPU table.

> **Throughput caveat.** At `SEQ_LEN` 4096 every linear/gated variant is *slower*
> per token than full attention (GDN 0.86–0.94×, Mamba2 0.82–0.93×, mLSTM
> 0.95–1.0×); only SWA is faster (1.06–1.14×). The quadratic term is still cheap
> at this context length, so the hybrids pay their extra state/gating cost without
> collecting the subquadratic win. Any GPU-hour advantage measured here is
> therefore a **floor** on what longer contexts would show.

## Plot conventions
Architecture is encoded three ways — colour, dash pattern, and legend — never by
colour alone: the categorical palette's worst colour-vision-deficiency pair sits in
the ΔE 6–8 band, which is only safe with that secondary encoding. Palette validated
all-pairs on a light surface; re-run the check if you change `ARCH_COLORS`.

## Notes
- Fit **cooldown** checkpoints for the final scaling law; `stable` points add pre-decay
  samples if you want the intermediate curve.
- The CSV is a snapshot — re-run `refresh_metrics.sh` as more evals finish. As of
  2026-08-05 (205 rows / 98 cooldowns / 76 matched pairs) every architecture reaches
  `D/N ≈ 2805` (0.1B @ 200BT), and Mamba2 now has a full 0.2B ladder. Still missing:
  Mamba2 0.9B at 30/50/120BT and 0.2B/6BT, GDN+mLSTM 0.9B/80BT, SWA 0.2B/120+200BT,
  the 0.9B/200BT row for everyone except SWA, and all 300BT cooldowns.
- **Training finishing is not the same as data arriving.** Cooldowns and evals are
  separate sweeps and nothing queues the evals automatically, so `contcd_decay*BT`
  runs can sit finished-but-unevaluated for days while the CSV looks stalled. To
  find them, diff the training dirs whose log contains `after training is done`
  against the `evalcd_*` dirs, then rerun just those points:
  `run_autoexp.py --config-name .../eval_all_scaling_all --array-subset '6-7,22-23,…'`.
  Indices come from the sweep expansion, where **architecture is the outer axis**
  (fullattn 0–49, SWA 50–99, GDN 100–149, mLSTM 150–199, Mamba2 200–249, 50 ladder
  points each) — derive them with `expand_sweep`, never by hand.
- `refresh_metrics.sh` stages the pull into a temp file and only replaces the CSV
  once it has ≥1 data row, keeping a `.bak` if the pull *shrank*. Redirecting ssh
  straight into the CSV truncates it before ssh runs, so a dropped connection or an
  expired ssh-agent used to destroy the last good snapshot. Evals never disappear —
  a row count that goes down means a partial pull, not progress.
