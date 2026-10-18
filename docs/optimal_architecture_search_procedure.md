# Optimal Architecture Search Procedure: Loss-Curve Models, HP Scaling Laws, Sequential Design

**Status:** proposal / design document (2026-09-23)
**Scope:** replace grid sweeps in `config/experiments/architecture_scaling_variants/**` (and future
optimizer / MoE / Engram / recurrent-depth studies) with a model-based, compute- *and* wall-clock-aware
procedure that

- **(a)** predicts the loss curve of every run and flags runs that do not behave as expected
  (spikes, bumps, stalls), so only *solid* runs enter the fits,
- **(b)** fits hyperparameter optima as parametric (power-law) functions of scale, with calibrated
  uncertainty, from solid runs,
- **(c)** uses these posteriors in a Bayesian sequential design loop that proposes the next runs,
  exploiting the known power-law structure, until the fits support large extrapolation runs,
- **(d)** includes *architecture* decisions (sequence mixer / hybrid ratio, optimizer, backbone, MoE,
  Engram, recurrence in depth) and drops variants early when they are unlikely to be competitive
  in the target regime **D/N ≈ 100–1000**,
- **(e)** models the trade-off between **sequential** (more informed, slower) and **parallel**
  (faster, partly uninformed) execution.

Contents:

1. [Where we are](#1-where-we-are-state-of-this-repo)
2. [What Marin does](#2-what-marin-does-public-procedure)
3. [Other literature](#3-literature-map)
4. [Overview of the procedure](#4-overview-a-layered-probabilistic-model--a-decision-layer)
5. [Layer A: loss-curve model and run QA](#5-layer-a--loss-curve-model-and-run-quality-assessment)
6. [Layer B: HP response surface and HP scaling laws](#6-layer-b--hp-response-surface-and-hp-scaling-laws)
7. [Layer C: envelope loss law and architecture comparison](#7-layer-c--envelope-loss-law-cost-model-architecture-comparison)
8. [Layer D: sequential design / Bayesian optimization](#8-layer-d--sequential-design-bayesian-optimization)
9. [Sequential vs parallel execution](#9-sequential-vs-parallel-execution-the-wall-clock--compute-trade-off)
10. [Architecture axes](#10-architecture-axes-how-each-enters-the-model)
11. [Implementation plan in this repo](#11-implementation-plan-in-this-repo)
12. [Pitfalls](#12-pitfalls-and-guard-rails)
13. [References](#13-references)

---

## 1. Where we are (state of this repo)

The current study (`multilingual/main_scaling_all.yaml` plus the arm files `gdn3_`, `qwen35_`,
`fla_pda_`, `fla_skda_scaling_all.yaml`) is a **fixed grid**:

- It covers 5+4 architecture variants (fullattn, swa7, gdn7, mlstm7, mamba2, gdn3, gdn3_qwen35,
  pda7, skda7).
- Each variant runs on 7 (size, tier) ladders from 0.1B to 0.9B, with 2–5 token budgets per ladder.
  D/N reaches about 4000 at 0.1B / 300BT.
- **One (gbs, lr) per (size, budget)**, pre-picked from the dense baseline sweep
  (`dump/scaling_experiments_multilingual/best_bs_lr_multilingual.csv`).
- A **WSD trunk per ladder**, with cooldown branches (`firstcd_`/`contcd_decay<B>BT`) at 0.8·D for
  every budget. One stable run therefore supplies several D points, which is the approach of
  Hägele et al. (2024).

The analysis lives in `scripts/scaling_laws/`:

- matched-pair loss ratio vs fullattn, and its trend in ln N / ln D (`compare.py`)
- iso-loss GPU-hour multiplier (`compare.py`)
- per-architecture Chinchilla Approach-3 fit with size-stratified bootstrap (`chinchilla.py`)
- a GPU-hour cost model, tok/s ∝ N^-0.68 (`costs.py`)

That README already names the three structural weaknesses this document addresses:

1. **The fit is not an envelope.** With one HP point per cell, `hpselect.py` is a no-op, and every
   architecture inherits the *fullattn* optimum. Any architecture whose LR optimum is shifted
   (typical for linear-attention and NoPE variants) is penalized by an unknown amount. As Wang et
   al. 2025 found for data curation, rankings at a fixed HP can flip.
2. **The exponents are weakly identified** (12× range in N), and the CIs across architectures
   overlap.
3. **Every variant runs "to the end"** on every ladder, whether or not the early evidence already
   rules it out.

Other gaps:

- There is **no curve-level QA**. Spikes and stalls are not detected, and training curves are
  unused. Only the final val loss of each cooldown enters the fit.
- The parameter-count confound is known: an earlier comparison hid a 30% parameter difference.
  Every comparison must therefore carry N (total, non-embedding, active) and measured throughput.

What we already have, and should reuse:

- the WSD trunk/branch mechanics
- sibling-resolved sweeps with `list`/`product` groups
- the monitor with conditions and actions (`oellm_autoexp/monitor/`)
- `aux.true_time` / `tok_s` throughput tables for cost prediction
- the offline eval sweep
- Megatron logs of train loss every 10 iterations and validation loss every `204800//gbs`
  iterations

---

## 2. What Marin does (public procedure)

The details below come from reading the Marin repository (HEAD 2026-09-22), its docs and
retrospectives, and the "Fantastic Pretraining Optimizers" paper. **Marin does *not* do
Bayesian optimization across scale, multi-fidelity HPO or learning-curve extrapolation.** Its
procedure is disciplined and mostly hand-engineered:

### 2.1 Scaling heuristic + IsoFLOP → scaling ladder

The code is in `lib/marin/src/marin/scaling_laws/isoflop_analysis.py`, and the recipe is
`docs/recipes/add_scaling_heuristic.md`. The recipe separates two things:

> "A scaling law tells you *what* to train for a compute budget. A scaling heuristic tells you
> *how* to train each candidate."

1. **Base point via BO at one small scale.** Google Vizier, 10 loops × 4 suggestions, tunes a
   ~130M model on 1B tokens (`experiments/references/reference_hyperparameter_sweep.py`). The
   search covers lr, adam_lr, β1, β2, ε, clip and z-loss, with the objective
   `uncheatable_eval/macro_loss`.
2. **A hand-set transfer heuristic** (`CompletedAdamHHeuristic`) moves the base point across scale:
   - $\eta = 0.0063\sqrt{B/B_0}\,(T_0/T)^{0.3}$, capped at 0.01
   - β2 adjusted so that its half-life is constant in tokens
   - depth $L = \mathrm{round}(d/(64 + 4\log_2 d - 9))$
   - batch $B = T/(4096\cdot 2^{16})$, rounded up to a power of 2 and halved until η < 0.01
   - no weight decay: the AdamH/MuonH "hyperball" optimizers keep a fixed weight norm

   The 0.3 horizon exponent comes from a small Vizier grid and is described as having a "weak
   per-point basis". The MoE heuristic fits
   $\eta_{adam} = 0.066\,T^{-0.395} d^{-0.150} \sqrt{\text{tokens/batch}}$ on 17 cells
   (R² = 0.996).
3. **IsoFLOP sweep (Approach 2).** Reference suite: 78 runs over 7 budgets from 3e18 to 3e20 FLOPs,
   5–15 widths each.
   - FLOPs are bucketed to powers of 1.1.
   - A Huber-robust quadratic in $\log D$ is fit per budget, giving $D^*$.
   - A log-log line gives $D^*(C) = A C^{\alpha}$.
4. **Held-out validation.** The ladder was checked at 1e21, 1e22 and 1e23 FLOPs, with 3 seeds at
   the lower two. The prediction error was **+0.5%, +0.2% and +0.2%**, against a seed spread of
   about 0.1% and a bootstrap CI of ±0.5% growing to ±4% at 1e23 ("Scaling laws that extrapolate
   300× past the fit", Open Athena / Delphi blog, May 2026).
5. **The first attempt failed.** A µP-style recipe that grew the batch at fixed step count, with
   **no token-horizon LR correction**, missed by 2.5% at 1e22 and **diverged at 1e23**, although
   it "looked clean on small runs". The lesson: small-scale fit quality says little about
   extrapolation. Hold out a scale and test the prediction.

### 2.2 Promotion gates for ideas (MoE workflow, `experiments/grug/moe/agent.md`)

Each variant is compared against a baseline law $L(C) = 1.6 + 88.3\,C^{-0.0941}$, with $L_\infty$
and α **pinned** from earlier fits and only A refit.

- **Gate 1.** Run at d512 and d768 at their compute-optimal points. The "effective speedup" must
  be greater than 1 at both, including the throughput ratio. The speedup is the compute the
  baseline needs to reach the variant's loss, divided by the variant's compute, then multiplied
  by the throughput ratio.
- **Gate 2.** Add d1024 and d1280. The speedup must hold at all 4 points, and a fit of
  $1.6 + A C^{-\alpha}$ must beat the baseline's projection at **1e21 and 1e23**.
- **Extra criteria.** Low curvature of the IsoFLOP bowl (robust to over- and under-training), and
  stability signals such as norm growth and activation outliers.

The workflow is designed to be run by coding agents. Of 80 logged experiments, 16 "worked".
Marin's speedrun leaderboard likewise ranks entries by a per-family fit $L = aC^{-b} + c$
**projected to 1e22 FLOPs**.

### 2.3 Fantastic Pretraining Optimizers (Wen, Hall, Ma, Liang 2025)

This is the optimizer study that came out of Marin. It is the most relevant template for
"compare variants fairly across scales".

- **Phase I: coordinate descent.** Each HP gets a discrete grid, and a change is accepted if it
  improves loss by more than 3e-3. This runs in 6 regimes: 130M/300M/520M at 1× Chinchilla, and
  130M at 2×/4×/8×.
- **Phase II: find the scaling-sensitive HPs.** An HP is scaling-sensitive if no single value
  stays within 6.4e-3 of the best loss in every regime. Only those are re-swept in 6 more regimes.
  The results: AdamW {LR, WD, warmup, batch}, Muon {LR}, Soap {LR, warmup, block size}.
- **Phase III: HP law** $h^*(N,D) = \alpha N^{-A} D^{-B} + \beta$, fit to 12 optima per optimizer
  and checked at 1.2B.
- **Speedup.** They fit the baseline's $L_N(D) = \alpha_N D^{-B_N} + \beta_N$ and invert it:
  speedup = $D_{\text{AdamW}}(L) / D_{\text{opt}}$.
- **Findings.**
  - Speedups shrink with scale: about 1.4× at 0.1B (1.3–1.4× below 520M), about 1.1× at 1.2B.
  - Rankings change with D/N: Muon wins at 1–4×, Soap/Kron at 8× and above.
  - **Rankings flip during LR decay**, so compare only cooled-down endpoints, or model the
    cooldown explicitly.

### 2.4 Big runs

- **Marin 8B** did not take its HPs from scaling laws (the "Tootsie" approach: adapt mid-run).
- **Marin 32B** suffered repeated loss spikes (Llama architecture without QK-norm). Clipping,
  update-norm skipping, optimizer-state rebuilds and Muon all failed to stop them. Warm-starting
  into a QK-norm architecture fixed it. Their own verdict: they should have de-risked QK-norm at
  small scale first. **Stability is an architecture property that must be tested at small scale
  with high LR** (Wortsman et al. 2023).
- The current MoE hero ladder (23B active / 535B total, 18T tokens) is **overtrained at a
  constant 791 tokens per active parameter** on every rung. This is the D/N regime we target, and
  a precedent for fitting ladders at fixed D/N instead of compute-optimal ones.

### 2.5 What we take from Marin

| Take | Improve on |
|---|---|
| Separate "scaling heuristic" (how to train at N,D) from "scaling law" (what to train) | Heuristic exponents from a weak grid → we fit them with uncertainty (Layer B) |
| Gates at 2, then 4 scales, projecting to held-out target budgets | Pass/fail on point estimates → posterior probabilities with explicit risk (Layer D) |
| Speedup = inverse-law compute multiplier × throughput ratio | Pinned exponents → hierarchical shrinkage instead of hard pinning |
| Held-out scale validation before large runs | Make it a mandatory step of the loop |
| Test stability at small scale (32B lesson) | Automated anomaly detection per run (Layer A) |
| Compare cooled endpoints only (optimizer paper) | Loss-curve model that *predicts* the cooldown from the trunk |

---

## 3. Literature map

The references are in §13. Unverified items are marked there.

| Need | Key works | Take-away for us |
|---|---|---|
| **Loss-curve prediction under arbitrary LR schedule** | Tissue et al. 2024 (LR-annealing law), Luo et al. 2025 (multi-power law, MPL), Hägele et al. 2024 (WSD) | $L(t)$ is a function of LR area $S_1$ plus an annealing bonus; 1–2 fitted curves predict any schedule, including our cooldown branches |
| **Cross-size curve structure** | Qiu et al. 2025 (supercollapse), Bergsma et al. 2025 (Scaling with Collapse) | Normalized curves of well-scaled runs collapse onto one universal curve. Deviations flag mis-scaled HPs or anomalies, which gives an early-stop criterion |
| **Learning-curve extrapolation / freeze-thaw** | Swersky 2014, LC-PFN (Adriaensen 2023), ifBO (Rakotoarison 2024), DyHPO | Bayesian posterior over curve continuation, and an acquisition over which run to "thaw" |
| **Instability** | Wortsman et al. 2023, Takase et al. (Spike No More), Molybog 2023, ZClip, OLMo 2, Huang et al. 2026 (mechanism monitors) | z-score detectors on grad-norm and loss residuals. Attention-logit and output-logit growth are predictable from small scale. LR sensitivity is a stability metric |
| **HP optimum laws** | DeepSeek LLM 2024, Porian 2024, Step Law 2025, Bjorck 2024, Power Lines 2025, Wang & Aitchison 2024, Wen et al. 2025, Yang et al. 2026 | $\eta^*\propto N^{-a}D^{-b}$ with $b\approx 0.3$ *at fixed batch* (Bjorck; Marin). Step Law reports $\eta^*=1.79N^{-0.713}D^{+0.307}$ because its batch co-scales as $B^*\propto D^{0.571}$; the sign of the D-exponent depends on whether B is held fixed, so the law must include B explicitly. $B^*$ depends mainly on D. Weight decay via the AdamW timescale $\tau = B/(\eta\lambda D)$. Log-linear LR laws curve at scale, so fit the *effective* LR and extrapolate along D |
| **HP transfer** | µP, CompleteP (Dey 2025), Complete^(d)P (Mlodozeniec 2025), u-µP, Everett 2024 | These reduce, but do not remove, what must be fit. Horizon (D) transfer remains the weak point, as Delphi v1 showed |
| **Scaling-law estimation** | Besiroglu 2024, Choshen 2024, Czech et al. 2026 (Approach-2 bias), Gemstones, Pearce & Song 2024 | Prefer parametric Approach 3 with care (summed Huber loss, variable projection). Use intermediate checkpoints. Bootstrap over *design choices*, not only noise |
| **Downstream prediction** | OLMo ladder (Bhagia 2024), DataDecide, Lourie 2025, Krajewski 2025 | Decide on loss / bpb. Downstream is predictable in only about 40% of cases. Single small-scale ranking is about 80% decision-accurate, a strong baseline to beat |
| **Sequential design of scaling laws** | **PLES** (Chen et al. 2026, power-law entropy search), **SL²** (Li et al. 2026), Schram et al. 2026 (surrogate-guided pruning), CARBS (Fetterman 2023), taKG (Wu 2019), Bayesian scaling-law PFN (Lee 2025) | Acquisition = information about the *law* (or the decision), per unit cost. Reported savings are about 10× over grids |
| **Batch / async BO** | GP-BUCB (Desautels 2014), q-EI/q-NEI/BoTorch, Kriging believer, local penalization, parallel TS (Kandasamy 2018), ASHA, CArBO, Pandora's box Gittins index (Xie 2024) | The cost of batching is only a constant factor when batches are diverse. Asynchronous beats synchronous under variable run times. Gittins index gives a principled stopping rule |
| **Architecture scaling** | xLSTM scaling laws (Beck 2025), hybrid linear attention analysis (Wang 2025), Meta hybrid analysis (Bae 2025), MAD (Poli 2024), NVIDIA Mamba study, Kimi Linear, MiniMax M2 blog | Compare full scaling curves, not single points. The best pure linear mixer is not the best hybrid component. Add recall and multi-hop probes, since loss parity at small scale hid multi-hop deficits at MiniMax |
| **MoE / Engram / recurrence** | Clark 2022, Krajewski 2024, Ludziejewski 2025, Optimal expert-attention allocation (2026), Engram (DeepSeek 2026), Huginn, MoR, ParScale | Continuous architecture knobs (granularity, expert share ρ, attention ratio) have optima that move as power laws in C. Engram vs MoE share is U-shaped with the optimum at ρ ≈ 0.75–0.8 |

---

## 4. Overview: a layered probabilistic model and a decision layer

```
 runs (logs, W&B, eval)                                   cluster (JUPITER, SLURM)
        │                                                         ▲
        ▼                                                         │  sweep YAML (list groups)
 ┌──────────────────────────────┐                                 │  + monitor conditions
 │ A  curve model + QA          │ per run: L(t) posterior,        │
 │    (MPL / LR-annealing law)  │ anomaly flags, weight w_r,      │
 │                              │ predicted cooled loss + CI      │
 └───────────────┬──────────────┘                                 │
                 ▼                                                │
 ┌──────────────────────────────┐                                 │
 │ B  HP response surface       │ L(a,N,D,θ) = L*(a,N,D)          │
 │    + HP optimum laws         │   + bowl(θ − θ*_a(N,D))         │
 │                              │ θ*_a = power law, hierarchical  │
 └───────────────┬──────────────┘                                 │
                 ▼                                                │
 ┌──────────────────────────────┐                                 │
 │ C  envelope law L*(a,N,D)    │ + cost model c(a,N,D) (GPU-h)   │
 │    hierarchical over archs   │ → compute multipliers M_a(C,r)  │
 └───────────────┬──────────────┘                                 │
                 ▼                                                │
 ┌──────────────────────────────┐   ┌──────────────────────────┐  │
 │ D  decision layer            │──▶│ E  scheduler             │──┘
 │  - arm elimination / gates   │   │  seq-vs-par: launch x now│
 │  - acquisition (info / cost) │   │  iff E[waste] < λ·Δt     │
 │  - stopping rule             │   └──────────────────────────┘
 └──────────────────────────────┘
```

Principles:

1. **Everything is one joint model with explicit uncertainty.** A run is never "the best point of a
   grid cell". Every run, including off-optimum and truncated ones, is evidence about the shared
   parameters. This is what makes the procedure data-efficient.
2. **Functional forms are plug-ins.** Each layer takes its functional family from a registry
   (§11). Candidate forms are compared by **leave-one-scale-out extrapolation error**, never by
   in-sample fit. Choshen, Gemstones and Delphi v1 all show that in-sample fit quality is
   uninformative about extrapolation.
3. **Decide on what matters.** The acquisition targets the quantities we will act on: the ranking
   and compute multiplier of architectures in the target regime, $\theta^*$ at the target scale,
   and the final loss of the large run. It does not target global model accuracy.
4. **Cost is GPU-hours from measured throughput, and wall-clock is priced separately.** FLOPs are
   reported alongside, but decisions use GPU-h (`costs.py`, `aux.tok_s`).

Notation:

- $a$: architecture, a categorical choice plus continuous knobs $\xi$ (e.g. attention ratio,
  expert count)
- $N$: non-embedding parameters, and $N_{act}$ for MoE. Always also log total parameters.
- $D$: tokens; $r = D/N$
- $\theta$: HPs, with $u = \log\theta$
- $t$: step or tokens within a run
- $c(\cdot)$: GPU-hours

---

## 5. Layer A: loss-curve model and run quality assessment

### 5.1 Curve model

Use the **multi-power law** (Luo et al. 2025) as the default and the Tissue et al. law as a
cheaper alternative. Both are explicit in the LR schedule $\eta_t$, so a single fit covers the
trunk and all its cooldown branches:

$$
L(t) = L_0 + A\,\big(S_1(t) + S_W\big)^{-\alpha} - LD(t),\qquad S_1(t)=\sum_{\tau\le t}\eta_\tau
$$
$$
LD(t) = B\sum_{k\le t}(\eta_{k-1}-\eta_k)\,G\!\big(\eta_k^{-\gamma}S_k(t)\big),\quad G(x)=1-(Cx+1)^{-\beta},\quad S_k(t)=\sum_{\tau=k}^{t}\eta_\tau
$$

Tissue et al. use the simpler form $L(s) = L_0 + A S_1^{-\alpha} - C S_2$, where $S_2$ is a
momentum-like sum of LR decrements.

Uses:

- **Trunk → cooldown prediction.** Fit on the stable trunk, predict every $L_{\text{cooled}}(D)$
  with a posterior band. A cooldown then becomes a *measurement that checks the prediction*
  instead of the only source of information. Once the decay parameters $(B, C, \beta, \gamma)$
  are calibrated per architecture family (they transfer well across sizes), **most cooldowns
  can be skipped or shortened during screening.** Keep only the ones the design layer asks for.
- **Early stopping or extension of trunks** (§8.4). A trunk at 30% already constrains $A$ and
  $\alpha$.
- **Robust summary statistic.** The fitted $L_{cooled}(D)$ at *arbitrary* D feeds Layer C, so
  budgets need not match exactly, and intermediate D values come at no cost (Choshen et al.:
  intermediate checkpoints improve scaling fits).

Data source:

- Megatron train loss (log interval 10) and validation loss (every `204800//gbs` iterations, on
  the 1% held-out split) from `slurm-*.log`, TensorBoard or offline W&B.
- Fit on **validation loss**, with train loss as a dense secondary likelihood that has its own
  noise and offset parameters. Train loss is noisier, and any per-batch data-order structure is
  shared across architectures, because every run uses the same seed and data order.
- The fit is a small Bayesian model per run (NumPyro/NUTS, or Laplace around a least-squares
  fit).
- Hierarchical priors across runs of the same architecture family and across sizes
  (supercollapse), so a young run borrows strength from old ones.

### 5.2 Collapse prior (cross-size)

Qiu et al. 2025 and Bergsma et al. 2025 show that when HPs are scaled well, normalized curves
$\tilde L(\tilde t) = (L(t) - \hat L_\infty)/(L(0^+) - \hat L_\infty)$, with $\tilde t = t/T$,
**collapse onto one curve** across sizes. With decay, the residual spread is below seed noise.

Two uses:

- As a **prior** for the curve model of a new size: fast, confident predictions from very early
  in a run.
- As a **diagnostic**: a run that drifts off the collapse curve early, while its own curve looks
  smooth, is mis-scaled (for example the LR scaling rule is wrong for this architecture). It is
  *not* evidence against the architecture itself. Record this as "HP-suspect", not "arch-bad",
  and trigger an HP probe.

### 5.3 Anomaly detection

Define residuals against the **posterior predictive** of the curve model, $e_t = y_t - \hat
y_t$, scaled by a robust noise estimate $\hat\sigma$ (median absolute deviation over a trailing
window). Seed-to-seed noise is about 0.1% at Marin scales; we should measure our own $\sigma_{seed}$
once with 3 seeds at 0.1B and 0.2B.

| Anomaly | Detector | Notes |
|---|---|---|
| **Spike** (transient) | $z_t = e_t/\hat\sigma > z_{sp}$ (e.g. 6) on train loss, **and/or** grad-norm z-score vs EMA (ZClip-style) $> z_g$; record height, duration until $e_t$ returns to within 2σ, and the "scar" (persistent offset afterwards) | Spikes that recover without a scar are harmless. Log them, keep the run, and set $w_r = 1$. A scar larger than 0.5% of loss sets $w_r < 1$, and a repeated pattern sets $w_r = 0$ |
| **Bump** (sustained deviation) | CUSUM or Bayesian online change-point detection (Adams & MacKay 2007) on $e_t$; a bump is a change-point to positive mean residual lasting more than $k$ eval intervals | Typical causes: data-order artifacts (should appear in *all* architectures at the same t, which is a useful cross-check), LR-warmup instabilities, or norm growth in a late cooldown (Marin 8B: fixed by z-loss) |
| **Stall** (slow degradation) | Posterior probability that the observed loss *decrease* over a window is below $q$ (e.g. 0.7) × the predicted decrease; equivalently the drift of $\hat\alpha$ as more data arrives | Distinguish "curve genuinely flatter" (changes the architecture's fit, keep) from "run broken" (e.g. dataloader cycling, frozen gates, NaN-ed parameters). The collapse diagnostic and mechanism monitors separate the two |
| **Divergence** | Loss increases above $L(0)$-scale, or NaN | Hard fail. Restart with the monitor, but *count* divergences per (architecture, LR) |
| **Precursors** | Attention-logit max / QK spectral entropy (Huang et al. 2026), output-logit norm, lm_head norm, gate saturation (mLSTM igate, GDN β), router entropy (MoE) | These appear thousands of steps before the loss notices. Log them from Megatron hooks; they are cheap |

Outputs per run:

- $w_r\in[0,1]$, the solidity weight. It **multiplies the likelihood contribution** of the run in
  Layers B and C: a soft weight, not hard filtering. Record the reason.
- **Stability features:** divergence or spike rate as a function of LR multiplier. This gives an
  empirical **max stable LR** $\eta_{max}(a, N)$, fitted as its own power law.
  - This matters because an architecture with a narrow stability margin is risky to extrapolate
    even if its loss is good: the Marin 32B lesson.
  - It also bounds the HP search (below).
  - Wortsman's "LR sensitivity", $\mathbb E_{\eta}[L(\eta) - L(\eta^*)]$ over a fixed LR range, is
    the scalar version of the same idea.

These detectors run **online in the monitor** (§11) as a condition, so a broken run is cancelled
or restarted from the pre-anomaly checkpoint (OLMo/Marin style), not paid to completion.

---

## 6. Layer B: HP response surface and HP scaling laws

### 6.1 Joint "bowl" model instead of argmin per cell

The loss at HP $u = \log\theta$ is modeled as the envelope plus a penalty bowl:

$$
\log L(a,N,D,u) \;=\; \log L^*(a,N,D) \;+\; \phi\big(u - u^*_a(N,D);\,H_a(N,D)\big) \;+\; \varepsilon
$$

- $\phi$ is a bowl with $\phi(0)=0$. The default is a quadratic $\delta^\top H\delta$. Better is an
  **asymmetric** form with a steeper wall at high LR (divergence) and a softer one at low LR.
  Step Law 2025 reports a convex surface with a broad plateau, which also suggests a quartic or
  "softplus-wall" alternative. The form is a plug-in (§11).
- The curvature $H$ may itself scale mildly with $(N, D)$. It gets a weak log-linear model.

**Why this matters:** with one run per cell (our current state), argmin selection gives no
information at all about $u^*$. With the bowl, *every* run at *every* HP contributes to both
$u^*$ and $L^*$. Three LR points at one scale plus single points elsewhere already identify the
LR law. This also avoids the biases of fitting IsoFLOP parabolas per cell (Czech et al. 2026).

### 6.2 HP optimum laws

For each HP that turns out to be scaling-sensitive, in the Fantastic Optimizers Phase-II sense:

$$
u^*_{a,j}(N,D) = c_{a,j} + p_{a,j}\log N + q_{a,j}\log D \qquad(\text{power law } \theta^*_j = e^{c}N^{p}D^{q})
$$

Default parametrization choices, informed by the literature:

- **Learning rate.** $\log\eta^* = c + p\log N + q\log D$, with prior means $p\approx -0.2$ to
  $-0.7$ (N exponents across DeepSeek, Porian, Step Law and Bjorck span this range and depend on
  the parametrization) and $q \approx -0.3$ at fixed batch (Bjorck; Marin's 0.3). Step Law's positive D-exponent (+0.307)
  comes from letting B grow with D, which is why the law below carries an explicit $\log B$ term
  (prior slope ≈ +0.5, the √B rule). Use wide priors.
  Also offer the *effective-LR* parametrization of Yang et al. 2026: fit in normalized-weight
  step size and extrapolate along D. This removes the upward curvature of $\log\eta^*$ seen at
  larger N.
- **Batch size.** $\log B^* = c_B + q_B\log D$ with **no N term a priori** (Power Lines: $B^*\propto D^{0.383}$;
  Zhang et al. 2024: critical batch $\propto D^{0.47}$). Add an N term only if the evidence demands it.
- **Weight decay.** Reparametrize as the AdamW timescale $\tau = B/(\eta\lambda D)$ (Wang &
  Aitchison; Bergsma et al. find $\tau^*\propto (D/N)^{-0.53}$). Fitting $\tau^*$ instead of
  $\lambda^*$ makes the WD optimum nearly scale-free and decouples it from LR and batch. For
  AdamH/MuonH-type optimizers, WD disappears entirely.
- **β2.** Use half-life in tokens (Marin: β2 = 0.9999^(B/B0)), not β2 itself.
- **Warmup.** Use tokens or fraction of steps. It is usually scaling-insensitive: *test* that and
  then freeze it.

Scaling-insensitive HPs (the Fantastic Optimizers test: one value is near-optimal in every
sampled regime) are **fixed**. Model complexity is spent only where the posterior says the
optimum moves.

**Uncertainty.** The posterior over $(c,p,q)$ gives, for any target $(N_T,D_T)$:

- $\hat u^*$ and its standard deviation. These are the confidence bounds on the optimal HP asked
  for in (b).
- **Expected regret from HP error at the target:**
  $\mathbb E[\phi(\hat u^* - u^*)] \approx \mathrm{tr}(H\,\Sigma_{u^*})$. This converts HP
  uncertainty into *loss* units, which is the currency the design layer uses (§8).

### 6.3 Hierarchical sharing across architectures

Most architecture changes *shift* the HP optimum more than they change its *slope*. So:

$$
c_{a,j} = c_{0,j} + \delta c_{a,j},\qquad p_{a,j} = p_{0,j} + \delta p_{a,j},\qquad
\delta c \sim \mathcal N(0,s_c^2),\ \delta p\sim\mathcal N(0,s_p^2),\ s_p \ll s_c
$$

Nested families share further levels: GDN variants → linear mixers → all. Consequences:

- A new architecture needs only **a few LR probes at one or two small sizes** to pin its offset
  $\delta c_a$. It inherits the slopes until data contradicts them.
- The shrinkage hyperparameters $s_c, s_p$ are learned, so if architectures *do* differ in slope,
  the model finds out, and the design layer will then ask for data at a second scale.

Every new variant should first get an **LR offset probe**: 3 LRs (×0.5, ×1, ×2 around the
predicted optimum) on the smallest size, on a short trunk only. This fixes the "fixed-HP ranking
flip" problem (Wang et al. 2025) at about 5% of the cost of the variant's ladder.

---

## 7. Layer C: envelope loss law, cost model, architecture comparison

### 7.1 Envelope law

Default form (Chinchilla, per architecture, hierarchical):

$$
L^*(a,N,D) = E + A_a N^{-\alpha_a} + B_a D^{-\beta_a}
$$

- $E$ is **shared** across architectures by default, since it is the irreducible loss of the data
  distribution. Allowing $E_a$ to differ is a model-comparison question, and in practice it trades
  off almost perfectly against $\alpha_a$ on small ranges. That is exactly the weak identification
  the current bootstrap CIs show.
- $\log A_a, \alpha_a, \log B_a, \beta_a$ are hierarchical, like the HP laws: base plus shrunken
  deltas.

Alternative forms in the registry:

- Farseer ($N$-dependent D-exponent)
- Kaplan-style joint $(N^{\alpha/\beta}/D + \dots)^\beta$
- Broken power laws (Caballero et al.) for regime changes
- MoE-extended forms (§10)
- The **D/N-slice form** $L^*(a, N; r) = E_r + A_{a,r}N^{-\alpha_{a,r}}$ at fixed $r$ (Marin hero
  ladder). This is robust when all we need is the overtrained regime and not the compute-optimal
  frontier.

Fitting:

- Minimize a **summed** robust loss (Huber on $\log L$, not *averaged*; Besiroglu et al.). Use
  variable projection (linear parameters in closed form given the exponents) and multistart; or
  go full-Bayes with NUTS.
- Size-stratified and **design-stratified bootstrap**: resample which sizes, budgets and HP
  points are included (Gemstones), not just residuals.

### 7.2 Cost model

$$
c(a,N,D) = \frac{D}{\text{tok\_s}_a(N)\cdot 3600}\ \text{GPU-h},\qquad \text{tok\_s}_a(N) = k_a N^{-\sigma_a}
$$

This is measured, not 6ND. It already exists as `costs.py` and `aux.tok_s`, with σ ≈ 0.68.

- Linear and hybrid mixers have a context-length-dependent advantage. Store $k_a$ per
  sequence length.
- FLOPs (6ND + attention) are carried along for reporting and cross-lab comparability.
- Decisions use GPU-h because that is what JUPITER bills.
- Optionally add an **inference-cost term** (Sardana et al.; Bian et al.) for architectures whose
  selling point is decode speed or KV-cache size.

### 7.3 The comparison metric: effective compute multiplier in the target regime

For architecture $a$ vs incumbent $b$ at a GPU-h budget $C$ and D/N ratio $r$:

$$
M_{a}(C, r) \;=\; \frac{c_b\big(L^{*-1}_{b}(L^*_a(N_a(C,r), D_a(C,r)))\big)}{C}
$$

In words: how many GPU-h the incumbent would need to reach $a$'s loss. $M_a > 1$ means $a$ is
better. This is Marin's effective speedup and our existing `iso_loss_multiplier.csv`, now
computed from posteriors.

The **target regime** $\mathcal R$ is a set of $(C, r)$:

- $C \in [C_{\text{next big run}}, 10\times]$
- $r \in [100, 1000]$

Its posterior is estimated by Monte Carlo over Layers B and C, which includes the **HP regret**
$\mathrm{tr}(H\Sigma)$ for both arms. An architecture whose optimal LR we do not know yet pays
for that uncertainty.

Report for every arm:

- $\Pr[M_a(C,r) > 1]$
- the credible interval of $M_a$
- the ratio vs fullattn (as today)
- **parameter counts (total, non-embedding, active) and measured tok/s** next to every loss number

---

## 8. Layer D: sequential design (Bayesian optimization)

### 8.1 What an "action" is

An action $x$ is **a trunk plus a set of branches**, which is the unit our sweep system already
produces:

- $x = (a, \xi, N, \theta, D_{\text{trunk}}, \{D_k\}\ \text{cooldowns})$
- Cost: $c(x) = c(\text{trunk}) + \sum_k c(\text{cooldown}_k)$, with a cooldown costing about
  $0.2\,D_k$ worth of tokens.

Two extra action types exploit the WSD structure. They are the **multi-fidelity** handles:

- **extend**: continue an existing trunk to a larger $D$ (freeze-thaw).
- **branch**: add a cooldown at an existing persistent checkpoint.

Both require a **geometric grid of persistent checkpoints** on every trunk, e.g. at
$D\in\{2^{k/2}\}$ tokens, not only at pre-planned `extra_steps`. Without such a checkpoint,
branching at an unplanned point means re-running. Storage cost is modest at ≤1B, and
`scripts/prune_decay_checkpoints.sh` can prune retired arms.

### 8.2 Acquisition: information about the decision, per cost

Let $Q$ be the quantity we will act on in the current phase:

| Phase | $Q$ |
|---|---|
| Screening | the vector of $\mathbb 1[M_a(\mathcal R) > 1-\delta]$ for all live arms, i.e. which arms stay |
| HP-law fitting | $u^*(N_T, D_T)$ for the chosen arm(s) at target scale |
| Pre-extrapolation | $L^*(a, N_T, D_T)$ itself (can we predict the big run within tolerance?) |

The acquisition is the expected information gain about $Q$, per GPU-hour:

$$
\alpha(x) = \frac{\mathrm{IG}(Q;\,y_x\mid\mathcal D)}{c(x)}
\;\approx\; \frac{\tfrac12\log\det\Sigma_Q - \mathbb E_{y_x}\big[\tfrac12\log\det\Sigma_{Q\mid y_x}\big]}{c(x)}
$$

This is the design principle of PLES (power-law entropy search), SL², taKG and CARBS. Because our
models are *parametric*, it is cheap to compute:

- **Linearize** the model around the posterior mean (Laplace / Fisher). $\Sigma_{Q|y_x}$ then
  follows from a rank-1 update per observation, which is milliseconds per candidate.
- Or use **fantasy updates**: sample $y_x$ from the posterior predictive, refit by importance
  reweighting of posterior samples, and measure the change in variance of $Q$.
- Candidates come from a structured set: the (a, N) grid of feasible model shapes × LR multipliers
  around $\hat u^*$ × D values on the checkpoint grid. This is a few thousand candidates, and
  exhaustive scoring is fine.

The power-law structure is where the efficiency comes from:

- Information about an exponent is maximized by **spreading** in $\log N$ / $\log D$. The
  optimal design for a linear model in log-space places mass at the extremes of the affordable
  range, plus some interior points to test the functional form.
- The acquisition naturally proposes "one more point at the largest affordable N" once the small
  end is saturated. This is the well-known result that *several small models plus one or two
  larger ones* beat a dense grid (Choshen et al.).
- The HP regret term makes the acquisition place LR probes **at the scale where $\Sigma_{u^*}$
  extrapolates worst**, which is typically the largest affordable N at the largest D/N.

**Model-form uncertainty.** Maintain several functional forms with posterior weights, e.g. from
leave-one-scale-out predictive likelihood. Add a term for **disagreement between forms** at the
target, a "query-by-committee" term, so the design also tests the form, not only its parameters.

### 8.3 Arm elimination (screening) with explicit risk

For each live arm $a$ and the incumbent $b$, after each batch of results:

- **Drop** $a$ if $\Pr\big[M_a(C,r) > 1 - \delta\ \text{for some }(C,r)\in\mathcal R\big] < p_{drop}$.
  - Example values: δ = 0.05, meaning "within 5% compute of the incumbent", and $p_{drop}$ = 0.05.
  - The "for some" makes dropping conservative: an arm that might win *only* at high D/N
    survives.
- **Promote** $a$ to incumbent if $\Pr[M_a > 1+\delta\ \forall (C,r)\in\mathcal R] > 1-p_{drop}$.
- Otherwise it stays **live**, and the acquisition decides whether it deserves more data.

This generalizes Marin's pass/fail gates (2 scales, then 4 scales, then projection to 1e21/1e23)
to posterior probabilities. It also generalizes successive halving / ASHA to a *model-based*
halving: arms are dropped on extrapolated evidence, not on rank at the current fidelity. PASHA's
insight still applies: grow the fidelity only while the posterior ranking is unstable.

Rules that keep elimination honest:

1. **Only cooled or cooldown-predicted losses count.** Rankings flip during LR decay (Wen et al.).
   The curve model's cooled prediction is fine once its decay parameters are calibrated for that
   architecture family: at least 2 measured cooldowns per family.
2. **Each arm is compared at its own inferred HP optimum**, plus its HP regret, never at the
   incumbent's HPs.
3. **Stability is a veto, not a tie-breaker.** An arm with $\eta_{max}(a,N)$ approaching
   $\eta^*(a,N)$ as N grows (the ratio's power-law fit crosses 1 before the target) is dropped or
   sent back for a stabilization fix (QK-norm, z-loss, gate init) regardless of loss.
4. **Capability probes for mixers.** Loss parity can hide recall and multi-hop deficits (MiniMax
   M2). Add cheap probes to the eval: MAD-style synthetic recall/copy tasks, needle and
   multi-query associative recall at 4k/16k, and a multi-hop QA set. They are gates for
   *promotion*, not for dropping. Loss decides dropping. Downstream accuracy is unreliable at
   small scale (Lourie; DataDecide).
5. **Speedups tend to shrink with scale** (Wen et al.; many "free lunches"). The hierarchical
   prior on $\delta\alpha_a$ must not be so tight that it forces a small-scale advantage to
   persist. Test calibration by holding out the largest size in retrospective analyses of our
   current sweep (§11, step 1).

### 8.4 Early stopping and extension of individual runs (within an arm)

During a trunk, the curve model posterior updates at each validation interval. Using the
**freeze-thaw / ifBO** logic over running trunks:

- **Stop** a trunk if the remaining steps' expected information, $\mathrm{IG}(Q; y_{\text{rest}})$
  per cost, falls below the current best alternative's $\alpha$.
  - Example: its LR probe is already clearly off-optimum, and the bowl is already identified.
  - Example: the arm was just dropped.
  - The partial curve and any existing branch checkpoints remain valid data.
- **Extend** a trunk if the acquisition prefers more D on this $(a, N)$ over a new run.
- **Restart from the pre-anomaly checkpoint** on a detected spike with scar, with the data order
  shifted or skipped (OLMo/Marin practice). If it repeats, treat it as an arm-level stability
  finding.

### 8.5 Stopping the search / when to launch the extrapolation run

Two complementary rules:

- **Value of information.** Stop exploring when the best $\alpha(x)\cdot$ (value of reducing $Q$)
  falls below the cost. The Pandora's-box **Gittins index** (Xie et al. 2024) gives the
  cost-aware form: stop when $\max_x g(x) \le$ current best, where
  $\mathbb E[(f(x) - g)_+] = c(x)$.
- **Extrapolation readiness.** Launch the big run once three conditions hold:
  1. The 90% predictive interval of $L^*(a, N_T, D_T)$ is narrower than the decision tolerance.
  2. The HP regret $\mathrm{tr}(H\Sigma_{u^*}) <$ ε (e.g. 0.2% loss).
  3. **A held-out-scale check passes.** Refit without the largest size, predict it, and require
     the error to be within its predicted interval. This is Delphi's validation step, applied
     *before* spending the big budget, since that is where Delphi v1 failed.

The big run itself then becomes a data point with a pre-registered prediction and a
`FileContent`-based monitor condition that compares actual vs predicted loss at each validation
step (§11).

---

## 9. Sequential vs parallel execution: the wall-clock / compute trade-off

Running 3 experiments now in parallel gives results 3× sooner. Running them sequentially lets
experiment 1 change (or cancel) experiments 2 and 3. We model this directly rather than by rule
of thumb.

### 9.1 Objective

Minimize total cost with wall-clock priced at a rate $\lambda$ (GPU-h per hour of delay):

$$
J = \text{GPU-h spent} + \lambda\cdot T_{\text{wall}}\qquad\text{(or: min GPU-h s.t. } T_{\text{wall}}\le T_{\text{deadline}}\text{; then }\lambda\text{ is the Lagrange multiplier)}
$$

$\lambda$ is set in the planner config and can change over time:

- High before a paper or allocation deadline.
- Near zero when the cluster queue is the bottleneck anyway.
- Effectively **negative compute cost** when an allocation is about to expire unused. Idle node
  hours that would be forfeited should be priced at about 0, which makes parallel launches
  nearly free.

### 9.2 Launch rule: expected commitment waste vs wall-clock gain

Let $P$ be the set of pending (running) actions and $x$ a candidate the planner currently ranks
above the launch threshold. The alternative to launching $x$ now is waiting for (partial) results
of $P$ at time $\Delta t_x$, the time until the pending results that are *informative about $x$*
arrive.

1. **Fantasize** $K$ outcomes $y_P^{(k)}$ of the pending runs from the posterior predictive, at
   the time the decision would be revisited.
2. For each fantasy, update the model and re-run the planner. Record whether $x$ would still be
   launched, or how much of it (e.g. a lower D, a different LR). Define the value $v_k(x)$ of the
   compute already spent on $x$ under fantasy $k$: the full $c(x)$ if still selected, the salvage
   value if changed (e.g. a trunk that is still useful as a probe), and 0 if obsolete.
3. **Expected commitment waste:** $W(x) = c(x) - \frac1K\sum_k v_k(x)$.
4. **Launch now iff** $W(x) < \lambda\cdot\Delta t_x\cdot\kappa_x$, where $\kappa_x\in[0,1]$ is
   the probability that $x$ is on the **critical path** to the final decision. A run that is not
   on the critical path gains nothing from starting early.

All of this is cheap, because the Layer B/C models are small parametric models; a planner pass
takes seconds.

The rule reproduces the right intuitions:

| Situation | $W(x)$ | Decision |
|---|---|---|
| Different arms at the same screening stage (independent questions) | ≈ 0: arm 1's result rarely changes what we want to learn about arm 2 | **parallel** |
| Different LR probes of the same arm at the same N | small if the batch is spread by the acquisition, which already accounts for pending runs | **parallel** |
| Larger N of an arm while its small-N results are pending | large: the small result may drop the arm (probability $1-\Pr[\text{survive}]$) or shift $\hat u^*$ | **sequential** unless λ is high |
| Extrapolation run while the held-out-scale check is pending | very large | **sequential** (hard rule) |
| Allocation expiring, nodes idle | $c(x)$ effectively ≈ 0 | **parallel**, launch the next most useful actions |

### 9.3 Making sequential cheaper: decide on partial results

$\Delta t_x$ is not the full duration of the pending runs.

- With the curve model, an arm at 20–30% of its trunk often already has
  $\Pr[\text{survive}] < p_{drop}$ or > 0.95.
- The planner therefore re-evaluates **at every validation interval** (event-driven through the
  monitor, §11), not at run end. This "semi-sequential" mode (launch, then re-decide on partial
  curves) captures most of the adaptivity at a fraction of the wall-clock.
- It works because WSD trunks are *anytime*: stopping early loses nothing already paid for.
- JUPITER's 12h job limit already splits runs into chained segments, and each chain boundary is a
  natural decision point.

### 9.4 Batch construction when launching in parallel

When several actions go out at once, select them **greedily with fantasized pending
observations**: Kriging believer / q-acquisition with `X_pending` (BoTorch-style), or local
penalization.

- This keeps the batch diverse in the information it delivers, which is the condition under
  which batching costs only a constant factor over sequential (GP-BUCB, parallel Thompson
  sampling).
- Prefer **asynchronous** replenishment (launch as soon as a slot or result frees up) over
  synchronous rounds. On SLURM with variable queue and preemption times, synchronous rounds wait
  on stragglers. Kandasamy et al. and ASHA show asynchronous is strictly better in wall-clock at
  equal regret per evaluation.

Practical defaults, until λ is calibrated:

- **Parallelize across independent questions:** architectures, axes, and HP probes within one
  scale.
- **Serialize along scale** within one question: small informs large.
- Never launch a run whose $\Pr[\text{still wanted after pending results}] < 0.5$ unless nodes
  would otherwise idle.

---

## 10. Architecture axes: how each enters the model

Two kinds of architecture knobs:

- **Categorical** (which mixer, which optimizer, which norm scheme): arms $a$ with hierarchical
  offsets in Layers B and C, eliminated by §8.3.
- **Continuous or ordinal** (attention ratio, window size, expert count/granularity, Engram share,
  recurrence count, state size): treated **like HPs**, with a bowl and an optimum that moves as a
  power law in $(N, D)$ or $C$. The optimal expert-to-attention FLOP ratio follows a power law in
  compute (2026 MoE allocation paper), and the Engram/MoE allocation is U-shaped with a stable
  optimum (DeepSeek Engram). Inherit the Layer B machinery: $\log\xi^* = c + p\log C$ (or
  $(N, D)$).

Some knobs are both. "GDN hybrid" is a categorical arm, and within it the linear-to-full ratio is
continuous. Screen the category at its prior-best ratio (3:1–7:1 per Wang et al. 2025) and fit
the ratio law only for survivors.

| Axis | Arm / knob | Specific modelling notes |
|---|---|---|
| **Sequence mixer / hybrid** (this branch) | Arms: fullattn, SWA, GDN, mLSTM, Mamba2, PDA, SKDA, Qwen3.5-style; knobs: linear:full ratio, window, NoPE vs RoPE, conv, gate inits | Throughput $k_a(N,\text{ctx})$ differs strongly, so compare in GPU-h. The best pure mixer is not the best hybrid component, so screen *as hybrids*. Add recall and multi-hop probes before promotion. Gate saturation (mLSTM igate, GDN β) as anomaly precursors |
| **Optimizer** | Arms: AdamW, Muon/MuonH, AdamH, Soap, … | Each has its own HP law set. Run the Fantastic Optimizers Phase-II sensitivity test to find which HPs need laws. Expect speedups to shrink with N, so the elimination rule needs the high-D/N part of $\mathcal R$ |
| **Backbone** | norm placement, QK-norm, residual scaling (nGPT, CompleteP α), depth:width | Depth:width is a continuous knob. Marin's $L(d)$ rule is a prior. **Stability arms** (QK-norm, z-loss) go through the $\eta_{max}$ analysis in §5.3 rather than loss alone |
| **MoE** | knobs: expert count $E$, granularity $G$, top-k, shared experts | N becomes $(N_{act}, N_{tot})$. Use MoE-aware law forms (Krajewski: $c + (g/G^\gamma + a)/N^\alpha + b/D^\beta$; Ludziejewski joint form with $\hat E$). Cost must include all-to-all communication through measured throughput. Router entropy and load imbalance as anomaly precursors |
| **Engram / conditional memory** | knob: share ρ of sparse parameters in Engram vs MoE, table size, n-gram order | Parameters that are neither active nor FLOP-bearing: log them as a *third* size axis $N_{mem}$ and fit $L$ with a memory term. The U-shape in ρ is a bowl, and its optimum location is fit as a law in C |
| **Recurrence in depth** | knob: recurrence count $r$ (train and test), prelude/coda sizes | Parameters ≠ FLOPs: $N$ (unique params) and effective depth enter separately. ParScale-like form $N_{eff} = N\cdot(1 + \kappa\log r)$ as a first plug-in. Cost is FLOP-dominated; measure it. Hidden-state collapse (Huginn) as an anomaly class |

**Parameter accounting is part of the design, not reporting.** Every arm must log total,
non-embedding, active and memory parameter counts, measured tok/s, and FLOPs/token computed from
the model (not the dense Qwen3 formula in `costs.py`, which is known to undercount some variants).
Otherwise $L^*(N)$ is fit on the wrong x-axis.

---

## 11. Implementation plan in this repo

> **Status (2026-09-23): a first prototype exists in [`optimization_procedure/`](../optimization_procedure/README.md).**
> It covers the hierarchical robust fit with learned cross-architecture shrinkage and LOO
> hold-out, curve QA, compute multipliers with elimination, information-gain design, the
> seq-vs-parallel scheduler, and a calibration ledger. It runs end-to-end on the repo's sweep data
> (`python -m optimization_procedure.run_first_test`) and has synthetic ground-truth tests. Its
> README lists first findings and limitations: a diverged mamba2 0.9B run, calibration inflation
> of about 2 from held-out-size validation, and a poorly identified LR optimum. The module plan
> below is the target layout. The prototype implements a subset of it with a flatter file
> structure.

Proposed package: `oellm_autoexp/scaling/`. It grows out of `scripts/scaling_laws/`, which
remains the reporting front-end.

| Module | Content |
|---|---|
| `ingest.py` | Parse `slurm-*.log`, offline W&B and TensorBoard into a run table (config hash, arch, N counts, θ, schedule array $\eta_t$, loss and val curves, grad norm, precursor stats, tok/s). Reuse `extract.py`, `live_throughput.py` and `dump/scaling_experiments*/` exporters |
| `forms/` | **Functional-form registry**: each form is a small class with `params`, `priors`, `predict(X)`, and optionally `jacobian`. Curve forms (MPL, Tissue, power + exp); bowls (quadratic, asymmetric, quartic); HP laws (log-linear, effective-LR, D-only for B); envelope laws (Chinchilla, Farseer, fixed-r slice, MoE). New forms are added by config, e.g. `scaling.forms.envelope=farseer` |
| `curve_model.py` | Per-run posterior, collapse prior, cooled-loss prediction |
| `anomaly.py` | Detectors from §5.3 → $w_r$ and flags; CLI usable from a monitor `CommandCondition` |
| `model.py` | Joint hierarchical model (Layers B and C). NumPyro NUTS for final fits, Laplace/variable projection for the inner planner loop. Design-stratified bootstrap as a sanity check |
| `validate.py` | Leave-one-scale-out and leave-one-arm-out predictive checks; model-form weights |
| `design.py` | Candidate generation, acquisition (IG/cost with Laplace updates), arm elimination, stopping rules |
| `scheduler.py` | Launch rule of §9.2 (fantasies → $W(x)$ vs $\lambda\Delta t\kappa$), async batch construction. **Emits sweep YAML** (a `list` group of concrete points) that composes against the existing base configs, so `run_autoexp.py` / the monitor stay unchanged |
| `ledger.jsonl` | Append-only log of every decision: model state hash, candidates, chosen actions, predictions registered *before* the results arrive. Makes the procedure auditable and lets us score the planner's calibration |

Integration points:

- **Checkpoint grid.** Add an option to set `save_extra_steps` to a geometric token grid on all
  trunks (§8.1), so `branch` and `extend` actions are always possible.
- **Monitor.**
  - Add an `AnomalyCondition` (wrapping `anomaly.py`) with actions Restart-from-checkpoint or
    Cancel.
  - Add a `PlannerTick` action on each new validation line: run `design.py` and possibly emit
    new jobs through the existing `NewJob` action. This is the event-driven semi-sequential mode
    of §9.3.
- **Throughput.** The `aux.tok_s` tables feed $c(x)$ and $\Delta t_x$. Queue-wait estimates can
  come from `sacct` history.

### Roadmap

1. **Retrospective on existing data (no new compute).**
   - Build `ingest`, `curve_model` and `anomaly` on the dense multilingual baseline (1166 runs, 272
     with val curves) and on the current main sweep.
   - Calibrate $\sigma_{seed}$, the MPL decay parameters per family, and anomaly thresholds.
   - Refit Layers B and C, where the baseline sweep's (gbs, lr) grid finally gives the bowl real
     data.
   - **Backtest the planner:** replay the sweep in time order and check what it would have
     launched, dropped and predicted. Measure decision accuracy vs the final full-grid results
     (DataDecide-style) and the compute that would have been saved.
2. **LR offset probes** (§6.3) for every live mixer arm at 0.1B and 0.2B: 3 LRs on short
   trunks. This fixes the "non-envelope" weakness of the current comparison at small cost.
3. **Model-based screening** for new arms (optimizer, backbone, MoE, Engram, recurrence) using
   §8.3. Target: every new idea gets an up/down decision for 5–10% of a full ladder's GPU-h.
4. **HP-law and envelope-law fitting** for survivors with the IG acquisition, plus the
   held-out-scale check.
5. **First extrapolation run** (e.g. 3–8B at D/N ≈ 100–300) with a pre-registered prediction and
   interval.

---

## 12. Pitfalls and guard rails

- **Fit ≠ extrapolation.** Always validate on a held-out scale before trusting a law (Delphi v1).
  Model-form weights come from out-of-sample prediction only.
- **Tier switches.** Never splice HP regimes inside one $L(D)$ fit (`hpselect.report_tier_switches`).
  The bowl model removes the need for tiers: HPs become covariates, not strata.
- **Approach-2 biases.** IsoFLOP parabolas are biased by grid width, off-centering and asymmetry
  (Czech et al. 2026). Our parametric plus bowl approach avoids them, but parabola checks, if
  used, must be centered and symmetric.
- **Averaged vs summed losses and optimizer stopping** in Approach-3 fits (Besiroglu et al.).
- **Weak identification.** With a 12× N range, $E$, $\alpha$ and $A$ trade off. Use shared $E$,
  hierarchical priors, and report *predictions at target* with intervals rather than exponents.
- **Rankings flip in cooldown** and at fixed HPs. Compare cooled (or cooldown-predicted) losses
  at each arm's own HP optimum.
- **Data-order confounds.** All arms share seed and data order, which is good for matched pairs.
  But bumps that appear across all arms at the same step are data, not architecture. Detect them
  jointly.
- **Downstream metrics** are noisy and poorly predictable at small scale. Use them as promotion
  gates at the largest affordable size, not for elimination.
- **Throughput drift** (GPFS stalls, node quality) affects GPU-h cost. Use the *median* measured
  tok/s per (arch, N, parallel layout), not per-run wall-clock.
- **Precision.** fp8 or bf16 changes can alter both loss and stability (Kumar et al., precision
  scaling laws). Treat precision as part of the arm.
- **Planner overconfidence.** Score the ledger: of the arms dropped at $p_{drop}$ = 5%, fewer than
  about 5% should later turn out competitive (spot-check by occasionally continuing a dropped arm,
  i.e. ε-exploration of about 5% of compute).

---

## 13. References

Legend: all arXiv IDs below were resolved against the arXiv API on 2026-09-23 (title, first author, year). Marin items refer to
`github.com/marin-community/marin` (HEAD 2026-09-22) and `marin.readthedocs.io`.

**Marin**
- Marin repo: `lib/marin/src/marin/scaling_laws/isoflop_analysis.py`, `docs/recipes/add_scaling_heuristic.md`,
  `experiments/references/{reference_scaling_suite,completed_adamh,reference_hyperparameter_sweep}.py`,
  `experiments/grug/moe/{heuristic.py,agent.md}`, `experiments/grug/moe_hero_ep/`, `docs/reports/agent-moe-experiments.md`.
- W. Held, "Scaling Laws That Extrapolate 300× Past the Fit" (Delphi), Open Athena blog, 2026-05-11. https://openathena.ai/blog/delphi/
- Marin 8B retrospective: https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/
- Marin 32B retrospective: https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/
- Marin speedrun: https://marin.community/speedrun/
- Wen, Hall, Ma, Liang, "Fantastic Pretraining Optimizers and Where to Find Them", arXiv 2509.02046.
- Wen, Dang, Lyu, Ma et al., "Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization", arXiv 2606.16899.
- Held, Hall, Liang, Yang, "Relative Scaling Laws for LLMs", arXiv 2510.24626.

**Loss curves, schedules, collapse**
- Tissue et al., "Scaling Law with Learning Rate Annealing", arXiv 2408.11029.
- Luo et al., "A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules", arXiv 2503.12811.
- Hägele et al., "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", arXiv 2405.18392.
- Bergsma et al., "Straight to Zero: Why Linearly Decaying the LR to Zero Works Best", arXiv 2502.15938.
- Qiu et al., "Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks", arXiv 2507.02119.
- Bergsma et al., "Scaling with Collapse: Efficient and Predictable Training of LLM Families", arXiv 2509.25087.
- Meterez et al., "Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging", arXiv 2602.03702.

**Instability**
- Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities", arXiv 2309.14322.
- Takase et al., "Spike No More", arXiv 2312.16903.
- Molybog et al., "A Theory on Adam Instability in Large-Scale ML", arXiv 2304.09871.
- ZClip, arXiv 2504.02507; AdaGC, arXiv 2502.11034; SPAM, arXiv 2501.06842.
- OLMo 2, arXiv 2501.00656.
- Huang et al., "Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability", arXiv 2606.28116.
- Adams & MacKay, "Bayesian Online Changepoint Detection", arXiv 0710.3742.

**HP scaling and transfer**
- DeepSeek-AI, "DeepSeek LLM", arXiv 2401.02954 ($\eta^*=0.3118C^{-0.125}$, $B^*=0.292C^{0.3271}$).
- Porian et al., "Resolving Discrepancies in Compute-Optimal Scaling of Language Models", arXiv 2406.19146.
- Li et al. (StepFun), "Predictable Scale: Part I — Step Law", arXiv 2503.04715 ($\eta^*=1.79N^{-0.713}D^{0.307}$, $B^*=0.58D^{0.571}$).
- Bjorck et al., "Scaling Optimal LR Across Token Horizons", arXiv 2409.19913.
- Bergsma et al., "Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training", arXiv 2505.13738.
- Wang & Aitchison, "How to set AdamW's weight decay as you scale model and dataset size", arXiv 2405.13698.
- Zhang et al., "How Does Critical Batch Size Scale in Pre-training?", arXiv 2410.21676 (critical batch $B^*=22.91\,D^{0.47}$).
- Yang et al., "On the Nonlinearity of Learning Rate Scaling for LLM Training", arXiv 2606.29158.
- Dey et al., "CompleteP", arXiv 2505.01618; Mlodozeniec et al., "Completed Hyperparameter Transfer…", arXiv 2512.22382;
  Blake et al., "u-µP", arXiv 2407.17465; Everett et al., "Scaling Exponents Across Parameterizations and Optimizers", arXiv 2407.05872.
- Essential AI, "Practical Efficiency of Muon for Pretraining" (telescoping sweeps), arXiv 2505.02222;
  Moonshot, "Muon is Scalable for LLM Training", arXiv 2502.16982.

**Scaling-law estimation and prediction**
- Hoffmann et al. (Chinchilla), arXiv 2203.15556; Besiroglu et al., "Chinchilla Scaling: A Replication Attempt", arXiv 2404.10102.
- Choshen, Zhang, Andreas, "A Hitchhiker's Guide to Scaling Law Estimation", arXiv 2410.11840.
- Czech et al., "Problems with Chinchilla Approach 2", arXiv 2603.22339.
- McLeish et al., "Gemstones", arXiv 2502.06857.
- Caballero et al., "Broken Neural Scaling Laws", arXiv 2210.14891.
- Pearce & Song, "Reconciling Kaplan and Chinchilla Scaling Laws", arXiv 2406.12907.
- Farseer, arXiv 2506.10972; Brandfonbrener et al., "Loss-to-Loss Prediction", arXiv 2411.12925;
  Ruan et al., "Observational Scaling Laws", arXiv 2405.10938; Kumar et al., "Scaling Laws for Precision", arXiv 2411.04330.
- Bhagia et al., "Establishing Task Scaling Laws via Compute-Efficient Model Ladders", arXiv 2412.04403.
- Magnusson et al., "DataDecide", arXiv 2504.11393; Lourie et al., "Scaling Laws Are Unreliable for Downstream Tasks", arXiv 2507.00885;
  Krajewski et al., "Revisiting the Scaling Properties of Downstream Metrics", arXiv 2512.08894.
- Nezhurina et al., "Scaling Laws for Robust Comparison of Open Foundation Language-Vision Models and Datasets", arXiv 2506.04598.

**Sequential design, BO, multi-fidelity**
- Chen et al. (Meta), "Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search", arXiv 2609.01431.
- Li et al., "Spend Less, Fit Better" (SL²), arXiv 2604.22753.
- Schram et al., "Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning", arXiv 2605.17234;
  Schram et al., "Zero-Shot Performance Prediction for Probabilistic Scaling Laws", arXiv 2510.16743.
- Lee et al., "Bayesian Neural Scaling Law Extrapolation with PFNs", arXiv 2505.23032.
- Fetterman et al., "Tune As You Scale" (CARBS), arXiv 2306.08055.
- Swersky et al., "Freeze-Thaw BO", arXiv 1406.3896; Adriaensen et al., LC-PFN, arXiv 2310.20447;
  Rakotoarison et al., ifBO, arXiv 2404.16795; Wistuba et al., DyHPO, arXiv 2202.09774.
- L. Li et al., Hyperband arXiv 1603.06560; L. Li et al., ASHA arXiv 1810.05934; Falkner, Klein, Hutter, BOHB arXiv 1807.01774; Bohdal et al., PASHA arXiv 2207.06940.
- Wu, Toscano-Palmerin, Frazier, Wilson, taKG, arXiv 1903.04703.
- Wang et al., "Can Small Training Runs Reliably Guide Data Curation?", arXiv 2512.24503.
- Multi-fidelity data-mixture BO: arXiv 2503.21023; ADMIRE-BayesOpt arXiv 2508.11551.

**Batch / asynchronous / cost-aware BO**
- Desautels, Krause, Burdick, GP-BUCB, arXiv 1206.6402.
- Kandasamy et al., "Asynchronous Parallel Bayesian Optimisation via Thompson Sampling" (published as "Parallelised Bayesian Optimisation via Thompson Sampling", AISTATS 2018), arXiv 1705.09236.
- Balandat et al., BoTorch, arXiv 1910.06403; Letham et al., q-NEI, arXiv 1706.07094; Wu & Frazier, q-KG, arXiv 1606.04414.
- Ginsbourger, Le Riche, Carraro, "Kriging is well-suited to parallelize optimization" (2010).
- González et al., "Batch BO via Local Penalization", arXiv 1505.08052.
- Lee et al., CArBO, arXiv 2003.10870; Xie et al., "Cost-aware BO via the Pandora's Box Gittins Index", arXiv 2406.20062.
- Li & Scarlett, "GP Bandit Optimization with Few Batches", arXiv 2110.07788.

**Architectures**
- Beck et al., "xLSTM Scaling Laws", arXiv 2510.02228.
- Wang et al., "A Systematic Analysis of Hybrid Linear Attention", arXiv 2507.06457.
- Bae et al. (Meta), "Hybrid Architectures for Language Models: Systematic Analysis and Design Insights", arXiv 2510.04800.
- Poli et al., "Mechanistic Design and Scaling of Hybrid Architectures" (MAD), arXiv 2403.17844; STAR, arXiv 2411.17800.
- Waleffe et al., "An Empirical Study of Mamba-based Language Models", arXiv 2406.07887.
- Yang et al., Gated DeltaNet, arXiv 2412.06464; Kimi Linear, arXiv 2510.26692; Jet-Nemotron, arXiv 2508.15884.
- MiniMax, "Why did M2 end up as a full attention model?" (HF blog, 2025).
- Clark et al., "Unified Scaling Laws for Routed LMs", arXiv 2202.01169; Krajewski et al., "Scaling Laws for Fine-Grained MoE", arXiv 2402.07871;
  Ludziejewski et al., "Joint MoE Scaling Laws", arXiv 2502.05172; "Optimal Expert-Attention Allocation in MoE", arXiv 2603.10379.
- Cheng, Tian et al. (PKU + DeepSeek-AI), "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs" (Engram), arXiv 2601.07372.
- Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning" (Huginn), arXiv 2502.05171;
  Bae et al., Mixture-of-Recursions, arXiv 2507.10524; Relaxed Recursive Transformers, arXiv 2410.20672;
  Chen et al., "Parallel Scaling Law" (ParScale), arXiv 2505.10475.
- Sardana et al., "Beyond Chinchilla-Optimal", arXiv 2401.00448; Bian et al., "Scaling Inference-Efficient LMs", arXiv 2501.18107.
