# Speed-scaling results — JUPITER, FP8

All 64 arms measured 2026-08-17/18. Every arm converged: `drift` (median vs
last-quartile median) is within ±0.7% throughout, so no arm is flagged by the
convergence guard in `scripts/korbi/scaling_plots.py`.

**Common configuration** unless stated per-arm: FP8 hybrid + `delayed` recipe,
`fp8_param_gather: false`, amax history 1024 / `max`, **fp32 gradient
accumulation** (`grad_reduce_in_bf16: false`), seq 4096, `save: null`,
21–25 excluded nodes. This is the *production* precision setting, not a
speed-test ceiling — see "FP8 vs bf16" below.

Regenerating is now **two steps**, not one. `scaling_plots.py` no longer parses
logs — re-reading 200+ multi-MB files over GPFS took minutes per figure. First
`gather_speed.py` harvests every run once into a CSV, then the plotter reads
only that. Step 1 must run **on JUPITER** (the logs are there); step 2 runs
anywhere.

```bash
# 1. on JUPITER — harvest. Note BOTH trees: the speed sweeps live in
#    pre_production_training, the flagship writes to production_training.
python3 scripts/korbi/gather_speed.py \
  --root /e/project1/e-sta-openeurollm/pre_production_training \
  --root /e/scratch/e-sta-openeurollm/pre_production_training \
  --root /e/project1/e-sta-openeurollm/production_training \
  --skip-first 2 --skip-frac 0.3 --out ~/speed_gather/all_runs_speed.csv
# copy it back, then

# 2. anywhere — plot.
python3 scripts/korbi/speed_scaling_plots.py --from-csv dump/all_runs_speed_20260830.csv \
  --series 'strong scaling (32B, GBS 4096)=speedscale_fp8strong-n(\d+)_' \
  --filter fp8=hybrid --paper --split --peak-tflops 989.4 --annotate-m \
  --mark 'sustained production (512n)=512:399.7' \
  --out dump/final_20260830/fig_fp8_strong.png --csv dump/final_20260830/fig_fp8_strong.csv
```

Warmup trimming happens at **gather** time, so `--skip-first/--skip-frac` have
no effect in step 2; re-run step 1 to change them. Harvest of 2026-08-30:
311 rows over 4 roots, node counts 1→1024.

Data files (CSV carries TP/PP/DP/GBS/mbs/M, p10/median/p90/plateau/drift,
aggregate PFLOP/s, efficiency). The 2026-08-30 regeneration reproduces every
2026-08-18 figure to within 0.1 TFLOP/s, so the two sets are interchangeable:

| file (under `dump/final_20260830/`) | contents |
|---|---|
| `fig_scaling_regimes.csv` | **the headline figure** — strong / weak-batch / weak-model side by side |
| `fig_overview.csv` | all five families, per-GPU + aggregate + efficiency |
| `fig_fp8_strong.csv` | 32B strong, 8→1024 nodes |
| `fig_fp8_weak.csv` | 32B weak-batch + weak-model |
| `fig_small_strong.csv` | 0.4B / 1.7B / 7B strong |
| `fig_small_weakbatch.csv` | 0.4B / 1.7B / 7B weak-batch |
| `fig_fp8_vs_bf16_strong.csv` | **new** — 32B strong in BOTH precisions |
| `fig_weakmodel_ladder.csv` | **new** — weak-model, 1→1024 nodes, bf16 + FP8 |
| `fig_prod_sustained.csv` | per-iteration series of the live 512-node flagship |

## Which config ends up in which figure

The configs in this directory declare **106 arms**; 101 produced usable logs. Before
2026-08-30 the figures covered only 61 of the 98 harvested `speedscale_` runs — the whole
bf16 half of the study was unplotted, which is what the "pull it across if the write-up needs
the delta" note below was pointing at. Current mapping:

| config | arms | figure |
|---|---:|---|
| `fp8_strong_scaling.yaml` | 8 | `fig_fp8_strong`, `fig_overview`, `fig_fp8_vs_bf16_strong` |
| `strong_scaling.yaml` (bf16) | 8 | `fig_fp8_vs_bf16_strong` |
| `fp8_weak_batch.yaml` | 8 | `fig_fp8_weak` |
| `fp8_weak_model.yaml` | 5 | `fig_fp8_weak`, `fig_overview`, `fig_weakmodel_ladder` |
| `weak_scaling_model.yaml` (bf16) | 10 | `fig_weakmodel_ladder` (both the DP=4 and DP=256 ladders) |
| `control_weakmodel_n64.yaml` | 1 | `fig_weakmodel_ladder` — supersedes the broken n64 arm |
| `fp8_small_strong.yaml` | 19 | `fig_small_strong`, `fig_overview` |
| `fp8_small_weakbatch.yaml` | 21 | `fig_small_weakbatch` |
| `fp8_small_1b7_fix.yaml` | 3 | `fig_small_strong` / `fig_small_weakbatch` (see the asterisks below) |
| `weak_scaling_batch.yaml` (bf16) | 9 | **deliberately not plotted** |
| `weak_model_parallelism.yaml`, `weak_model_final.yaml` | 8 | **not a scaling curve** |
| `fp8_small_weakmodel.yaml` | 3 | **too sparse to plot** |
| `fp8_small_datacheck.yaml` | 2 | diagnostic, no throughput result |

Three deliberate omissions, so nobody re-derives them:

* **`weak_scaling_batch.yaml` (bf16 weak-batch).** Three of its eight arms — 128, 256 and
  512 nodes — fail the convergence guard, degrading by −5.7 / −10.4 / −6.7% between the
  whole-run median and the last-quartile plateau. That is the sick-node/FS-stall signature,
  so those medians are biased HIGH and the curve would overstate the machine. The FP8
  weak-batch sweep re-ran the same design cleanly; use `fig_fp8_weak`.
* **`weak_model_parallelism.yaml` / `weak_model_final.yaml`.** These pick the best (TP,PP)
  per model size — several arms share a node count and differ only in parallelism. Plotted on
  a node axis they draw two points at the same x, which is not a scaling curve. The selection
  table lives in the header comment of `fp8_weak_model.yaml`; leave it a table.
* **`fp8_small_weakmodel.yaml`.** Three arms (32/128/512 nodes), so the reference arm alone
  sets the shape and efficiency runs above 1.0. Numbers are in the CSV; a 3-point curve is
  not worth a figure.

**Matching a `-rerun` arm needs care.** `control_weakmodel_n64.yaml` re-ran the degraded
64-node arm, so the harvest holds `weakmodel-1b9-n64` (35 iters, drift −43.9%) *and*
`weakmodel-1b9-n64-rerun` (210 iters, drift +0.1%). A `--series` regex ending in `-n(\d+)_`
matches only the broken one, because the rerun has `-n64-rerun_`. Drop the trailing
underscore — `-n(\d+)` — and the plotter's "most iterations wins" rule then picks the rerun
by itself.

MFU is **not** stored — it is computed at plot time from `--peak-tflops 989.4`
(GH200 **BF16 dense**). Quote the peak alongside any MFU number. These runs are
FP8, whose peak is 2x that, so the same 400 TFLOP/s reads as **40.4% of the BF16
peak** or 20.2% of the FP8 peak. The BF16 denominator is the one that compares
against published BF16 baselines; say which you mean.

---

## The three regimes disagree — say which one you are quoting

`scripts/korbi/scaling_curves.py` draws all three on one shared y-axis
(`fig_scaling_regimes`). For the **same 32B model at 512 nodes**:

| regime | what is held fixed | TFLOP/s | efficiency |
|---|---|---:|---:|
| strong | model + GBS fixed | **417.9** | 0.744 |
| weak, model axis | model ∝ nodes, GBS ∝ nodes | 359.8 | 0.889 |
| weak, batch axis | model fixed, GBS ∝ nodes | 329.1 | 0.787 |

**A 27% spread with no configuration change** — it is entirely which question was
asked. Strong scaling is the honest worst case (work per GPU falls as nodes rise,
so a declining curve is correct, not a defect). Weak-model is the regime a
compute-budget estimate actually lives in. Never mix them in one sentence.

Regenerate with:

```bash
python3 scripts/korbi/scaling_curves.py --from-csv dump/all_runs_speed_20260830.csv \
  --peak-tflops 989.4 --out dump/final_20260830/fig_scaling_regimes.png \
  --csv dump/final_20260830/fig_scaling_regimes.csv
```

Colour is the model family and is fixed per family across the strong and
weak-batch columns; the palette is the validated categorical order and was
checked with the data-viz palette validator (worst adjacent CVD ΔE 9.1,
normal-vision 22.9; aqua and yellow fall below 3:1 on a light surface, so every
series is endpoint-labelled and the companion CSV is the table view).

---

## How much of this is noise? (measured 2026-08-30)

Two different variances get called "the error bar", and only the smaller one is
drawn on the figures.

| term | what it is | measured |
|---|---|---|
| within-run | p10–p90 of TFLOP/s across iterations of ONE job — the whiskers | **median 0.67%** of the rate; <1% for 44 of 58 arms |
| run-to-run | same config, executed again | **0.81% CV** across 21 repeated configs |

The cleanest single estimate of the second is the 512-node production campaign:
16 healthy jobs, byte-identical config, **CV 0.92%**, full range 4.4%.

**Both are ~10x smaller than any difference the scaling figures show**, which is
why the whiskers are invisible on most arms. That is a result, not an omission.

**The oft-quoted "run-to-run spread is 2.7%" is a single pair.** It comes from
jobs 1365560 vs 1365861 (320.5 vs 312.2) in `SPEED_RESULTS.md`, and two samples
estimate a spread badly. With 21 repeated configs the typical CV is ~0.8%. Keep
"treat <3% as noise" as the decision rule anyway — the 4.4% production range
shows a single slow node does occasionally push one job that far — but do not
believe that 2.7% is the *typical* spread, and do not use it to dismiss a
consistent 2% effect seen across several arms.

**A long whisker is a stall, not uncertainty.** `fp8small-strong-1b7-n128` has
p10 326.3 against a median of 480.1 because a few iterations dropped to
**39.9 TFLOP/s** — a 12x stall, not a 30% measurement band. Its plateau is 481.2
and its drift +0.23%, that is the bulk rate is solid. Read the plateau/drift columns
before reading a whisker.

**Estimator quirk: below 20 iterations, p10 IS the minimum.** `gather_speed.py`
computes `p10 = s[max(0, int(n*0.10) - 1)]`, so for n ≤ 19 the index is 0. Arms
like `fp8small-strong-7b-n64` (n=14) therefore show a whisker whose lower end is
the single worst iteration, which is not a percentile. Affects only the short
shakeout arms; every 70-iteration arm is fine.

---

## Headline numbers

| quantity | value | where |
|---|---|---|
| peak per-GPU throughput | **561.8 TFLOP/s** (56.8% MFU) | 32B, 8 nodes, M=1024 |
| **32B @ 512 nodes, GBS 4096** | **417.9 TFLOP/s, 855.9 PFLOP/s** | production geometry, shakeout |
| **32B @ 512 nodes — SUSTAINED** | **399.7 TFLOP/s (40.4% MFU), 818.6 PFLOP/s** | the live flagship, 75k iterations |
| 32B @ 1024 nodes, GBS 4096 | 322.9 TFLOP/s, 1.32 EFLOP/s | M=8 |
| largest aggregate measured | **1.73 EFLOP/s** | 32B weak-model, 1024 nodes, GBS 16384 |
| best weak scaling | **0.900 over 64x nodes** | 7B weak-batch, 8→512 |
| largest scale run | **1024 nodes = 4096 GH200** | ~19% of JUPITER |

The sustained row is the one to quote for a schedule; everything else in this
document is a 50-iteration shakeout that never pays for a checkpoint, an eval or
a restart. The two differ by only **4.4%**, which is the real result — see
"Sustained production" below.

---

## Sustained production — the 512-node flagship, 2026-08-24 → 08-30

Everything above is a benchmark. This is the same configuration left running.
Campaign `oellm_32b_dense_prod_dataopt5_gbs4096_lr3e-4`, 512 nodes / 2048 GH200,
FP8 hybrid + delayed with fp32 grad accumulation, z-loss 1e-4, PP=4/VPP=4,
ft_launcher with hot spares — that is every production cost included.

| quantity | value |
|---|---|
| iterations | 5 → **75,125** of 894,000 (8.4% of the 15 T schedule) |
| tokens delivered | **1.260 T** in 6.1 calendar days |
| TFLOP/s/GPU | p10 375.9 \| **median 399.7** \| p90 404.6 → **40.4% MFU** |
| aggregate while running | **818.6 PFLOP/s** |
| steady-state training time | 90.0 h |
| SLURM allocated time | 103.2 h (18 allocations) → **87.2% in-allocation efficiency** |
| restarts | 14 job handovers, **median 1 min** between allocations |
| allocation cost | 211,425 GPU-h |

Two readings, and they answer different questions:

* **87.2% of allocated time is spent iterating.** The missing 12.8% is startup
  (8–15 min of NCCL sync and kernel compile on every 512-node allocation),
  checkpoint saves, eval, and teardown. This is the number that describes the
  *system*.
* **61.3% of calendar time.** That charges the run for two gaps it did not
  cause: a **21.0 h queue wait** and a **17.4 h stretch with nothing submitted**.
  Excluding those two, end-to-end is **82.7%**. The other 15 handovers are a
  median of one minute apiece, so chained resubmission is not a cost centre.

Reproduce with `scripts/korbi/prod_sustained_plot.py` (figure
`dump/final_20260830/fig_prod_sustained.png`); it needs the grepped iteration
lines plus an `sacct` dump — see that script's docstring.

**Watch the log_interval when recomputing this.** Megatron reports the *mean*
ms/iter since the previous log line, and production runs `log_interval: 5`.
Summing the raw per-record ms undercounts steady-state time 5x — 18.1 h instead
of 90.0 h — which turns 87% efficiency into 17%. Weight every record by its
iteration gap.

The earlier campaign `oellm_32b_dense_prod_gbs4096_lr3e-4` (2026-08-23/24,
7 jobs, iterations 1→16,394) ran the same geometry at a median of 400.0–403.5
TFLOP/s before being superseded by the option-5 data mix. It corroborates the
level; it is not a second independent measurement of it.

---

## 32B strong scaling (GBS 4096 fixed, TP=4 / PP=4 / VPP=2)

| nodes | M | TFLOP/s | PFLOP/s | efficiency |
|---:|---:|---:|---:|---:|
| 8 | 1024 | 561.8 | 18.0 | 1.000 |
| 16 | 512 | 545.1 | 34.9 | 0.970 |
| 32 | 256 | 544.7 | 69.7 | 0.970 |
| 64 | 128 | 532.8 | 136.4 | 0.948 |
| 128 | 64 | 519.1 | 265.8 | 0.924 |
| 256 | 32 | 488.8 | 500.5 | 0.870 |
| **512** | **16** | **417.9** | **855.9** | **0.744** |
| 1024 | 8 | 322.9 | 1322.6 | 0.575 |

Efficiency holds >0.92 to 128 nodes and falls away as M shrinks — this curve is
the M curve in disguise, not a node-count effect.

## 32B weak scaling

**Batch axis** (GBS ∝ nodes, M=8 fixed): 418.4 → 314.0 TFLOP/s over 8→1024
nodes, efficiency 1.000 → **0.750**, 1.29 EFLOP/s at the top.

**Model axis** (model size ∝ nodes, parallelism tuned per size):

| nodes | model | TP/PP | M | TFLOP/s | PFLOP/s | eff |
|---:|---|---|---:|---:|---:|---:|
| 64 | 1.9B | 1/1 | 2 | 404.5 | 103.6 | 1.000 |
| 128 | 3.5B | 2/1 | 4 | 327.4 | 167.6 | 0.809 |
| 256 | 7B | 2/2 | 8 | 375.5 | 384.5 | 0.928 |
| 512 | 17B | 4/2 | 16 | 359.8 | 736.9 | 0.889 |
| 1024 | 32B | 4/4 | 32 | **421.2** | **1725.2** | 1.041 |

Efficiency >1.0 at 1024 nodes is an artefact of the reference: the series is
anchored at 64 nodes where TP=PP=1 and M=2, a weak operating point. It is not
superlinear scaling. Parallelism here is **tuned per size** — that is the correct
methodology for a compute-budget estimate, and the winning (TP,PP) is listed
above so the caption can state it.

**The ladder actually runs from 1 node** (`fig_weakmodel_ladder`).
`weak_scaling_model.yaml` declares the same 1.9B → 32B model ladder twice, at two
data-parallel widths, and both are in the harvest:

| ladder | nodes | DP | 1.9B | 3.5B | 7B | 17B | 32B |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16, small | 1→16 | 4 | 411.5 | 360.0 | 333.5 | 353.5 | 318.1 |
| bf16 | 64→1024 | 256 | 374.0 | 325.0 | 272.9 | 324.2 | 337.7 |
| FP8 | 64→1024 | 256 | 404.5 | 327.4 | 375.5 | 359.8 | 421.2 |

They are **not one continuous curve** and must not be drawn as one: at 16 nodes
the model is already 32B, and at 64 nodes it restarts at 1.9B. What the pair
does show is that the same model ladder costs about the same per GPU at DP=4 and
at DP=256 — a 64x wider world — which is the weak-scaling claim worth making.

The bf16 dip at 256 nodes (272.9) is the **mis-parallelised 7B arm**: it ran
TP4/PP1, and `weak_model_parallelism.yaml` later measured TP2/PP2 at 321.4, that is
+17.8%. The FP8 ladder was built on those tuned choices, which is why it has no
dip. The comparison is bf16-with-a-known-bad-arm against FP8-tuned, so read the
1.9B/3.5B/17B/32B columns for the precision delta and ignore the 7B one.

## Small families, strong scaling (GBS 16384 fixed, TP/PP per family)

| nodes | 0.4B | 1.7B | 7B |
|---:|---:|---:|---:|
| 8 | 411.0 | **515.1** | – |
| 16 | 402.2 | 496.9 | – |
| 32 | 398.7 | 499.7 | 483.8 |
| 64 | 393.5 | 454.6 | 478.2 |
| 128 | 380.5 | 480.1 | 457.6 |
| 256 | 341.1 | 446.0 | 441.4 |
| 512 | 272.2 | 285.3 * | **405.2** |
| **eff @512** | 0.662 | 0.554 * | **0.838** |

## Small families, weak-batch (GBS = 32 × nodes)

| nodes | 0.4B | 1.7B | 7B |
|---:|---:|---:|---:|
| 8 | 368.2 | 454.2 | 455.8 |
| 128 | 337.2 | 426.6 | 433.3 |
| 256 | 302.2 | 308.2 * | 424.4 |
| 512 | 269.8 | 285.3 * | **410.3** |
| **eff @512** | 0.733 | 0.628 * | **0.900** |

**Larger models scale better.** 7B holds 0.900 across a 64× node range while
0.4B falls to 0.733 — the small models run out of work per GPU (0.4B at 512
nodes is M=1, a single micro-batch per rank).

---

## `*` — the 1.7B asterisks. READ BEFORE CITING.

Three arms run **TP=2** while the rest of the 1.7B family runs TP=1:
`strong-1b7-n512`, `weakbs-1b7-n512`, `weakbs-1b7-n256` (see
`fp8_small_1b7_fix.yaml`). They are **not like-for-like** with their own series:

* the weak-batch design holds M constant at 2; these run M=4;
* the efficiency drop at those points (0.939 → 0.679 in weak-batch) is
  substantially the parallelism change, **not** a scaling result.

Caption them as *"TP=2, required to run at this scale"*. Do not fold them into
a fitted scaling curve without saying so. **`fig_overview` therefore stops the
1.7B series at 256 nodes** (its `--series` regex enumerates 8–256 rather than
using `\d+`), so the headline figure carries only like-for-like arms; the
per-family figures still show all of them.

They carry TP=2 because **they will not run at TP=1**: at M=2 with DP ≥ 1024 the
job hangs forever at the dataset barrier, always at exactly 2078 index loads,
immediately after the BlendedDataset-*train* index and before the *valid* one.
Ruled out by direct measurement, not assumption:

| hypothesis | disproved by |
|---|---|
| cluster / 512 nodes | `strong-7b-n512` completed 100/100 at 512 nodes |
| dataset or index | successful arms request identical splits `(3661824000, 3768320, 163840)` and load the same valid-index hash `b516b2a59069`; `BUILD=0` everywhere |
| node faults | four stalled allocations share **zero** nodes |
| file system contention | a stalled arm ran alone, loading 2078 indices in ~50 s |
| premature cancellation | only declared stalled after >900 s of silence |

What correlates:

| M | DP | outcome |
|---:|---:|---|
| 2 | ≤512 | works |
| ≥4 | ≤1024 | works |
| 1 | 2048 | works (0.4B @512) |
| 8 | 1024 | works (7B @512) |
| **2** | **≥1024** | **hangs, 3/3 retries** |

TP=2 halves DP and doubles M onto a proven shape. **The mechanism is not
understood — this is a workaround, not a diagnosis.** If M=2 at wide DP matters
for the flagship, reproduce it with `NCCL_DEBUG=INFO` to see which rank never
reaches the barrier.

---

## FP8 vs bf16

At identical geometry (512 nodes, GBS 4096, TP=4/PP=4, 32B):

| precision | TFLOP/s |
|---|---:|
| FP8 hybrid + fp32 grad accum | **417.9** |
| bf16 | 336.1 |

**+24%**, with fp32 gradient accumulation *included* on the FP8 side, so this is
a sustained production number. The earlier 437–438 TFLOP/s speed-test figures
ran **without** fp32 grad accumulation (measured cost: 7.8%).

**The full curve is now plotted** — `fig_fp8_vs_bf16_strong`, both precisions at
all eight scales, from `strong_scaling.yaml` (bf16) and `fp8_strong_scaling.yaml`:

| nodes | bf16 | FP8 | FP8 gain |
|---:|---:|---:|---:|
| 8 | 420.8 | 561.8 | +33.5% |
| 16 | 410.5 | 545.1 | +32.8% |
| 32 | 409.2 | 544.6 | +33.1% |
| 64 | 393.5 | 532.8 | +35.4% |
| 128 | 392.2 | 518.9 | +32.3% |
| 256 | 377.0 | 488.8 | +29.7% |
| **512** | **336.1** | **417.9** | **+24.3%** |
| 1024 | 275.4 | 322.9 | +17.2% |

**The FP8 advantage SHRINKS with scale — that is the load-bearing observation.**
A flat ~33% from 8 to 128 nodes, then 29.7 → 24.3 → 17.2%. FP8 speeds up
*compute*; it does nothing for the DP collectives or the pipeline bubble, and
those are a growing share of the iteration as M falls from 1024 to 8. So the
gain decays exactly where the strong-scaling efficiency does. Do not quote "FP8
is worth 30%" for a 1024-node plan — at that scale it is worth 17%.

Note `fp8_param_gather` buys nothing: 437.8 (false) vs 437.0 (true) at matched
geometry, within noise. It stays **off** because it caused a reproducible CUDA
illegal memory access in the async checkpoint worker.

---

## Gap to the production stability runs: z-loss + ft_launcher ≈ 6%

The `fp8strong-n512` arm (417.9) and the 32B production warmup runs at the
**identical geometry** (512 nodes, GBS 4096, TP=4/PP=4/VPP=2, mbs=2, FP8
delayed + fp32 grad accum) do not agree:

| run | job | TFLOP/s | ms/iter | mem |
|---|---|---:|---:|---:|
| `speedscale_fp8strong-n512_gbs4096` | 1397791 | **417.9** | 4083 | 0.622 |
| `sc-...-n512-wu2000` (lr 3e-4) | 1401329 | 392.4 | 4346 | 0.680 |
| `sc-...-n512-wu3000` | 1401313 | 388.2 | 4392 | 0.681 |
| `sc-...-n512-wu4000` | 1400372 | 392.6 | 4343 | 0.779 |

**−6.1%.** Diffing the two argument dumps, they differ in exactly **four**
settings — everything else, model shape included, is identical:

| setting | speedscale | production |
|---|---|---|
| `output_z_loss_coeff` | None | 1e-4 |
| `enable_ft_package` | False | True |
| `calc_ft_timeouts` | False | True |
| `wgrad_deferral_limit` | 32 | 0 |

* **z-loss: −2.2%**, measured single-variable on the 16-node arms built for it
  (`sc-base` 339.0 → `sc-zloss1e-4` 331.6 TFLOP/s; `sc-zloss1e-3` 328.9). It adds
  a logsumexp over the 262 144-entry vocab plus its gradient, every step.
* **ft_launcher: ~−3.9%**, by elimination — the only remaining difference.
  Inferred, not measured, and only ~1.4× the 2.7% run-to-run noise floor, so
  treat it as "probably real, worth one test" rather than established. To pin it
  down: one 50-iteration speed test with `enable_ft_package: false` on the
  production recipe (~1400 GPU-h, no numerics implications).
* **`wgrad_deferral_limit: 32` is a no-op at this scale — do not "port" it.**
  It is a *memory* fix required below 256 nodes (`strong_scaling.yaml:89-113`):
  the default 0 means *defer all micro-batches*, so the last PP stage retains an
  `embedding_activation_buffer` + `grad_output_buffer` entry per micro-batch
  until the flush. At 512 nodes DP=128 → M=16, and 16 < 32, so the cap never
  binds.

Neither cost is a regression; both are deliberate purchases. z-loss buys logit
stability at raised LR, and the FT package buys node-failure survival — on
2026-08-17 four separate 1024-node allocations died to single-node faults, so
~4% against losing a 20 000 GPU-h run to one bad node is likely a good trade.

**Consequence for citation:** these speed-scaling numbers are the *ceiling for
the precision setting*, not what a production run sustains. Subtract ~6% when
projecting a flagship schedule from this table.

**Measured in full, 2026-08-30: the correction is 4.4%, not 6%.** The flagship
sustains 399.7 against this table's 417.9 at the same geometry — and it does so
with VPP=4, which the speed sweep did not have. So the ~6% estimate above was
right in sign and slightly pessimistic in size. Use 399.7 directly rather than
discounting 417.9.

**At 1024 nodes, do not discount 322.9 — it is measured separately.** The
PP×VPP campaign of 2026-08-22 re-ran that scale on the settings the flagship
actually ships and got **313.7** (PP=4/VPP=4, CUDA graphs off), with production
shape at 306.0. Table and mechanism in
`config/experiments/oellm_32b_dense/SPEED_RESULTS.md`, findings 5–7.

---

## Operational notes for anyone re-running these

* **≥256-node jobs are silent for 8+ minutes** during NCCL sync and kernel
  compilation. Do not cancel before 15 min — job 1400304 was killed at 7.4 min
  and was healthy. See memory `jupiter_startup_silence.md`.
* **PP=1 needs two settings cleared**, not one: `pipeline_model_parallel_layout:
  null` *and* `defer_embedding_wgrad_compute: false`. The parent chain sets an
  8-stage layout that Megatron reads as an interleaved schedule and asserts on;
  `num_layers_per_virtual_pipeline_stage: null` does **not** clear it.
* **7B does not fit at TP=1** with fp32 grad accumulation (140 OOMs at 32
  nodes). It uses TP=2 + `sequence_parallel` throughout — consistently, so its
  series *is* internally like-for-like.
* **Submit in tiers.** Launching 43 arms at once put ~2700 nodes into
  simultaneous index loading and lost four 512-node arms. Batches of ~900–1300
  nodes landed cleanly.
* Two nodes were excluded after being found in stalled allocations and never in
  healthy ones: `jpbo-075-09` (`/e/project1` not mounted — job logs live there,
  so its ranks block silently) and `jpbo-074-31` (IB link downed counters).
