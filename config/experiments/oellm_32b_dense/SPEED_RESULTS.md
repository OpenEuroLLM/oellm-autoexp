# OELLM 32B dense — speed results at 256 / 512 / 1024 nodes

Last updated 2026-08-17. Numbers measured on JUPITER (GH200, 4 GPUs/node) with
`nemo_26.04.sif`.

Common to every row unless stated: Qwen3-32B dense (64 layers, hidden 5120, 256k vocab,
untied embeddings), seq 4096, **GBS 4096**, TP=4, CP=1, sequence-parallel on,
`tp_comm_overlap`, distributed optimizer, **no recompute**. All are 50-iteration shakeouts
(`exit_interval: 50`, `save: null`) unless marked *sustained*. "Steady state" = median over
iterations 4..N, discarding warmup. `days@15T` = wall clock to train 15 T tokens at that step
time, ignoring restarts and checkpoints.

> **Read the caveats at the bottom before quoting any single number.** In particular the
> run-to-run spread at this scale is **2.7%**, so differences under ~3% are not real.

**VPP is derived, not chosen.** `arguments.py:549` sets
`VPP = num_layout_groups // PP`, where the groups are the `|`-separated entries of
`pipeline_model_parallel_layout`. The production layout has 8 groups, so PP=4 gives VPP=2 by
inheritance. A 16-group layout gives PP=4 → VPP=4.

---

## 1024 nodes (4096 GPUs)

| job | shape | precision / notes | TFLOP/s/GPU | Tok/s/GPU | ms/iter | mem | days@15T |
|---|---|---|---|---|---|---|---|
| **1395802** | **PP=4/VPP=4** | **fp8 delayed, 16-group layout** | **346.6** | **1665.2** | **2459.8** | 0.6063 | **25.5** |
| 1373121 | PP=4/VPP=2 | fp8 delayed + `fp8_param_gather` | 342.6 | 1645.7 | 2488.9 | 0.6248 | 25.8 |
| 1396484 | PP=4/VPP=2 | fp8 delayed (VPP=4 baseline) | 333.5 | 1602.2 | 2556.5 | 0.5518 | 26.5 |
| 1396982 | PP=4/VPP=2 | fp8 tensorwise, **fp32 grad** | 317.8 | 1526.4 | 2683.4 | 0.5327 | 27.8 |
| 1375942 | PP=4/VPP=2 | fp8 delayed + pg, **fp32 grad** | 315.6 | 1516.1 | 2701.7 | 0.6673 | 28.0 |
| 1368280 | PP=2/VPP=4 | bf16 | 274.5 | 1318.7 | 3106.1 | 0.7757 | 32.1 |
| 1367660 | PP=4/VPP=2 | bf16 | 273.1 | 1312.1 | 3121.8 | 0.5958 | 32.3 |
| 1368214 | PP=8/VPP=1 | bf16 | 211.8 | 1017.4 | 4025.9 | 0.3685 | 41.7 |

*Sustained* fp8 (VPP=2, no param-gather; long runs, so checkpoints and eval are inside the
median): 1380848 = 297.3 TF / 1428 tok/s / 2868 ms over 8000 iterations; 1379130 = 297.4 /
1429 / 2867 over 4204. Expect ~13% below the shakeout figure in production.

### 1024 nodes — PP x VPP campaign, 2026-08-22 (production settings)

Sessions 1787349257 / 1787354842, configs `speed_pp_vpp_n1024.yaml` and
`speed_pp_vpp_n1024_nocg.yaml`. **These are the first 1024-node numbers taken on the settings
the flagship actually ships**: fp32 gradient accumulation, z-loss 1e-4, fp8 hybrid + delayed.
Every row in the table above was taken with bf16 grad reduction and no z-loss, so *do not
compare across the two tables* — compare within this one, against `pp4-vpp2` (the production
config).

| job | shape | graphs | TFLOP/s/GPU | Tok/s/GPU | ms/iter | mem | vs prod |
|---|---|---|---|---|---|---|---|
| **1451776** | **PP=4/VPP=4** | **off** | **313.7** | **1507.1** | **2717.8** | **0.6491** | **+2.5%** |
| 1451774 | PP=4/VPP=4 | local | 311.4 | 1495.7 | 2738.5 | 0.8726 | +1.8% |
| 1451772 | PP=4/VPP=2 | local | 306.0 | 1470.1 | 2786.3 | 0.6990 | ref (production) |
| 1452155 | PP=8/VPP=4 | off | 300.3 | 1442.6 | 2839.3 | 0.6473 | −1.9% |
| 1452151 | PP=8/VPP=8 | off | 282.0 | 1354.9 | 3023.2 | 0.6933 | −7.8% |
| 1452153 | PP=16/VPP=4 | off | 232.3 | 1115.9 | 3670.6 | 0.8087 | −24.1% |
| 1452154 | PP=8/VPP=4 + `overlap_p2p_comm_warmup_flush` | off | 170.7 | 819.8 | 4996.6 | 0.6466 | −44.2% |
| 1451775 | PP=8/VPP=4 + `overlap_p2p_comm_warmup_flush` | local | 165.3 | 793.9 | 5159.6 | 0.9262 | −46.0% |

Did not produce a number: PP=4/VPP=8 and PP=8/VPP=8 and PP=16/VPP=4 OOM'd with graphs ON
(see finding 6); PP=8/VPP=4 (1451778) and PP=4/VPP=8 (1452152) each stalled in a *startup*
phase — userbuffers registration and fused-kernel compile respectively — and were cancelled by
the monitor. PP=4/VPP=8 is therefore still unmeasured after two attempts.
Campaign cost: 5,319 + 4,085 = **9,404 GPU-h** (the submit gate quoted 35,499; it assumes the
full 40 min wall, but `exit_interval: 50` exits each arm after ~8-10 min).

### Cross-check against the parallel `_jj` campaign (wandb `oellm_32b_dense_throughput_opt`)

A second 1024-node campaign ran the same week under
`/e/project1/e-sta-openeurollm/pre_production_training_jj/`. Its configs were recovered from the
rendered sbatch with `scripts/korbi/megatron_args_to_yaml.py` and diffed against ours: at
identical geometry the **only** throughput-relevant difference is `output_z_loss_coeff`
(unset there, `1e-4` here). TP, PP, layout, mbs, GBS, fp8 settings, graphs-off and fp32 grad
accumulation are byte-identical, so the two campaigns are comparable after one correction.

| arm | job | PP | VPP | z-loss | raw TFLOP/s | z-loss-adj | mem | it |
|---|---|---|---|---|---|---|---|---|
| `jj pp4-vpp4-do4` | 1453106 | 4 | 4 | off | 326.9 | 313.8 | 0.6084 | 8 |
| `jj pp4-vpp4-do1` | 1453099 | 4 | 4 | off | 326.8 | 313.7 | 0.5785 | 8 |
| `jj pp8-vpp2-do1` | 1454332 | 8 | 2 | off | 320.1 | 307.2 | **0.5149** | 8 |
| `pp4-vpp4-nocudagraph` | 1451776 | 4 | 4 | **on** | 313.7 | 313.7 | 0.6491 | 50 |
| `pp8-vpp4-nocg` | 1452155 | 8 | 4 | **on** | 300.3 | 300.3 | 0.6473 | 50 |

Adjustment factor is 1.042, taken from the one exactly-matched pair (1453099 vs 1451776). The
512-node ablation puts z-loss at 2.79% (`speed_ablation_n512.yaml`, jobs 1397791/1407138), so the
true cost is somewhere in 2.8-4.2% and the residual is run-to-run spread plus the fact that the
`_jj` arms use `exit_interval: 8` — a median over 5 samples against our 47. **Treat the adjusted
column as +-3%.** z-loss also costs ~7 memory points (0.5785 vs 0.6491 at identical geometry),
which is the 262k-vocab logit reduction.

Three things this settles that our own sweep could not:

1. **PP=8/VPP=2 is the memory-efficient near-tie, and we wrongly skipped it.** 307.2 adjusted vs
   313.7, that is −2% (inside noise) at **mem 0.5149 vs 0.6491**. It was dropped from
   `speed_pp_vpp_n1024.yaml` on the modelled bubble (21.9%, the worst of any arm) — but finding 5
   shows the bubble is not what decides this, so the reasoning that excluded it was wrong.
2. **At PP=8 the VPP optimum is 2, not 4** (307.2 vs 300.3), the opposite of PP=4 (VPP=4 beats
   VPP=2). Consistent with p2p dominance: p2p scales as `0.084 GB * PP * VPP`, so the deeper
   interleave costs more at PP=8 than the bubble it buys back.
3. **HSDP does nothing at 1024 nodes.** `num_distributed_optimizer_instances` 4 vs 1 is
   326.9 vs 326.8 (jobs 1453106/1453099, verified single-variable). Same null result as at 512
   nodes; it can come off the follow-up list.

Best-known 1024-node configuration therefore stays **PP=4 / VPP=4 / graphs off**, with
**PP=8 / VPP=2 / graphs off** as the choice if memory headroom matters more than ~2%.

## 512 nodes (2048 GPUs)

| job | shape | precision / notes | TFLOP/s/GPU | Tok/s/GPU | ms/iter | mem | days@15T |
|---|---|---|---|---|---|---|---|
| 1365865 | PP=4/VPP=2 | fp8 delayed, cudagraph, HSDP4 | 437.6 | 2102.0 | 3897.3 | 0.7134 | 40.3 |
| 1366812 | PP=4/VPP=2 | fp8 delayed + `fp8_param_gather` | 436.9 | 2098.5 | 3903.7 | 0.6345 | 40.4 |
| 1366543 | PP=4/VPP=2 | fp8 tensorwise + first/last bf16 | 417.3 | 2004.4 | 4087.0 | 0.5797 | 42.3 |
| 1396514 | PP=4/VPP=2 | fp8 tensorwise, **fp32 grad** | 410.7 | 1972.8 | 4152.5 | 0.6236 | 42.9 |
| 1365955 | PP=4/VPP=2 | bf16 + cudagraph/defer-wgrad | 321.7 | 1545.2 | 5301.5 | 0.6873 | 54.9 |
| 1365560 | PP=4/VPP=2 | bf16 + HSDP4/pad-buckets | 320.5 | 1539.4 | 5321.5 | 0.6941 | 55.1 |
| 1364573 | PP=4/VPP=2 | bf16 baseline | 316.7 | 1521.2 | 5385.2 | 0.3891 | 55.7 |
| 1365076 | PP=4/VPP=2 | bf16 baseline (repeat) | 316.5 | 1520.4 | 5388.0 | 0.3892 | 55.8 |
| 1365861 | PP=4/VPP=2 | bf16 (repeat of 1365560) | 312.2 | 1499.5 | 5463.0 | 0.6940 | 56.5 |
| 1364776 | PP=2/VPP=4 | bf16 | 308.3 | 1481.0 | 5531.4 | 0.5450 | 57.2 |
| 1365957 | PP=4/VPP=2 | bf16, fp32 grad accum | 305.6 | 1468.2 | 5579.5 | 0.8150 | 57.7 |
| 1364775 | PP=1/VPP=1 | bf16 | 273.7 | 1314.6 | 6231.6 | 0.7473 | 64.5 |
| 1364788 | PP=8/VPP=1 | bf16 | 257.9 | 1238.8 | 6612.6 | 0.1790 | 68.4 |
| 1364790 | PP=8/VPP=1 | bf16, mbs=4 | 236.7 | 1137.1 | 7204.3 | 0.2478 | 74.6 |

## 256 nodes (1024 GPUs)

Job 1388292 (fp8 + batch-size ramp), post-ramp tail at GBS 4096, last 500 of 5697 iterations:
**440.0 TFLOP/s/GPU**, 2814 tok/s/GPU, 5822 ms/iter, 60.2 days@15T.

---

## Findings

### 1. PP=4 is the optimum — "more PP" is not better

| nodes (bf16) | PP=1 | PP=2 | PP=4 | PP=8 |
|---|---|---|---|---|
| 512 | 273.7 | 308.3 | **316.7** | 257.9 |
| 1024 | — | 274.5 | 273.1 | 211.8 |

Both extremes lose, for opposite reasons:

- **PP too low** → params/GPU rises, and the distributed optimizer moves `2 × params/GPU` per
  step (reduce-scatter + all-gather): ~16 GB/GPU/iter at PP=2 vs ~8 GB at PP=4. PP=2's smaller
  pipeline bubble is almost exactly cancelled by this, which is why PP=2 and PP=4 tie at 1024.
- **PP too high** → the bubble explodes. With the fixed 8-group layout, PP=8 collapses VPP to 1,
  giving bubble = `(PP−1)/(VPP·M)` = 7/16 = 43.8%.

### 2. VPP=4 buys +3.9% at 1024 nodes, and costs ~10% memory

Same-session A/B, identical config, only `pipeline_model_parallel_layout` differs:

| arm | layout | VPP | ms/iter (min–max) | TFLOP/s | Tok/s/GPU | mem |
|---|---|---|---|---|---|---|
| 1396484 | 8 groups | 2 | 2556.5 (2538.1–2583.2) | 333.5 | 1602.2 | 0.5518 |
| 1395802 | 16 groups | 4 | 2459.8 (2442.8–2528.9) | 346.6 | 1665.2 | 0.6063 |

**+3.9%** — above the 2.7% noise floor, but in the "real but marginal" band. Worth ~1 day on a
26-day 15 T run at 1024 nodes.

Two things to know before reusing this:

- **It is 1024-specific.** The gain comes from the bubble, `(PP−1)/(VPP·M)` with
  `M = GBS/(mbs·DP)`. At 1024 nodes M=8, so VPP=2's bubble is 18.8%. At 512 nodes M=16 and it is
  already 9.4%, so the modelled VPP=4 gain there is only ~+2.5% — probably under the noise floor.
  **Do not port this to 512 without measuring.**
- **It costs memory: +5.5 points, ~+10% relative.** An earlier prediction from Megatron source
  (that in-flight microbatches rise 10→18 while layers-per-chunk fall 8→4, so peak activation
  memory would be flat) was **wrong**. The activation-only model misses that higher VPP also
  multiplies model chunks per rank, hence per-chunk gradient and p2p buffers.

**VPP=8 (32 groups) is not worth testing.** Measured realisation of the modelled bubble gain was
only ~34% (predicted +11.5%, got +3.9%) because part of the nominal bubble already overlaps DP
comm. VPP=8 has half as much bubble left to recover (4.7 points), so ~+1.8% gross, while p2p
doubles again (~2 points) — net ≈ 0, and below the noise floor, that is not measurable at
acceptable cost.

### 3. FP32 gradient accumulation is free at 512 nodes, costs 7.9% at 1024

Switch is `grad_reduce_in_bf16: false`; Megatron then auto-enables
`accumulate_allreduce_grads_in_fp32` (`arguments.py:742-745`) — **but only if
`main_grads_dtype == fp32`**. That holds by default and by way of `base_defaults.yaml:295`; setting
`main_grads_dtype: bf16` would silently disable *both* paths. Neither flag appears on the
command line (`--grad-reduce-in-bf16` is store-true; the fp32 flag is set inside Megatron), so
the argument dump in the log is the only way to confirm what ran.

| scale | grad | job | recipe | TFLOP/s | ms/iter | mem | delta |
|---|---|---|---|---|---|---|---|
| 512 | bf16 | 1366543 | tensorwise | 417.3 | 4087.0 | 0.5797 | ref |
| 512 | **fp32** | 1396514 | tensorwise | 410.7 | 4152.5 | 0.6236 | **−1.6% (≤ noise)** |
| 1024 | bf16 | 1373121 | delayed + param_gather | 342.6 | 2488.9 | 0.6248 | ref |
| 1024 | **fp32** | 1375942 | delayed + param_gather | 315.6 | 2701.7 | 0.6673 | **−7.9%** |
| 1024 | **fp32** | 1396982 | tensorwise | 317.8 | 2683.4 | 0.5327 | (cross-recipe check) |

Compare **within** rows only — the 512 pair is the tensorwise branch, the 1024 pair is
delayed + `fp8_param_gather`. Grad precision is the sole variable inside each pair. Job 1396982
corroborates the fp32 *level* at 1024 on a different recipe (within 0.7% of 1375942); it is not a
second measurement of the delta.

Mechanism: fp32 doubles the DP reduce-scatter/all-gather bytes, and that comm is roughly constant
per iteration while compute per iteration halves as node count doubles — a small slice of a
4.15 s step at 512, a large slice of a 2.68 s step at 1024. Memory cost is consistent: +7.6%
relative at 512, +6.8% at 1024.

**Practical:** if fp32 grad accumulation is wanted for stability, 512 nodes gives it for free.

### 5. PP=4 is the optimum, and raising PP is monotonically WORSE (2026-08-22)

The 2026-08-17 reading of finding 1 was that PP=8 lost because the 8-group layout collapsed VPP
to 1, and that PP=8 with a real interleave would win by halving the distributed-optimizer
traffic. **That was measured and it is false.** At matched VPP=4 and graphs off:

| PP | DP | M | params/GPU | DP-opt GB/GPU/iter | p2p GB/GPU/iter | TFLOP/s |
|---|---|---|---|---|---|---|
| 4 | 256 | 8 | 2.05 B | 12.3 | 1.34 | **313.7** |
| 8 | 128 | 16 | 1.03 B | 6.2 | 2.68 | 300.3 |
| 16 | 64 | 32 | 0.51 B | 3.1 | 5.37 | 232.3 |

The optimizer traffic really does halve and quarter, and it does not help: throughput falls
monotonically. PP=16 has the *least* DP traffic of any shape ever run here and is the worst
geometry measured at 1024 nodes. So the premise that DP comm dominates the iteration at this
scale is wrong — pipeline depth and p2p do. Note PP=16 also has the *highest* memory (0.8087)
despite a quarter of the parameters, which is the p2p buffers and 64 model chunks.

Corollary for the arithmetic in `speed_test_vpp1024.yaml`: at 1024 nodes with TP=4/mbs=2/GBS=4096
the microbatch count is tied to PP (`M = 2*PP`), so the bubble `(PP-1)/(VPP*M)` reduces to
`~1/(2*VPP)` and is essentially PP-independent. PP therefore has no bubble lever at all here —
only the DP/p2p trade, which it loses.

**VPP=4 is the sweet spot.** VPP 2 -> 4 buys +1.8% (306.0 -> 311.4, both with graphs), which is
inside the 2.7% noise floor and well below the +3.9% seen in bf16 without z-loss. VPP 4 -> 8
costs −7.8% at PP=8. Deeper interleaving has stopped paying by 4.

### 6. CUDA graphs are a pure loss at 1024 nodes — and they cost four arms

Jobs 1451774 / 1451776 differ only in `cuda_graph_impl`:

| graphs | TFLOP/s | mem |
|---|---|---|
| `local` | 311.4 | 0.8726 |
| `none` | 313.7 | 0.6491 |

**0.7% slower and 22.4 memory points more expensive.** The memory is what matters: three arms
OOM'd outright, reporting how much of the card sat in graph private pools —
PP=4/VPP=8 25.95 GiB, PP=8/VPP=8 36.53 GiB, PP=16/VPP=4 54.84 GiB.

The graph *count* is constant across all of these — `layers_per_rank * M * 2 = (64/PP) * (2*PP) * 2
= 256` regardless of PP or VPP. What grows is the number of POOLS: `cuda_graphs.py:1175` indexes
`fwd_mempools[vpp_rank]`, that is one forward mempool per virtual stage, so nothing is reused across
chunks. `cuda_graph_use_single_mempool` exists in `TransformerConfig` but has no CLI argument in
this Megatron and cannot be reached from the config system.

**Recommendation: `cuda_graph_impl: none` at 1024 nodes.** The production config sets `local`.

### 7. `overlap_p2p_comm_warmup_flush` is a 1.76x PESSIMIZATION — do not enable it

Jobs 1452154 / 1452155, same PP=8/VPP=4 shape, graphs off, identical memory (0.6466 vs 0.6473),
only the flag differs: **170.7 vs 300.3 TFLOP/s**. It reproduced with graphs on (165.3, job
1451775), so it is the flag and not an interaction.

NB the first reading of job 1451775 blamed allocator thrash, because it ran at mem 0.9262 against
`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`. The graphs-off control at mem 0.6466
rules that out — the memory pressure was real but incidental.

### 4. Scaling: 1024 nodes buys wall clock, not efficiency

Best confirmed fp8 config per scale:

| scale | Tok/s/GPU | total tok/s | days@15T | node-days for 15 T |
|---|---|---|---|---|
| 256 nodes | 2814 | 2.88 M | 60.2 | ~15,400 |
| 512 nodes | 2102 | 4.30 M | 40.3 | ~20,600 |
| 1024 nodes | 1665 (VPP=4) | 6.82 M | 25.5 | ~26,100 |

Per-GPU efficiency 512 → 1024 at fixed GBS: **86% bf16, 79% fp8** (77% for the fp32-grad
config). The loss is the pipeline bubble doubling plus DP comm that no longer amortises — both
consequences of holding GBS fixed while doubling GPUs.

**1024 nodes costs ~27% more node-days than 512 for ~37% less wall clock.** Worth it only if
deadline-bound rather than allocation-bound. The single lever that fixes both loss terms at once
is raising GBS (which restores M and amortises the DP comm) — a training-dynamics decision, not
a config knob.

---

## How to reproduce

All commands run from the repository root on JUPITER (`scripts/oellm_32b.sh` handles `cd` +
`PYTHONPATH`). Sync local edits first with `bash sync_to_jupiter.sh`.

| result group | config | command |
|---|---|---|
| bf16 / fp8 shakeouts, 512 nodes | `speed_test`, `fp8_speed_test` | `scripts/oellm_32b.sh --config-name experiments/oellm_32b_dense/speed_test` |
| same at 1024 nodes | as above | append `backend.aux.active_nodes=1024` |
| VPP sweep (finding 2) | `speed_test_vpp1024` | `--array-subset 0` = VPP2 baseline, `1` = VPP4 |
| fp32 grad (finding 3) | `fp8_speed_test_fp32grad` | `--array-subset 0` = 512n, `1` = 1024n |
| sustained 1024-node | `stability_check_nodes1024` | 8000 iterations, checkpoints on |
| 256-node ramp | `stability_check_nodes256_bsramp` | post-ramp tail only |

Overrides on `run_autoexp.py` are **positional**, not `-o` (that flag is `render_config.py`
only):

```bash
scripts/oellm_32b.sh --config-name experiments/oellm_32b_dense/fp8_speed_test_fp32grad \
    --array-subset 1 --submit-and-exit "slurm.sbatch.time=01:00:00"
```

Check what a config will actually run *before* spending an allocation — note that
`render_config.py` does **not** apply sweep arms, so use `--dry-run` for anything swept:

```bash
PYTHONPATH=. python scripts/visualize_plan.py --config-name experiments/oellm_32b_dense/<cfg>
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/oellm_32b_dense/<cfg> --dry-run
```

Recompute the steady-state medians from any run's log (this is exactly how the tables above were
produced — median over iterations 4..N):

```bash
LOG=/e/project1/e-sta-openeurollm/pre_production_training/<run>/logs/slurm-<jobid>.log
sed 's/^\[[^]]*\]: *//' "$LOG" \
  | grep -oE 'elapsed time per iteration \(ms\): [0-9.]+|TFLOP/s/GPU\): [0-9.]+|Tok/s/GPU\): [0-9.]+|mem usages: [0-9.]+'
```

Confirm what a run *actually* used (settings such as `accumulate_allreduce_grads_in_fp32` are set
inside Megatron and never appear on the command line):

```bash
sed -n '/------------------------ arguments/,/end of arguments/p' "$LOG" | sed 's/^\[[^]]*\]: *//'
```

---

## Caveats

* **Run-to-run spread at 512 nodes is 2.7%.** Jobs 1365560 and 1365861 are byte-identical
  configs and measured 320.5 vs 312.2 TFLOP/s. Treat anything under ~3% as noise, and do not
  compare numbers across sessions when the effect you care about is that small.
* `TFLOP/s/GPU` is Megatron's own figure and counts recompute FLOPs. No run here uses recompute,
  so it ranks identically to Tok/s/GPU — but do not compare it against runs that do.
* TFLOP/s is **per GPU**. 1024 nodes has twice the GPUs of 512, so a lower per-GPU number still
  means higher total throughput (1.55× for 2× the hardware).
* Shakeout ≠ production. Sustained runs land ~13% lower once checkpoints, eval and dataloader
  jitter enter the median.
* `days@15T` assumes zero restarts. At 1024 nodes single-node faults are frequent enough that
  this is optimistic — see the failure taxonomy in `speed_test_vpp1024.yaml`.
* **1024-node speed tests have no fault tolerance** (`spare_nodes: 0`, `--max-restarts 0`, static
  rendezvous), so one bad node loses the whole allocation. Use
  `override /slurm: jupiter_autoexclude` and budget retries.
* **A `/e/project1` stale-file-handle outage is indistinguishable from a hung job** if you only
  read logs — it makes `grep -c` return 0 and can make a running job look restarted. Always check
  `ls /e/project1/... >/dev/null` before concluding a run has stalled.
