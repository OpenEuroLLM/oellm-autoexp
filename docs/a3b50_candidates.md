# 35B vs 50B hybrid MoE candidates for the 300k GPU-h run (2026-10-22)

Goal: within 3 days, prepare the model for a 300k GPU-h production run (~6-7T tokens at the
speeds below; 1,000 tok/s/GPU = 1.08T tokens). Candidates: the Qwen3.5-35B-A3B MoE shape, and
three ways to ~50B at about the same active size.

**Data:** `/gpfs/projects/ehpc533/users/surf593435/a3b50_speed_2026-10-22/`, readable by group
ehpc533:
- `results.csv`: all numbers below;
- one folder per arm: slurm log, launch script, watchdog GPU memory traces (stage = node index // 4).

**Suite:** `a3b50()` in `scripts/korbi/ep_speed/gen_ep_suites.py` ->
`config/experiments/korbi/hybrid_speed/ep_a3b50_mn5.yaml`.

**Setup.** All arms: 16 nodes, PP4 x VPP2 x EP4, MoonEP (64 SMs), full recompute, 4k, mbs 4,
gbs 2048, real text, no document masking, 12 steps, medians of steps 3-12, one run per arm.
"Production settings" = angular MuOwn (4e-3) + mHC-4 (mixer lr x0.25, a checkpoint every 2
layers, `RECOMPUTE_OFFLOAD=1`, `MHC_FUSED_READ=1`). The learning rates are benchmark values, not
tuned.

## Candidates

All: CKDA 3:1 (the plan's ratio, not the campaign's 5:1), NoPE GQA attention + XSA + gate, top-8
+ 1 shared expert of the routed width, sigmoid router + expert bias, 262k vocabulary.

| | hidden x layers | routed experts | total / active (incl. emb.) |
|---|---|---|---|
| **A** 35B (Qwen3.5 MoE) | 2048 x 40 | 256 x 512 | 34.2B / 3.0B |
| **B** 50B sparse | 2048 x 40 | 384 x 512 | 50.3B / 3.0B |
| **C** 50B wide | 2048 x 40 | 256 x 768 | 50.4B / 3.5B |
| **D** Kimi-Linear shape | 2304 x 27 (layer 1 dense 9216), CKDA 32 x 128, attn 32 / 4 | 256 x 1024 | 49.4B / 3.7B |

D differs from Kimi-Linear-48B-A3B in: CKDA for KDA, GQA for MLA, the 262k vocabulary, and no
routed scaling factor.

## Results (tok/s/GPU, TFLOP/s/GPU, worst-stage peak GiB of ~63.7)

| | A | B | C | D |
|---|---|---|---|---|
| Adam, plain | 7,742 / 115.6 / 46.2 | 7,392 (-4.5 %) / 110.4 / 56.1 | 7,236 (-6.5 %) / 132.7 / 54.1 | 7,480 (-3.4 %) / 142.6 / 52.6 |
| MuOwn, plain | | | 6,912 / 126.7 / 58.5 | |
| **production settings** | **6,301** / 94.1 / 49.1 | not run; est. ~5,900-6,050, ~60-62 GiB | **5,676 (-9.9 %)** / 104.0 / 58.3 | **6,044 (-4.1 %)** / 115.3 / 59.6 |
| tokens in 300k GPU-h | 6.8T | ~6.5T | 6.1T | 6.5T |

Percentages are against A with the same settings.

- **MuOwn** costs -4.5 % and +4.4 GiB on stage 0.
- **mHC-4** costs -18 to -19 % on every shape (the 39B production hybrid: -18 to -20 %). Memory
  stays flat with host-offloaded checkpoints; host RAM peaks at ~123 GiB per node.
- **D's loss spiked** in both 12-step runs at these learning rates, with Adam (14.6 at step 4,
  recovered) and with MuOwn + mHC-4 (grad norm 27-43, ended at 13.1). A, B and C were stable.

## Add-ons (loss from the campaign; scale up to ~490M active / 2.5B tokens)

- **mHC-4:**
  - Loss: -0.051 / -0.040 / -0.037 at s1 / s2 / s3, flat with tokens, ~N^-0.18 with size (about
    -0.027 extrapolated to 3B active), for ~18 % of the tokens.
  - Multi-key retrieval: s3 (cooled) plain 0.50 / 0.24 vs mHC-4 0.82 / 0.84 at 16k / 32k; s2
    (uncooled) plain 0.70 / 0.38 vs mHC-4 0.60 / 0.50 (§12.11). The retrieval benefit is not yet
    consistent.
- **Engram** (20 % of inactive params as tables): -0.015 / -0.018 at s1 / s2, shrinking with
  tokens (§12.2); -5.4 % throughput at production. At 6T tokens little or nothing is expected to
  remain, and the tables memorize under data replay.

## Assessment

1. **B first, pending its production-settings run.** Only the expert count changes vs A, so A's
   learning-rate work should transfer. Active compute and EP traffic stay at 3B, and at ~6T tokens
   the extra sparsity should pay. Its 16-node memory (~60-62 GiB estimated) eases at production
   node counts.
2. **A is the fallback:** the most headroom and stable, but ~30 % fewer parameters for ~5 % more
   tokens.
3. **D** is the fastest 50B, but needs its own learning-rate check (it spiked) and a rebalanced
   last stage. Only if that is clean within the prep window.
4. **C** is stable but the slowest (-10 %).

Add-ons: keep MuOwn (settings from `screen5opt`); mHC-4 is open until `ds5lc_s2` (per layer as
the compromise); leave Engram out.

## Open before launch

- B with production settings.
- Speed and memory at the production node count with 16k document-masked data.
- MuOwn settings and learning rate for the chosen shape.
- The mHC-4 decision.
- Checkpoint save / resume at scale.

Not tested here: CKDA 5:1, LatentMoE, 80B-A3B (its "sparsity pays over long training" argument
applies at 6T tokens).

## Notes

- Submit the suite with `HYDRA_STAGED_SWEEP_WORKERS=1`. With 8+ arms the DAG resolver forks
  workers, and `RootConfig` fails to unpickle (`hydra_staged_sweep/parallel.py`).
- The suite uses its own inductor / Triton / HF caches under `/gpfs/scratch/ehpc533/users/$USER/cache/`.
  The base slurm config's `$HYBRID_ENV/cache` is not writable from other accounts.
- Code: Megatron `a2f7be60b` for the runs. The later `3faedd827` changes no MoE / MoonEP / CKDA /
  mHC code.
