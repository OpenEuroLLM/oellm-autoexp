# Ambitious model size at 64 nodes (2026-10-29)

The question (user, 2026-10-29): how large can the ambitious model be on 64 nodes of the cns144 pool
host, with more routed experts and possibly more Engram?

- Config: `config/experiments/korbi/production/hybrid_size64_mn5.yaml` (suite `size64`, pool
  `mn5_pool_p129`, host002 = cns144 job 47998967).
- Recipe: production `hybrid_main_mn5`: LDM, 7:1 attention first, CKDA, `muown_classical`,
  Megatron-FSDP + MoonEP EP 4, mbs 4, gbs 4096 x 16k, full recompute with host offload.
- Run length: 12 steps; speed is the mean of steps 4-12.

## Naming and parameter counts

Names are total-active-Engram:
- Total excludes Engram: the non-expert part (2.33-2.35B) plus experts x 0.2265B. One routed expert
  over the 48 MoE layers is 3 x 2048 x 768 x 48.
- Active is 2.93B per token for every arm: mixers / attention, LDM, 5 routed + 1 shared expert,
  routers, output head; no input embedding (`inference_sizing.py` convention). With the input
  embedding it is 3.47B.
- Engram tables: 2 layers x 2 orders x 8 heads x 64 x `engram_table_size` rows (19.48M rows = 40B).

## Results

| arm | model | s/step | tok/s/GPU | TFLOP/s | mem (driver) | max alloc / reserved GiB | non-PyTorch GiB | GPU-h for 12T |
|---|---|---|---|---|---|---|---|---|
| e216 (production) | 51B-A2.9B-E40B | 50.4 | 5,200 | 88.4 | 0.892 | 41.8 / 45.7 | | 641k |
| e216_pab | 51B-A2.9B-E40B | 50.5 | 5,193 | 88.3 | 0.898 | 41.8 / 45.7 | | 642k |
| e256_eng60 | 60B-A2.9B-E60B | 54.2 | 4,836 | 82.2 | 0.937 | 43.2 / 49.1 | 10.9 | 689k |
| e288 | 68B-A2.9B-E40B | 56.2 | 4,662 | 79.2 | 0.977 | 43.5 / 49.1 | | 715k |
| e288_pab | 68B-A2.9B-E40B | 56.5 | 4,637 | 78.8 | **0.992** | 43.5 / 50.7 | | 719k |
| e288_cap80 | 68B-A2.9B-E40B | 56.7 | 4,625 | 78.6 | 0.935 | 43.5 / 49.1 | 11.2 | 721k |
| e360_cap (cap 0.75) | 84B-A2.9B-E40B | 80.1 | 3,273 | 55.6 | 0.919 | 45.3 / 47.4 | 11.6 | 1,018k |
| e360_cap80 | 84B-A2.9B-E40B | 64.1 | 4,093 | 69.5 | 0.967 | 45.3 / 50.3 | 11.6 | 814k |
| e360_eng80_cap80 | 84B-A2.9B-E80B | 67.6 | 3,876 | 65.9 | 0.967 | 46.1 / 50.0 | 11.6 | 860k |
| e384_cap80 | 89B-A2.9B-E40B | 67.7 | 3,871 | 65.8 | 0.960 | 47.0 / 50.3 | 11.7 | 861k |
| e384_eng80_cap80 | 89B-A2.9B-E80B | 72.0 | 3,641 | 61.9 | 0.978 | 47.6 / 50.6 | 11.8 | 916k |
| e360_eng80 (no cap) | 84B-A2.9B-E80B | out of memory in step 1, in NCCL's cuMemAlloc | | | | | | |
| e432 (no cap) | 100B-A2.9B-E40B | out of memory in step 1 | | | | | | |

Column notes:
- **pab**: the bounded post-act norm (`moe_expert_post_act_norm`, `mlp_post_act_norm`,
  `post_act_norm_floor 1.0`).
- **capXX**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,per_process_memory_fraction:0.XX`.
- **mem**: Megatron's "mem usages" = 1 - free / total from the driver, so it includes memory outside
  PyTorch.
- **non-PyTorch**: from `MEM_BREAKDOWN=1`. It is MoonEP's prefetch pools (one slot per local expert,
  2.4 GiB at 360), the MoonEP buffer (1.6 GiB), the CUDA context (0.5 GiB) and NCCL (< 1 GiB).
- **Code**: e216, e288, e432, e360_eng80, e360_cap and e384_cap ran on Megatron aed8c0f86. e216_pab,
  e288_pab and e256_eng60 ran on the same Megatron with cute_kernels 4739513 (numerics / tests only).
  The cap80 arms ran on eccf83ee8; the Engram-80B cap80 arms on 77e012979. These later commits added
  metrics and an off-by-default post-act-norm variant. Differences under ~1 % between the code
  groups are not meaningful.

## Findings

- **Step time is about 33 s + 0.081 s x experts** (fitted on e216 / e288).
  - The fit predicts e256 within 0.5 s and e360_cap80 within 1.9 s.
  - The expert term is per-micro-batch FSDP gathers and reduce-scatters of the local experts plus
    MoonEP's prefetch. It does not shrink with more nodes. At 216 experts it is ~35 % of the step.
  - So throughput ~ experts^-0.35 to experts^-0.5 over this range, not 1 / size.
- **What runs out of memory is the allocator cache, not the peak.**
  - e360_eng80 failed in NCCL's own allocation while PyTorch still held its cache.
  - The fix is a 0.80 cap, which leaves ~1-1.5 GiB for NCCL. It is nearly free when it isn't tight
    (e288_cap80 -0.8 %).
  - 0.75 is too tight at 360 experts. The allocator frees and re-maps every step: -20 % vs cap 0.80.
- **More Engram is cheap in compute and memory.**
  - Memory: +20B of tables is ~+0.5 GiB peak, with no growth outside PyTorch.
  - Speed: +0.5 s per step at 256 experts. At the cap's edge (e384) +40B cost 4.3 s, mostly
    allocator pressure.
  - Value: the Engram share of the inactive parameters is already ~45 % at E40B, above the
    allocation screen's optimum of 20-30 % (campaign doc section 12.2).
- **The bounded post-act norm is free**: -0.1 % speed, +0.4 GB.
- **The uncapped limit at mbs 4 is 288 experts, and with pab it is too tight for production.**
  - With cap 0.80, 384 experts + Engram 80B runs (89B-A2.9B-E80B, ~169B) with ~1.4 GiB free. That
    is enough for a test, not for months of saves and evals.
  - With cap 0.80, 360 + E80B keeps ~2.1 GiB free.

## Recommendation (2026-10-29, pending the user's decision)

Restart on e216 (51B-A2.9B-E40B) + the bounded post-act norm.
- It is the fastest: 641k GPU-h for 12T.
- It keeps a memory margin of 0.90.
- At ~29 GB (Q4_K_M) it fits 32 GB consumer GPUs. With ~3.5 bits / weight it fits 24 GB. The
  Engram tables can stay in host RAM.
- Bigger models buy little at 43x routed sparsity: s1 32x -> 64x was -0.008 nats. That is eaten by
  the lost tokens.
- Use Megatron >= 570195e7c so the LDM metrics are logged.
