# Pipeline parallelism vs FSDP for the hybrid MoE (2026-10-23)

Question (user): instead of PP4 x VPP2 + EP4 + DP, use EP4 + FSDP (ZeRO-3, no pipeline). Is
there a strong argument against it, and does it matter more for wide residuals (mHC-4, LDM)?
This note is the back-of-the-envelope comparison, the measurements so far, and the open
points. Speed context: docs/ep_fp8_speed.md §6d-5..§6d-7.

## 1. Transferred bytes per trained token, per GPU

Everything is per GPU and per trained token (the unit of tok/s/GPU), so the layouts compare
directly.

| symbol | meaning | ours |
|---|---|---|
| h | hidden size | 2048 |
| n | residual width in units of h | 1 PreNorm, 4 mHC-4, ~8.5 LDM (35 KB / token state) |
| b, b_g | bytes per value: activations / parameters (b), gradients in the sync (b_g) | 2 (bf16) / 2 or 4 |
| P, v | pipeline stages, virtual chunks per stage | 4, 2 |
| T | tokens per micro-batch (mbs x seq) | 32k at mbs 2, 16k sequences |
| m | micro-batches per pipeline per step | 32 (8 nodes, gbs 512), 8 (128 nodes, gbs 2048) |
| Psi | parameters per EP shard = dense + experts / EP | 2.1B + 25.4B/4 = 8.45B (112 experts); 11.25B (160) |

**Pipeline (PP x VPP + DP).** A micro-batch crosses each of the v*P virtual-stage boundaries
once forward (activations) and once backward (their gradients), each n*h*b per token. Per GPU
(a pipeline's P GPUs share its m*T tokens):

    B_pp  ~ 2 * v * P * n * h * b                     per token  (independent of model size)

Each GPU holds Psi/P parameters and all-reduces their gradients over DP once per step (~2x
their size), over the m*T/P tokens it trains per step:

    B_dp  ~ 2 * Psi * b_g / (m * T)                   per token  (the "DP scaling" term, ~1/m)

**FSDP (ZeRO-3, EP for the experts, full recompute).** Each micro-batch all-gathers the
parameters twice (forward, recompute + backward) and reduce-scatters the gradients once:

    B_fsdp ~ Psi * (2*b + b_g) / T                    per token  (independent of m and n)

(Keeping the parameters gathered across the step changes this: see the ZeRO-1 / ZeRO-2 table
below, where the traffic falls with the tokens per GPU per step, m * T.)

Break-even micro-batch: T* = Psi * (2b + b_g) / (2 v P n h b + 2 Psi b_g / m).

**Batch size.** Without a pipeline every GPU is a data-parallel rank:
gbs = mbs x (number of GPUs) x m, with m the micro-batches per GPU per step. At 128 nodes (512
GPUs) mbs 8 and one micro-batch per step is gbs 8 x 4 x 128 = 4,096 sequences (67M tokens at
16k); gbs 8,192 is e.g. mbs 2 x m 8. ZeRO-3 re-gathers the parameters for every micro-batch, so
there only T (the micro-batch) amortizes the traffic, not the gbs. Keeping parameters (and
gradients) across the micro-batches of a step makes the gbs the lever, at the cost of memory:

| mode (Megatron-FSDP `data_parallel_sharding_strategy`) | bytes per token | extra memory per GPU |
|---|---|---|
| ZeRO-3 (`optim_grads_params`) | Psi (2b + b_g) / T | none (all sharded) |
| ZeRO-2 (`optim_grads`): parameters kept, gradients reduce-scattered per micro-batch | Psi b / (m T) + Psi b_g / T | full bf16 parameters (Psi b ~ 17 GB) |
| ZeRO-1 (`optim`): parameters and gradients kept, synchronized once per step | Psi (b + b_g) / (m T) | parameters + gradients (17 + 17-34 GB) |

At 128 nodes, gbs 8,192 = mbs 2 x m 8 per GPU: ZeRO-3 1,550, ZeRO-2 ~580, **ZeRO-1 ~130 (bf16
gradients) / ~190 (fp32) KB per token** -- below PP4 x VPP2's ~320 (mHC-4, m = 32), with no
bubble and no width term. Memory for ZeRO-1 with bf16 gradients: ~34 GB + ~15 GB activations
(mbs 2) + ~6 GB non-PyTorch ~ 55 of 63 GB: tight but plausible (fp32 gradients would not fit).
The same m per GPU is gbs 512 at 8 nodes with mbs 2, so 8-node tests measure this regime.

ZeRO-1 / ZeRO-2 fall with m * T, the tokens per GPU per step: B_zero1 ~ Psi (b + b_g) x (number
of GPUs) / (gbs x seq). A larger mbs and a larger m both help; mbs costs activation memory (the
recompute working set grows with T), m only time. (ZeRO-3: only T.) ZeRO-1's Psi (b + b_g) is
~45 GB at 160 experts (Psi = 11.25B), ~34 GB at 112 (8.45B): its tests use 112 experts.

**Expert parallelism.** EP does not enter the batch size: EP ranks are data-parallel ranks for
the dense part (each processes its own micro-batch; tokens go to the expert owners by
all-to-all), so gbs = mbs x (number of GPUs) x m with or without EP (only TP, PP, CP divide
it). EP enters Psi: Psi = dense + experts / EP, and every FSDP / ZeRO term is proportional to
it -- EP8 would make it 2.1B + 3.2B = 5.3B (~0.6x the traffic), but on MN5 EP beyond a node only
runs on the slower ncclep backend. EP's own all-to-all (MoonEP dispatch / combine, ~top-k x h x b
per token and direction, ~120 KB per token with backward and recompute) is the same in both
layouts and stays on NVLink inside the node at EP4, so it cancels from the comparison.

## 2. Numbers

KB per token per GPU; GB/s at 5,800 tok/s/GPU (the per-GPU InfiniBand link is ~25 GB/s,
NDR200, one HCA per GPU):

| layout | activations | parameters / gradients | total | at 5.8k tok/s |
|---|---:|---:|---:|---:|
| PP4 x VPP2, PreNorm (n = 1), m = 32 | 64 | 64 (fp32 grads) | ~130 | 0.7 GB/s |
| PP4 x VPP2, mHC-4 (n = 4), m = 32 (8 nodes) | 256 | 64 | ~320 | 1.9 GB/s |
| PP4 x VPP2, mHC-4, m = 8 (128 nodes, gbs 2048) | 256 | 258 | ~515 | 3.0 GB/s |
| PP4 x VPP2, LDM (n ~ 8.5), m = 32 | 545 | 64 | ~610 | 3.5 GB/s |
| FSDP, mbs 2 (T = 32k), any residual | 0 | 1,550 (bf16 grads) | ~1,550 | 9.0 GB/s |
| FSDP, mbs 4 (T = 64k) | 0 | 775 | ~775 | 4.5 GB/s |
| FSDP, mbs 8 (T = 128k) | 0 | 390 | ~390 | 2.3 GB/s |

Break-even for mHC-4 at m = 8: T* ~ 100k tokens, i.e. FSDP moves less than PP from ~mbs 6 at
16k; for LDM it is ~3x earlier (~mbs 2-3); for PreNorm it never is at these sizes.

## 3. Reading it

- **Volume is not what decides PP.** PP moves little, but its transfers sit on the critical path
  (each stage waits for the previous one) and it pays the bubble, ~(P-1)/(v*m): 4.7 % at m = 32,
  18.8 % at m = 8. The 128-node run lost ~12 % to the bubble and ~6 % to scale (4,803 vs 5,813
  tok/s/GPU; §6d-7).
- **FSDP moves 3-6x more at mbs 2-4, but all of it can overlap with compute:** 18-36 % of the link
  at the target throughput. Its costs are the overlap quality (NCCL kernels compete with MoonEP's
  64 SMs) and the first gather / last reduce of a step.
- **Wide residuals only enter PP's term** (proportional to n), FSDP's does not. mHC-4 is 4x the
  PreNorm activation traffic and LDM ~8.5x. The same width also multiplies PP's activation memory
  of the in-flight micro-batches, which is why PP needed the checkpoint offload to host for
  mHC-4 and why LDM was dominated by the 11-slot PCIe offload (2,852 tok/s/GPU; §6d-4). This is
  the structural argument for FSDP with wide residuals.
- **What mHC-4 costs under PP today:** 7:1 hybrid, 8 nodes, Adam, 160 experts, no Engram (f2,
  same launch): **PreNorm 7,677 vs mHC-4 6,110 tok/s/GPU (-20 %)**; the mHC kernels are ~6.6 % of
  the GPU time (profile §6d-5), the rest is the layout's: the 4-wide p2p, the offload, the
  memory-driven settings. Under FSDP only the kernels' share should remain: the f arms measure it.

## 4. FSDP in this Megatron: status

- Megatron-FSDP (megatron/core/distributed/fsdp/): units = TransformerLayer /
  MoETransformerLayer / MambaLayer (the hybrid's layers are covered); requires the regular
  distributed optimizer (no angular MuOwn path yet: MuOwn needs whole matrices -> reduce to an
  owner rank, update, gather from there, as Moonshot's distributed Muon; same volume as
  reduce-scatter + all-gather); `ckpt_format: fsdp_dtensor`.
- Frozen parameters are sharded too: the Engram tables (already sharded by our own code) must be
  hidden from FSDP before Engram runs under it.
- 1-node smoke (f_smoke1n, 8 blocks, 32 experts): runs.
- 8 nodes (f2): **every FSDP arm OOMs in the first backward** (inside MoonEP's backward, 54.5 GiB
  allocated at mbs 4; 13.3 GiB in use at the start with the shards). Diagnostics (f3): the
  all-to-all dispatcher instead of MoonEP, and an allocator trace of iteration 1. Suspect:
  MoonEP's expert-gradient path (it writes fp32 wgrads through FSDP's lazily allocated
  unsharded `main_grad` buffers).

### Measured (2026-10-23, 8 nodes, 7:1 hybrid, Adam, no Engram, 16k)

c1 (112 experts; PP and ZeRO-3 with MoonEP), tok/s/GPU:

| | PreNorm | mHC-4 |
|---|---:|---:|
| PP4 x VPP2, mbs 2, gbs 512 (= gbs 8192 at 128 nodes) | **7,780** | **6,128** (-21 %) |
| PP4 x VPP2, mbs 2, gbs 256 (= gbs 4096 at 128 nodes) | 7,437 | 5,846 |
| FSDP ZeRO-3, MoonEP, mbs 4, gbs 512 | 6,699 (memory 100 %) | OOM |
| FSDP ZeRO-2 / ZeRO-1 (bf16 gradients), mbs 2 | OOM | OOM |

f3 (160 experts, PreNorm, ZeRO-3, mbs 2): **all-to-all dispatcher 5,649 (memory 0.88), MoonEP
OOM.**

- **FSDP's problem is memory, not speed:** ZeRO-3 runs as soon as it fits, at -14 % vs PP with
  the memory saturated (the allocator under pressure). The estimate above (~13 GB shards + ~15 GB
  activations) is far below the ~60 GB measured: something besides the shards and the activations
  takes ~30 GB. ZeRO-1 / ZeRO-2 add the full parameters (+ gradients) on top and OOM.
- **MoonEP adds memory under FSDP** (all-to-all fits at 160 experts and mbs 2, MoonEP does not).
- **mHC-4 costs -21 % under PP** at both batch sizes: the layout cost FSDP should remove.
- Memory traces of the 1-node smoke (MoonEP vs all-to-all, iteration 2) are queued (f5).

### Measured after the memory fix (2026-10-24, c5: 8 nodes, 7:1, 112 experts, Adam, gbs 512)

**The FSDP memory problem was the recompute's chunk inputs.** Without a pipeline every GPU holds
all 96 layers (48 recompute chunks of 2); Megatron-FSDP's backward-hook bookkeeping kept every
chunk's input (fetched back from host) and its input gradient referenced until the micro-batch's
backward ended: 46 + 45 x 128 MB live at the peak for PreNorm (memory trace f6), 4x that with
mHC-4. CheckpointFunction.backward now drops the inputs' gradients and releases a fetched input
copy once the chunk's recompute is done (Megatron random.py; offload tests: bitwise-equal
gradients). mHC-4 under ZeRO-3: peak allocated 21.8 GiB at mbs 2, 2 fetched inputs live at the
peak. (Two slow runs, ~157 s / step, shared 7 nodes of rack as07r5: excluded.)

| HybridEP on both sides | PreNorm | mHC-4 | mHC-4 cost |
|---|---:|---:|---:|
| PP4 x VPP2, mbs 2 | 6,717 | 5,497 | -18 % |
| FSDP ZeRO-3, mbs 2 | 6,819 | 6,184 | -9 % |
| **FSDP ZeRO-3, mbs 4** | | **6,554** (memory 0.86) | |
| (PP4 x VPP2 with MoonEP, c1 / c3) | 7,780 | 6,151 | -21 % |

- **mHC-4 under FSDP: +19 % over PP at the same EP backend** (6,554 vs 5,497), 2.4 % below PP's
  PreNorm (6,717): the width cost of PP is gone, as the traffic model predicts.
- Already **+6.6 % over PP with MoonEP** (6,151), today's best PP. MoonEP is +16 % over HybridEP
  under PP (7,780 vs 6,717): with MoonEP under FSDP, mHC-4 could reach ~7,500.
- 8 nodes are pessimistic for FSDP's memory: the shards are split 32 / 8 ways (dense / experts),
  at 128 nodes 512 / 128 ways (~20 GB less per GPU).

### MoonEP under FSDP (2026-10-24, c7: 8 nodes, 7:1, 112 experts, Adam, gbs 512)

Two MoonEP bugs under Megatron-FSDP (Megatron moonep.py):
1. FSDP sets no `main_grad` attribute (its unsharded grad bucket comes from `get_main_grad()`):
   MoonEP fell back to private fp32 grads (~25-36 GB) the optimizer never saw -- the earlier
   MoonEP ZeRO-3 numbers (c1 / c3) did not train the experts.
2. FSDP's buckets are fresh for every backward (`overwrite_main_grad`); MoonEP accumulated into
   them uninitialised: grad norm 2,814 at step 1 (healthy ~3), NaN at step 3, then an illegal
   address in MoonEP's dispatch on the garbage routing (2-node run with CUDA_LAUNCH_BLOCKING).
   Now: zero the bucket, then accumulate. (fp32 gradient communication for now: MoonEP's wgrad
   needs fp32 main grads.)

| FSDP ZeRO-3 + MoonEP | PreNorm | mHC-4 |
|---|---:|---:|
| mbs 2 | 7,101 | 6,353 |
| **mbs 4** | **8,319** | **7,320** |
| (PP4 x VPP2 + MoonEP, c1 / c3) | 7,780 | 6,151 |
| (FSDP ZeRO-3 + HybridEP, mbs 4) | | 6,554 |

- **mHC-4 at 7,320 tok/s/GPU** (+19 % over PP + MoonEP, 2.4 % below the ~7,500 target); PreNorm
  at 8,319 is above PP's 7,780.
- Losses match the HybridEP runs (7.68 vs 7.67 at step 15): the expert gradients are right.
- **MoonEP stays balanced:** the step time is flat (~35.8 s) while the load-balancing loss goes
  1.00 -> 1.53 (steps 5-7); HybridEP's step grew 15 % there (the routing specialises when the LR
  reaches its peak; the expert bias pulls it back to ~1.06 by step 15).

## 5. Next

1. ~~Fix the FSDP OOM~~ (done: recompute-input release). MoonEP under FSDP (still OOMs:
   its expert-gradient path through FSDP's lazily allocated main_grad buffers).
2. 8-node A/B at the 7:1 layout: PP4 vs FSDP for PreNorm, mHC-4, LDM (f arms), mbs 2 / 4, and
   the ZeRO-1 / ZeRO-2 strategies (`optim` / `optim_grads`, bf16 gradients) at gbs 512 (m = 8
   per GPU, i.e. gbs 8,192 at 128 nodes).
3. The large setting (mHC-4 + Engram, the candidate stack) under FSDP: Engram tables hidden from
   FSDP; then angular MuOwn for sharded parameters if FSDP wins.
