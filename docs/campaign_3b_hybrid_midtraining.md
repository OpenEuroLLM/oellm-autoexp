# Campaign: 1M-context inference on consumer hardware (midtraining + next pretraining)

**Status:** plan, 2026-11-04. Nothing has been run or implemented for this campaign yet. It follows
[`hybrid_moe_3b_campaign.md`](hybrid_moe_3b_campaign.md), which covers the architecture search up to the
ambitious main run (referenced below as "campaign"). It builds on
[`inference_model_size.md`](inference_model_size.md) and the inference experiments `experiment/008`-`013`.

- **Configs:** `config/experiments/korbi/campaign_3b_hybrid_midtraining/`.
- **Running main run:** `config/experiments/korbi/production/hybrid_ambitious_v4_2ph_256_mn5.yaml`. Nothing in
  this document changes it. A restart of the main run or any change to its Slurm jobs needs the user's explicit
  go (AGENTS.md).

Labels used throughout:
- **[V]** verified from a source or measured by us.
- **[est.]** my own arithmetic, to be replaced by measurements.
- **[open]** a decision is still needed.

## 0. Goal and constraints

**Goal (user, 2026-11-04):**
- 1M+ token context on **consumer hardware**.
- Decode **and** prefill must stay reasonable at **batch 16**. Batch 1 is not the only operating point.
- The model is ambitious v4 (84B-A2.9B-E80B). The campaign has two outputs:
  1. a **midtraining stage** run from a v4 checkpoint: long-context extension, sparse attention, quantization,
     offload robustness;
  2. the design input for any **next pretraining run**, covering the decisions that only pay off from scratch.

**Hardware classes [open: confirm the reference devices]:**

| class | example | memory | link to the weights that do not fit |
|---|---|---|---|
| A, discrete GPU | RTX 5090 32 GB (1.8 TB/s), plus 128-256 GB DDR5 | weights are split between GPU and host | PCIe 5 x16, ~50 GB/s effective; NVMe ~7-14 GB/s |
| B, unified memory | Strix Halo / DGX Spark, 128 GB at ~256-273 GB/s; Mac Studio up to 512 GB at ~800 GB/s | everything in one pool | no transfer; bandwidth-bound |

**Feature gate** (same spirit as the first campaign):
- An inference feature that changes the LM is adopted only if its loss and eval cost stays within seed noise
  (~0.002-0.003 at s3) or is clearly paid back on the inference metric.
- An inference feature that leaves the LM unchanged (probes, drafters, engine work) needs no loss gate.

**Budget [open]:** the first campaign's 9k GPU-h cap was ~8.4k spent by 2026-10-24. This campaign needs its own
budget line. Rough per-item costs are given in §5.

## 1. Byte and FLOP budget of v4 at the target [est.]

**Model numbers [V]:**
- hidden 2048, 48 blocks;
- 6 GQA attention layers: 16 query heads, 2 KV heads × 128, NoPE;
- 42 CKDA layers: 16 heads, 128 × 128 state;
- 48 MoE layers: 360 experts × 768 wide, top-5, plus a shared expert;
- 2 Engram layers: 80B table parameters;
- untied vocabulary of 262k.

**Assumptions for the estimates:**
- One routed expert = 3 × 2048 × 768 = 4.72M parameters, about 2.5 MB at 4.25 bit (Q4 plus scales).
- Routed experts in total: 17,280, i.e. 81.5B parameters, about 43 GB at Q4.

| component | size | decode read per token, batch 1 | per step, batch 16 | where it lives |
|---|---|---|---|---|
| non-expert weights, 2.34B (head 0.54B) | ~1.3 GB Q4 | ~1.3 GB, the head being ~25 % at 4 bit | same, shared by the batch | GPU |
| routed experts | ~43 GB Q4 | 240 reads, ~0.6 GB | ~72 distinct per layer if routing were uniform, so ~8.6 GB touched per step | class A: a GPU cache plus host; class B: unified |
| Engram, 80B | 80 GB FP8 / ~40 GB Q4 | 2 layers × 24 rows × 64, a few KB | ×16 | host RAM or NVMe; addresses depend only on token ids, so **exactly prefetchable** |
| attention KV | 6 KB per token (bf16) | full attention at 1M reads 6.3 GB per token | 100 GB per step at 1M | must be offloaded; see §3.2 |
| attention KV, sparse (top-k 2048 + window 512) | as above, plus an index | ~13 MB fetched (bf16; the window stays on the GPU) | ~200 MB, i.e. ~4 ms over PCIe 5 | GPU index plus host KV |
| CKDA state | 42 MiB per sequence (fp32) | read (and written) every token | 0.7 GB | GPU |

**Consequences:**

1. **Sparse attention is mandatory at 1M, for prefill and for decode.**
   - Prefill of 1M tokens costs ~5.9 PFLOP in the non-attention part.
   - Dense causal attention in the 6 layers adds **~26 PFLOP**. Sparse attention with 2.5k attended keys adds
     ~0.13 PFLOP.
   - Decode without sparse attention reads 6.3 GB of KV per token per sequence.
   - At 64k, by contrast, attention is only ~1/4 of the forward FLOPs. This is why sparse attention barely
     speeds up training at 64k (LSA reports 1.1× per layer at 64k).
2. **At batch 16 the binding constraint is the union of experts per step, not the KV.**
   - 16 tokens × top-5 touch ~70 of the 360 experts per layer, about 8.6 GB per step.
   - Class B at 256 GB/s: ~34 ms per step, so ~470 tok/s total, ~30 tok/s per sequence.
   - Class A, with ~50 % of experts cached on the GPU: a 20-30 % miss rate means ~1.7-2.6 GB per step over PCIe,
     so ~35-50 ms per step. Misses can also be computed on the host CPU, in parallel with the transfer.
   - The levers are:
     - the cache hit rate (prediction and cache policy);
     - fewer bytes per expert (Q4 → Q3 with QAT);
     - fewer active experts per token: adaptive k through zero experts, see §3.1 R4.
3. **Speculative decoding helps at small batch, less at batch 16 with offloaded experts.**
   - Verifying B draft tokens per sequence grows the expert union. At batch 16 with 4 verified tokens, ~213
     distinct experts per layer (uniform estimate) serve ~3 accepted tokens per sequence. That is roughly
     break-even in expert bytes per accepted token against plain decode.
   - At batch 1 with resident experts the engine is launch-bound (experiment 009), so speculative decoding pays
     off there.
   - **Verification length must therefore be load-aware** (as with DSpark's confidence head), and decided per
     batch and per hardware class.
4. **Long prefill at batch 16** (16 × 1M tokens):
   - ~95 PFLOP of MoE and dense compute, i.e. ~10 min at ~150 TFLOP/s effective on class A, and several times
     that on class B.
   - Experts are streamed once per prefill chunk: 43 GB per 64k chunk is ~0.9 s over PCIe, which can overlap
     with compute.
   - Whether this is "reasonable" sets the target. [open: TTFT targets]

## 2. Starting point

- **Routing locality** [V] (campaign §"Expert offloading by MoE granularity", traces from s1/s2):
  - consecutive tokens share only **14-17 %** of their experts;
  - a 256-token window touches **~75 %** of all experts;
  - an LRU GPU cache of 30 % / 50 % of the expert bytes serves ~2/3 / ~0.83 of the reads;
  - LRU beats static pinning;
  - MoE granularity is not a tie-breaker.
- **Server decode today** [V] (`experiment/012`): ~100-108 tok/s per request at batch 1 on 4 × H100 with EP4.
  Going from 1 to 8 requests per GPU raises the step time from ~10 to ~16 ms. The engine is launch-bound.
  - Remaining costs: MoE, CKDA, the mHC read, NCCL.
  - Neither the HF export nor the vLLM port has a multi-token, speculative or offload path yet.
- **Retrieval depends on the 6 attention layers** [V]:
  - 7:1 retrieves 0.62 of the 64 facts at 64k, against 0.12 for 5:1;
  - pure CKDA is ~0.045 worse than the hybrid at s1.

  Sparsifying the few attention layers is therefore riskier than in a dense transformer.
- **NoPE plus qk-norm:** eval-time log-n scaling helps retrieval beyond 8k (campaign).
- **mHC-4 costs ~20 % of training throughput** at the production layout [V], and its read was ~20 % of the
  batch-1 decode profile in 008. This matters for the next model (§3.4).
- **v4 training constraints:**
  - full recompute in 2-layer chunks, with input offload;
  - strict `fsdp_dtensor` load and a packed optimizer-scalar format (`ac1020efc`);
  - MoonEP dispatch, where zero experts are not supported (alltoall / allgather only);
  - **CKDA and Engram do not support packed sequences under context parallelism** (`fla_cp` raises;
    Engram gathers the full sequence). This blocks >64k training with document masking.

## 3. What goes into (pre/mid-)training, and what can stay post-hoc

The question is which pieces must be in the model's training and which can be added later.

**Short answer:** most model-side items belong in training:
- sparse attention and its selection layout;
- KV compression and quantization;
- the long-context extension;
- expert-miss robustness and adaptive k;
- 4-bit QAT.

Two items are better post-hoc, because they never change the LM and must be retrained after post-training anyway:
- the **drafter**;
- the **expert predictor probe**.

MTP in pretraining is doubtful for us (§3.3).

| item | changes the LM | where | why / gate |
|---|---|---|---|
| native sparse attention: selection + window (LSA / NSA / CSA2-like, §3.2) | yes | **next pretraining: from scratch at 64k** (V4.1 recipe); **v4: midtraining conversion**, unless S2 shows conversion late is clearly worse than native, in which case it goes in at a v4 restart [open] | needed for 1M prefill and decode |
| selection granularity and layout suited to offload (pages, one index per token shared by all heads) | via the above | fixed **before** training the sparse layers | PCIe/DMA efficiency of the KV fetch |
| KV compression (m = 2 entries) | yes | with the sparse attention | KV bytes: 6 KB → 3 KB per token |
| KV quantization (FP8 / FP4 + scales) | small | QAT at the end of midtraining (V4.1 adds FP4 QAT in post-training) | 4× less host KV and fetch |
| context extension to 1M (CP for packed CKDA / Engram, data, NoPE length generalization) | yes | midtraining | the target itself |
| expert-miss robustness (simulated misses, handoff change 2) | yes, a regularizer | pretraining tail / midtraining | skip a missed expert instead of stalling |
| adaptive k through zero experts (`moe_zero_expert_num`, exists in the fork) | yes | next pretraining; a v4 restart only if very cheap | fewer expert bytes per token; needs MoonEP support |
| pre-gated / look-ahead routing (layer L picks L+1's experts) | yes | next pretraining only | exact prefetch; ablation R1 |
| detached next-router predictor (+ previous tokens, + look-ahead) | no | **post-hoc, offline** (R0); retrain after post-training | the handoff patch conflicts with full recompute (§3.1) |
| MTP / Soft-MTP | yes; +~20 % FLOPs per depth because of the 262k head | only if M1 shows a quality gain ≥ its compute | DFlash beats MTP on Qwen3.5-A3B; V4.1 dropped MTP |
| drafter (DFlash / DSpark) | no | post-training, backbone frozen, long-context and multilingual data | must match the final chat model |
| 4-bit expert QAT | small | end of midtraining / post-training | RTN costs only +0.004-0.005 (campaign), so PTQ may suffice |
| Single-Pass mHC; YOCO / CED-like KV sharing | yes, architecture | next pretraining candidates (ablation N1, N2) | training speed and prefill |
| CKDA delayed-state verify, Engram multi-token, expert cache, KV offload engine | no | inference engine | §3.3, §3.2 |

### 3.1 Expert prediction and offload

**Handoff patch** (`dump/handoff-next-router-prefetch`, Huu) [V, 2026-11-04 survey]:
- **Change 1** predicts layer L+1's router logits from layer L's router input. The input is detached, so the LM
  is unchanged.
- **Change 2** drops the experts the predictor would have missed.
- Both apply cleanly on Megatron `209683c90`. They **raise at start-up with `recompute_granularity: full`**,
  which production and the s1/s3 screens all use.
  - With 2-layer chunks, every L → L+1 pair crosses a checkpoint.
  - The first pass runs under `no_grad` and the recompute runs in reverse order.
- Further issues:
  - Change 2 probably mismatches MoonEP's top-k slot handling (`moonep.py:756`). A dropped expert still gets an
    index, while `tokens_per_expert` counts the reduced `routing_map`.
  - The new parameters break `strict_fsdp_dtensor_load`.
  - `cli_metadata.py` and `config_schema.py` must be regenerated.
- A recompute-compatible design:
  - stash layer L+1's targets (top-K plus logits, ~8 MiB per layer at 64k) in the first pass;
  - attach the loss during layer L's recompute;
  - `router_replay.py` is the existing pattern for this.

**Predictor features to compare (R0, offline on traces from a frozen checkpoint):**
1. **Free baseline:** router_{L+1}(h_L).
2. **Linear from h_L:** the handoff's predictor.
3. **Previous tokens:** layer L+1's top-k at t-1, t-2. Per layer, a learned stay probability plus a 360 × 360
   transition table (~130k parameters). This is the user's "previous token(s)" feature for decoding.
4. **Combined:** 2 + 3.
5. **Look-ahead:** d = 2-4 layers. A ~200 µs layer hides only ~10 MB of PCIe transfer, so batch 16 needs d > 1.
6. **Drafted future tokens:** in speculative decoding, drafted tokens give future positions, and their experts
   (and Engram rows) can be prefetched one verification ahead.

**Metrics:**
- recall@K for K = 5-30;
- union recall at batch 16;
- simulated stall time for class A and class B with `scripts/korbi/campaign/expert_cache_sim.py`, under LRU,
  predictive replacement and Belady's optimum (the ceiling);
- bytes per token.

**Skip policy (R2, on v4, no training):**
- the loss and evals when the missed experts are skipped (renormalized) at each prefetch budget, or computed on
  the CPU;
- this decides whether miss-robustness training (R1) is needed at all.
- `moe_masked_layer_eval_iters` and router replay are the tools.

**Engine:**
- a GPU expert cache, asynchronous prefetch over CUDA streams or UVA, and host-CPU computation of misses
  (llama.cpp / KTransformers-style);
- Engram rows from host or NVMe with exact one-token-ahead prefetch (the UVA Engram path from 012 already exists).

### 3.2 Sparse attention and KV offload at 1M

**Sources:**
- **DeepSeek-V4.1 "CSA2"** [V; the report has no "NSA2"]:
  - a compressed main KV (m = 2 in the encoder half, m = 1 in the decoder half);
  - a lightning-style indexer (32 heads × 128, FP4) picks the top 512 entries;
  - plus a 128-token sliding window;
  - Full / Reindex / Reuse layer modes share KV and top-k across layers;
  - FP4 main KV, 890 B per token;
  - **trained sparse from scratch at 64k, with no dense warm-up**;
  - it is built around MLA-like 512-dimensional shared-latent attention.
- **LSA** (arXiv 2610.42147) [V]:
  - keys are L2-normalized, pooled 4:1 and clustered (Ward linkage) within 512-key blocks; queries score the
    bucket centroids in a 64-dimensional index space, take the top 1024 buckets, and expand them to ≤ 4096 keys,
    plus 4 sinks and a 512-token window;
  - one index list per token, shared across heads (max over query heads);
  - no auxiliary loss; gradients stop through the selection;
  - **N-LSA was tested on a GDN 1:3 hybrid with NoPE and 1 KV head**, which is close to us: equal validation
    loss, RULER 32.2 vs 34.7 (full) at 128k, 1.6× per layer at 128k, 1.1× at 64k;
  - **D-LSA** trains only the index projections plus a temperature, with KL to the frozen model's attention
    (352M tokens), and also has a training-free PCA variant.
- **NSA** (2502.11089) [background]: compression, block selection and window branches with learned gates;
  selection shared per GQA group.
- **V3.2 DSA** [background]: token-level top-2048, dense indexer warm-up with KL, then sparse training.
  Megatron's DSA kernels are MLA/MQA-only (`unfused_dsa_fn`).

**Fit to v4:**
- GQA with 2 KV heads suits one index per token (LSA / NSA style) better than per-head selection.
- Only 6 attention layers sit 8 blocks apart, so CSA2's cross-layer reuse is less natural. Reindex-style reuse of
  one compressed KV is still worth an arm.

**Design for KV offload with the indexer in place:**
- **On the GPU:**
  - the index structure: LSA centroids, ~25 MiB per layer per sequence at 1M (scaled from the paper's
    246 MiB at 10M), or compressed indexer keys at FP4 (~32 MB per layer per sequence at m = 2, 128 dims);
  - the window and sinks;
  - an LRU cache of recently selected KV pages;
  - the CKDA state and the speculative tail.
- **On the host:** the full KV (FP8 / FP4 + scales), written page by page during chunked prefill. The index is
  built incrementally: LSA clusters per 512-key block, or the CSA2 compressor per entry.
- **Per decode step:** score the index on the GPU, then gather the selected pages through UVA zero-copy (the same
  mechanism as 012's UVA Engram) or batched DMA.
  - ~13 MB per sequence per step at bf16 (6 layers × 2048 keys × 1 KB), ~200 MB at batch 16, ~4 ms over PCIe 5.
  - Less in practice: FP4 and page reuse between consecutive steps.
- **Training-side requirements** (they decide which variant can be served efficiently):
  - the selection unit must be a contiguous page of 64-128 tokens, or LSA's 4-token pools stored cluster-ordered
    so that a bucket is one contiguous read;
  - a fixed top-k budget;
  - one index per token;
  - measured selection overlap between consecutive steps (how much of the cache survives).
- **Long-context infrastructure:**
  - CP for packed CKDA (`fla_cp`) and for Engram (local n-gram hashing, with a halo of 3 tokens);
  - CP for the new sparse module (Megatron's DSA CP is all-gather only);
  - 1M-token data: concatenated repositories and books, plus synthetic retrieval;
  - long-context evals beyond 64k: RULER and an extension of the campaign's `prepare_tasks.py longctx` (64-fact
    retrieval, passkey) to 128k-1M.

### 3.3 Speculative decoding

**CKDA multi-token verify with a delayed state update** (the user's proposal). There is no snapshot and no
rollback:
- Keep the committed state S_c at position c. Keep the newest tokens c+1..t (accepted but not yet folded, plus
  the draft) as a **tail cache** of (k, v, gates, β) per layer and head.
- For every tail position i, compute the output with the chunk form, exactly as in the training kernel:
  - the decayed read of S_c;
  - plus the intra-chunk term over the tail (the WY / UT-transform lower-triangular solve over ≤ C + B tokens).
- **Rejection:** drop the rejected tail entries. Nothing is undone.
- **Folding:** once the accepted tail reaches the chunk size C (e.g. 64), fold it into S with one chunk-state
  update.
- **Benefits:**
  - state writes drop by a factor of C;
  - the same path serves batched prefill/decode mixing;
  - tree verification works with a tree mask on the intra term (a solve per branch).
- **Caching the tail's WY factors** makes each new token O(n·d) plus the read of S_c.
- **Things to verify:**
  - the sign gauge and `signed_sigmoid2` gates with β up to 2 (negative eigenvalues) in the tail solve; fp32 at
    C = 64 is what training already uses;
  - tail memory: ~66 MB per sequence (bf16, 128 tokens);
  - the short-conv states (kernel 4, 3 tokens of history) and Engram's n-gram history, which simply live in the
    tail as well;
  - attention KV truncates trivially; mHC is stateless.
- This is the prerequisite for every drafter (`experiment/012` open items).

**Drafter:**
- **DFlash** [V]:
  - 5 draft layers, with KV injection of fused hidden states from 5 target layers;
  - block 16 with bidirectional attention inside the block;
  - trained post-hoc on target-regenerated responses;
  - Qwen3.5-35B-A3B: 2.3× vs MTP's 1.6× speedup;
  - degrades beyond its training length (4k → 16k) unless fine-tuned on long data.
- **DSpark** [V, V4.1's drafter]: 3 SWA blocks, a Markov head, and a confidence head driving load-aware
  verification length.
- **Open points for us:**
  - which mHC stream to tap (collapsed vs per-layer read);
  - long-context training of the drafter;
  - multilingual acceptance;
  - a reduced draft vocabulary (the 262k head);
  - verification length per batch and hardware class (§1.3).

**MTP in pretraining:**
- the upstream code runs our hybrid stack, but on the production config it needs:
  - `recompute_num_layers` 1;
  - Liger fused cross-entropy (full-vocab logits are ~32 GiB per depth at 64k);
  - `input_ids` for Engram;
  - an FSDP unit.
- **Soft-MTP** (2610.38331) [V]: a straight-through KL gradient-recovery trick at zero extra cost.
  - Acceptance length +2 %; zero-shot average +0.5 at 1.2B.
  - Perplexity is still worse than next-token prediction.
  - Dense ≤ 1.2B only.
- **Decision through M1:** pretraining MTP only if it is loss-neutral or better at s3 at equal wall clock.

### 3.4 Next-model architecture candidates (from V4.1)

- **Single-Pass mHC:** shift the input-mixing coefficients by one block so that a fused kernel reads and writes
  the residual once. This halves residual traffic. It targets our ~20 % mHC throughput cost. → N1.
- **CED / YOCO-like:** global KV from the middle layer serves the decoder half. In a hybrid this saves prefill
  compute only if the decoder half has no recurrent layers (SWA plus cross-attention only). That is a large change,
  but halving 1M prefill is a big win for consumer TTFT. → N2.

## 4. Order of work

**Phase 0** (now, on the v4 checkpoint and existing traces; no LM change, cheap):
- R0: predictor features;
- R2: skip-policy eval;
- S0: profile v4 attention at 64k-1M in the HF/vLLM path;
- B4: D-LSA retrofit on v4, PCA variant first;
- KV FP8 / FP4 PTQ eval;
- the CKDA delayed-state verify kernel, with a parity test against sequential decode on the local GPU.

**Phase 1** (screens, 8k, s2/s3, see §5):
- R1: miss robustness and pre-gating;
- R3: routing locality;
- R4: adaptive k;
- M1: MTP / Soft-MTP;
- N1: Single-Pass mHC.

**Phase 2** (s3 long-context extension 8k → 64k → 128k/256k; needs the CP work):
- S1: sparse variants;
- **S2: conversion timing.** Native from scratch vs converted at 50 % vs converted at the end. This decides
  whether v4 should switch at a restart or in midtraining;
- S3: KV QAT;
- S4: selection granularity and layer reuse.

**Phase 3:**
- a v4 midtraining pilot on 16-32 nodes: conversion + 128k-256k, then 1M;
- the drafter (M2) on the final post-trained model.

## 5. Ablation plan

The screens follow the first campaign:
- flag data, 8k with document masking, gbs 64 × 8k, 4000 steps (2.1B tokens), stable schedule;
- paired data order;
- s1 768 × 12 (~13-27 GPU-h per arm), s2 1024 × 18 (~27-54 GPU-h), s3 1280 × 24 with 7:1 × 3 (~50-126 GPU-h);
- seed noise ~0.002-0.003.

The handoff patch needs either the recompute-compatible redesign (§3.1) or recompute off at the screen sizes.

| ID | question | scale and arms | metrics | cost [est.] |
|---|---|---|---|---|
| R0 | How predictable is routing? | v4 checkpoint and an s3 checkpoint; `router_trace.py` on 8k/64k text, code, math and multilingual data; features 1-6 of §3.1 | recall@K, union recall at batch 16, simulated stall time for class A/B, bytes per token | ≤ 50 GPU-h |
| R2 | Do we need to train for misses? | v4: skip missed experts (renormalized) at prefetch budgets K; compare with CPU computation | Δ validation loss and evals at each budget | small |
| R1 | Offload-aware training | s3: reference / detached predictor (loss must match) / predicted miss-drop p = 0.1 / random-drop control / undetached predictor / pre-gated routing | loss; degradation under the skip policy; recall | ~6 × 50-126 |
| R3 | Routing locality | s3: reference vs a temporal-locality term at two strengths (e.g. a per-token stay bonus) | loss vs LRU hit rate and union at batch 16 | ~3 × 50-126 |
| R4 | Adaptive k | s3: zero experts (identity / zero, 1-2× k) vs reference, at equal mean active compute | loss; mean real experts per token; bytes per step at batch 16 | ~3 × 50-126; dispatcher support |
| M1 | MTP in pretraining? | s2/s3: next-token only / MTP-1 λ 0.1 / Soft-MTP-1 / Soft-MTP-2 | loss at equal tokens and equal wall clock; evals; self-speculative acceptance | ~4 × 50-126 |
| N1 | Single-Pass mHC | s3: mHC-4 vs Single-Pass mHC-4 | loss; tok/s | ~2 × 126 |
| S0 | Where does attention time go? | v4 HF/vLLM at 64k, 128k, 256k, 1M; prefill and decode | ms per layer by component | dev queue |
| B4 | Retrofit sparse attention | v4 + D-LSA (PCA → trained index); top-k 2048/4096 | 64-fact retrieval, passkey, RULER at 64k-256k vs full; selection overlap between steps | ~50-100 |
| S1 | Which native sparse attention? | s3 extension 64k → 128k: full / N-LSA / NSA-style pages + window / CSA2-like (m = 2 + indexer + SWA 128) / SWA-only control | position-binned NLL, retrieval, RULER, per-layer speed, offload fetch bytes | ~5 × 150+ |
| S2 | Native vs converted | s3: sparse from scratch / converted at 50 % / converted at the end (DSA-style indexer warm-up + sparse training) | as S1 | ~3 × 150+ |
| S3 | KV quantization | s3 / v4: FP8, FP4 + E4M3 scales per 16 channels; PTQ vs QAT | Δ long-context metrics | small-medium |
| S4 | Selection layout | s3: page size 64 / 128 / LSA 4-pools; per-layer vs one shared index (Reindex / Reuse) | quality vs DMA efficiency | ~3 × 150 |
| M2 | Which drafter? | frozen s3 target first, then v4: post-hoc MTP head / DFlash (block 8/16, 3/5 layers, mHC tap) / DSpark-like / EAGLE-3 | τ, real speedup at batch 1/4/16 per hardware class, expert bytes per accepted token, per language, 8k/32k | ~100-200; for v4, regenerating target responses dominates (~200+ GPU-h) |

## 6. Open decisions

1. Reference devices and targets: tok/s per sequence at batch 16 at 1M, and TTFT for 1M prefill, per hardware
   class.
2. Budget line for this campaign.
3. Whether the rest of v4's pretraining gets any change, e.g. sparse attention at a restart if S2 favours
   native. A restart needs the user's go. Otherwise everything goes into midtraining.
4. Whether a next pretraining run (v5) is in scope, which decides R1 / R4 / N1 / N2.

## 7. Log

- **2026-11-04:** campaign opened. Surveys:
  - the handoff patch (applies cleanly, conflicts with full recompute and MoonEP);
  - Megatron MTP blockers;
  - papers: Soft-MTP 2610.38331, DFlash 2602.06036, V4.1 CSA2 / DSpark, LSA 2610.42147.

  Nothing run.

## References

- Soft-MTP: https://arxiv.org/abs/2610.38331 (code: github.com/JaydenTeoh/Soft-MTP)
- DFlash: https://arxiv.org/abs/2602.06036 (github.com/z-lab/dflash); DSpark: https://arxiv.org/abs/2607.05147
- DeepSeek-V4.1 report: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf
- LSA: https://arxiv.org/abs/2610.42147 (github.com/Prefrontal-AI/latent-sparse-attention)
- NSA: https://arxiv.org/abs/2502.11089; TreeWY (GDN speculative verification): https://arxiv.org/abs/2608.20961
