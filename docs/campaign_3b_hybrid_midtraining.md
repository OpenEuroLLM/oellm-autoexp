# Campaign: 1M-context inference on consumer hardware (midtraining + next pretraining)

**Status:** plan, 2026-11-04. Updated 2026-11-05: user targets (§0), phase split (§4), R0/R2 enqueued (§3.1,
§7). It follows
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

**Goal (user, 2026-11-04 / 11-05):**
- Two end-state operating points:
  - **10M tokens of context** for a single stream;
  - **500k tokens at batch 16**, with decode **and** prefill reasonable.
- The **"ultimate" device is an RTX 3090 / 4090 (24 GB)**, with large host RAM and disk available for offloading.
  - Weights are stored quantized; matmuls still run in BF16 (3090, Ampere) or FP8 (4090, Ada).
  - Expert offloading must therefore be pushed as far as possible, with both **next-layer** and **next-token**
    router prediction (§3.1).
- The model is ambitious v4 (84B-A2.9B-E80B). The campaign has two outputs:
  1. a **midtraining stage** run from a v4 checkpoint: long-context extension, sparse attention, quantization,
     offload robustness;
  2. the design input for any **next pretraining run**, covering the decisions that only pay off from scratch.

**Hardware classes:**

| class | example | memory | link to the weights that do not fit |
|---|---|---|---|
| **A0, the target** | RTX 3090 / 4090, 24 GB (~0.94 / 1.0 TB/s), 128-256 GB DDR4/5 host RAM, NVMe | ~15 GB for experts after weights, state, index and activations | PCIe 4 x16, ~25 GB/s effective; host DDR ~50-90 GB/s for host-side expert compute; NVMe ~7 GB/s |
| A, larger discrete GPU | RTX 5090 32 GB (1.8 TB/s) | as A0, more cache | PCIe 5 x16, ~50 GB/s |
| B, unified memory | Strix Halo / DGX Spark 128 GB at ~256-273 GB/s; Mac Studio up to 512 GB at ~800 GB/s | everything in one pool | no transfer; bandwidth-bound |

**Precision (user, 2026-11-05):** no FP4 in the (still long) midtraining, since H100s have no FP4. KV quantization
is FP8. 4-bit **storage** of the weights for A0 comes from post-training quantization; the measured RTN cost is
+0.004-0.005 (campaign). An optional fake-quant QAT pass runs in BF16 and needs no FP4 hardware.

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

### 1.1 The target points on a 24 GB 3090 / 4090 [est.]

**500k at batch 16:**
- **KV:** 6 KB × 500k × 16 = 49 GB (bf16) or 25 GB (FP8). It lives in host RAM. The sparse fetch is ~100-200 MB per
  step, i.e. 4-8 ms over PCIe 4.
- **GPU index for all 16 streams:** must stay at ~1-1.5 GB.
  - A CSA2-like FP8 indexer key of 128 dimensions at m = 2 is 32 MB per layer per stream, 3 GB in total. That is
    too much next to the expert cache.
  - Index dimension ≤ 64 and compression m ≥ 4 (or LSA centroids) fit.
  - **Index size is therefore a training-time choice** (S4).
- **Experts:** ~15 GB of GPU cache holds ~35 % of the 43 GB (Q4).
  - A step touches ~70 experts per layer, ~8.6 GB.
  - Prediction plus cache leave maybe 30-45 % misses, ~3 GB per step. That is ~120 ms over PCIe 4 alone, or
    ~35-45 ms when the misses are split between PCIe fetch and host-CPU computation (KTransformers-style).
  - The result is ~25 steps/s, i.e. ~400 tok/s in total and ~25 tok/s per stream. **R2's substitute policy and R4
    (adaptive k) are the levers that would move this most.**

**10M at batch 1:**
- **KV:** 61 GB (bf16) or 31 GB (FP8) in host RAM.
- **GPU index:** LSA's 246 MiB per layer at 10M (the paper) is 1.5 GB for 6 layers. CSA2-like keys (FP8, 128
  dimensions, m = 2) would be 3.8 GB.
- **Prefill:** ~60 PFLOP outside attention.
  - Sparse attention adds ~1.3 PFLOP.
  - **A flat indexer that scores every key costs ~40 PFLOP** (64 dimensions, 6 layers, T²/2), which is as much as
    the rest of the model.
  - At 10M the selection must therefore be hierarchical: LSA's tree search, or a coarse-to-fine block index like
    V4.1's hierarchical indexer.
  - That is ~5-10 min at ~150-300 TFLOP/s effective (FP8 on the 4090, BF16 on the 3090 is half). Streaming all
    experts per 256k-token chunk adds ~70 s over PCIe 4, which overlaps with compute.
- **Decode:** the KV fetch (~13 MB per step) and a flat index read (~2 ms) are small. The expert misses dominate
  (§1, batch 1: 240 expert reads per token, ~0.6 GB).

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
| KV quantization (FP8 + scales; **no FP4 in midtraining**, user 2026-11-05) | small | FP8 QAT at the end of midtraining | 2× less host KV and fetch |
| context extension to 1M (CP for packed CKDA / Engram, data, NoPE length generalization) | yes | midtraining | the target itself |
| expert-miss robustness (simulated misses, handoff change 2) | yes, a regularizer | pretraining tail / midtraining | skip a missed expert instead of stalling |
| adaptive k through zero experts (`moe_zero_expert_num`, exists in the fork) | yes | next pretraining; a v4 restart only if very cheap | fewer expert bytes per token; needs MoonEP support |
| pre-gated / look-ahead routing (layer L picks L+1's experts) | yes | next pretraining only | exact prefetch; ablation R1 |
| detached next-router predictor (+ previous tokens, + look-ahead) | no | **post-hoc, offline** (R0); retrain after post-training | the handoff patch conflicts with full recompute (§3.1) |
| MTP / Soft-MTP | yes; +~20 % FLOPs per depth because of the 262k head | only if M1 shows a quality gain ≥ its compute | DFlash beats MTP on Qwen3.5-A3B; V4.1 dropped MTP |
| drafter (DFlash / DSpark) | no | post-training, backbone frozen, long-context and multilingual data | must match the final chat model |
| 4-bit weight storage (experts, then the rest) | small | PTQ after training; optional fake-quant QAT in BF16 at the end | RTN costs only +0.004-0.005 (campaign); matmuls stay BF16 / FP8 |
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

**R0 / R2 as implemented** (`scripts/korbi/campaign_midtraining/router_probe.py`):
- **Setup:** one MN5 node on the HF export of v4_2ph step 8000, with the export's fast kernels.
  - The probe replaces every MoE forward with the router's own rule in torch. A parity check against the export's
    forward is written to the summary.
  - Data are the fixed validation sets (web_en, multilingual, code, math, reasoning: 16 rows × 2k, i.e. 16 decode
    streams at the same position; long_64k: 1 × 64k).
  - Config: `config/experiments/korbi/campaign_3b_hybrid_midtraining/r0_r2_router_probe_mn5.yaml`.
- **R0 targets:** layer m's own top-5.
- **R0 features**, i.e. what a decoder knows before layer m runs for token t:

  | feature | what it is | prediction type |
  |---|---|---|
  | `x<d>` | the router input of the MoE layer d = 1 / 2 / 4 below, same token | next layer |
  | `xprev` | layer m's router input at t-1 | next token |
  | `selprev` | layer m's experts at t-1 | next token |
  | `emb` | token t's embedding, known once the token is sampled | — |
  | `xlastprev` | the last MoE layer's input at t-1 | R0 only |

- **R0 predictors:**
  - **free:** the target router on `x<d>` / `xprev`, the previous token's experts, a decayed 16-token history, an
    oracle;
  - **learned online** (prequential: every batch is scored before it is trained on; each starts at its free
    baseline; two learning rates, the best one goes on to R2):
    - `lin_d<d>`, `lin_prev`;
    - `tokahead` = xprev + selprev: before token t is sampled, i.e. prefetch across the token boundary;
    - `start` = emb + xprev + selprev: at the start of t's step;
    - `start_deep`;
    - `combo_d<d>` = start + x<d>: next-layer **and** next-token information.
- **R0 metrics:**
  - recall@K and gate-weighted recall@K for K = 5-64;
  - batch-16 union coverage and union size;
  - cache windows (last 1-64 positions, per stream or over the batch) as the reference;
  - top-5 traces of the first batches, for offline LRU / Belady simulation with `expert_cache_sim.py`.
- **R2:** forwards in which each token and layer may only use an available set, built from:
  - a predictor's top-K, per stream or the batch-16 union;
  - and / or a cache window.

  Misses are skipped (renormalized or not) or substituted by the router's top-5 within the available set. Also
  run: drop-the-lowest-1/2 as calibration. Reported: ΔNLL, miss rate (by count and gate weight), available-set
  size. Skips propagate through the layers.
- **Not yet covered:**
  - drafted future tokens as a prefetch source (needs the drafter);
  - host-CPU computation of misses (an engine question, not a loss one).

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
    246 MiB at 10M), or compressed indexer keys in FP8 (~32 MB per layer per sequence at m = 2 and 128 dims; ≤ 64 dims and
    m ≥ 4 for the batch-16 budget of §1.1);
  - the window and sinks;
  - an LRU cache of recently selected KV pages;
  - the CKDA state and the speculative tail.
- **On the host:** the full KV (FP8 + scales), written page by page during chunked prefill. The index is
  built incrementally: LSA clusters per 512-key block, or the CSA2 compressor per entry.
- **Per decode step:** score the index on the GPU, then gather the selected pages through UVA zero-copy (the same
  mechanism as 012's UVA Engram) or batched DMA.
  - ~13 MB per sequence per step at bf16 (6 layers × 2048 keys × 1 KB), ~200 MB at batch 16, ~4 ms over PCIe 5.
  - Less in practice: FP8 KV and page reuse between consecutive steps.
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

**Context phases of the model (user, 2026-11-05):**
- The 64k extension is already v4_2ph's phase 2: 64k with gbs 1024 from step 23,691, i.e. 90 % of the tokens.
- **C1, dense long context:** extend beyond 64k with dense attention, to 256k-512k. This needs packed-sequence CP
  for CKDA and Engram, long data, and NoPE length generalization (log-n scaling).
- **C2, attention sparsification:** convert the 6 attention layers to sparse attention at the end of C1's length.
  - Indexer warm-up against the dense attention (DSA-style KL, or D-LSA-style index distillation), then joint
    sparse training.
  - Then extend to 1M-2M. 10M is reached through the sparse layers' length generalization and verified by evals;
    training at 10M is not planned.
- **C3, deployment hardening** (each item only if its ablation pays off):
  - FP8 KV QAT;
  - miss / substitute robustness;
  - a fake-quant pass;
  - then post-training and the drafter.
- S2 (native vs converted) remains the check that converting late is good enough. If not, the next pretraining
  run trains sparse from scratch.

**Phase 0** (now, on the v4 checkpoint; no LM change, cheap):
- R0 + R2 (§3.1): enqueued 2026-11-05;
- S0: profile v4 attention at 64k-1M in the HF/vLLM path;
- B4: D-LSA retrofit on v4, PCA variant first;
- KV FP8 PTQ eval;
- the CKDA delayed-state verify kernel, with a parity test against sequential decode on the local GPU.

**Phase 1** (screens, 8k, s2/s3, see §5):
- R1: miss robustness and pre-gating;
- R3: routing locality;
- R4: adaptive k;
- M1: MTP / Soft-MTP;
- N1: Single-Pass mHC.

**Phase 2** (s3 long-context mirror of C1 / C2; needs the CP work):
- S1: sparse variants;
- S2: conversion timing;
- S3: KV FP8 QAT;
- S4: selection granularity, index size and layer reuse, under the 24 GB GPU-index budget of §1.1.

**Phase 3:**
- v4 midtraining: a C1 pilot on 16-32 nodes, then C1 / C2 at scale;
- the drafter (ablation M2) on the final post-trained model.

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
| R5 | CED-style pre-routing | s3: decoder-half MoE layers route from the middle layer's state (same token) vs reference; also d = 2 / 4 pre-gating | loss; lookahead gained; prefetch exactness | ~3 × 50-126 |
| R6 | Chunk-causal expert subsets | s3: per chunk of 256 / 1024 tokens, M = 32 / 64 / 128 experts per layer from the prefix, top-5 within | loss vs M; resident bytes at batch 1 / 16 | ~4 × 50-126 |
| R7 | Hash routing in the lowest layers | s3: token-id (or bigram-hash) routing for MoE layers 0-2 vs learned | loss; prefetch at token time | ~2 × 50-126 |
| M1 | MTP in pretraining? | s2/s3: next-token only / MTP-1 λ 0.1 / Soft-MTP-1 / Soft-MTP-2 | loss at equal tokens and equal wall clock; evals; self-speculative acceptance | ~4 × 50-126 |
| N1 | Single-Pass mHC | s3: mHC-4 vs Single-Pass mHC-4 | loss; tok/s | ~2 × 126 |
| S0 | Where does attention time go? | v4 HF/vLLM at 64k, 128k, 256k, 1M; prefill and decode | ms per layer by component | dev queue |
| B4 | Retrofit sparse attention | v4 + D-LSA (PCA → trained index); top-k 2048/4096 | 64-fact retrieval, passkey, RULER at 64k-256k vs full; selection overlap between steps | ~50-100 |
| S1 | Which native sparse attention? | s3 extension 64k → 128k: full / N-LSA / NSA-style pages + window / CSA2-like (m = 2 + indexer + SWA 128) / SWA-only control | position-binned NLL, retrieval, RULER, per-layer speed, offload fetch bytes | ~5 × 150+ |
| S2 | Native vs converted | s3: sparse from scratch / converted at 50 % / converted at the end (DSA-style indexer warm-up + sparse training) | as S1 | ~3 × 150+ |
| S3 | KV quantization | s3 / v4: FP8 per-page / per-channel scales; PTQ vs QAT (no FP4, user 2026-11-05) | Δ long-context metrics | small-medium |
| S4 | Selection layout | s3: page size 64 / 128 / LSA 4-pools; per-layer vs one shared index (Reindex / Reuse) | quality vs DMA efficiency | ~3 × 150 |
| M2 | Which drafter? | frozen s3 target first, then v4: post-hoc MTP head / DFlash (block 8/16, 3/5 layers, mHC tap) / DSpark-like / EAGLE-3 | τ, real speedup at batch 1/4/16 per hardware class, expert bytes per accepted token, per language, 8k/32k | ~100-200; for v4, regenerating target responses dominates (~200+ GPU-h) |

## 6. Open decisions

1. Targets: the points are set (10M at batch 1, 500k at batch 16, on a 3090 / 4090); the tok/s and TTFT
   numbers to aim for are still open.
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
- **2026-11-05:**
  - User targets: 10M context at batch 1 and 500k at batch 16; 3090 / 4090 as the target device; no FP4 in
    midtraining; a dense long-context phase before sparsification.
  - R0 / R2 implemented (`router_probe.py`) and checked locally on a reduced random-weight model:
    - the oracle recall is 1.0;
    - the probe routing matches the export forward (ΔNLL 0 to 2e-4);
    - short and long domains both run.
  - Enqueued into `mn5_pool_o260c` as `campaign_mid_r0r2_v4_2ph_8000_8c9181`: 1 node (a buffer node of the 260; the
    main run uses 256), snapshot `autoexp_midr0r2`, ≤ 3 h, export `ambitious_v4_2ph_fp8_8000`.
    - Output: `/gpfs/scratch/ehpc533/hybrid_nucleus/campaign_midtraining/r0r2_v4_2ph_8000/`
      (`r0_summary.md`, `r2_summary.md`).
    - It waits behind another session's FLAG-eval shards on the other buffer nodes.

- **2026-11-05, R0 / R2 results** (o260c.0030, 1 node, 59 min; full summaries in
  `docs/campaign_3b_hybrid_midtraining_results/r0r2_v4_2ph_8000/`):
  - **How the run went:** two earlier attempts (.0023, .0027) were preempted when the main run took its nodes back.
  - **Parity:** probe routing vs the export forward differs by +3e-4 NLL on one web_en batch.
  - **R2 noise:** 2 batches × 32k tokens per domain, so ΔNLL of about ±0.002 is noise.
  - **R0: next-layer prediction works without training.** The free baseline is layer m's router on the input of
    the MoE layer below.

    | lookahead | R@5 | R@10 | gate-weighted R@10 |
    |---|---:|---:|---:|
    | d = 1 | 0.78-0.82 | 0.91-0.95 | 0.96-0.97 |
    | d = 2 | 0.69-0.76 | 0.84-0.91 | — |
    | d = 4 | 0.58-0.68 | 0.74-0.83 | — |

    - By depth (web_en): R@10 is 0.47 / 0.62 / 0.64 at MoE layers 0-2, ≥ 0.91 from layer 3, and ~0.99-1.00 in the
      top 16 layers. The first 3 layers are the hard part; keep them resident.
  - **R0: next-token prediction is weak.**
    - The previous token's experts give R@5 0.25-0.32, against 14-17 % at s1 / s2. It rises with depth, from 0.05
      at layer 0 to ~0.35 at layers 19-22.
    - The best learned token-ahead predictor (`start_deep`: emb + xprev + selprev + previous last-layer input)
      reaches R@5 0.29-0.42 and R@10 0.42-0.57.
  - **R0: the learned next-layer predictors did not beat the free baseline.** Online training at 3e-6 / 3e-5 is
    flat to slightly worse, e.g. lin_d1 code R@5 0.764 vs 0.775. This setup (online, domain-interleaved,
    prequential, a listwise loss) is not the way to learn them; an offline-trained, held-out-tuned version is still
    open.
  - **Batch 16:** the true expert union per step is **45-65 per layer** (code 45, web 65), below the 72 of the
    uniform estimate in §1. The per-stream top-10 of the d = 1 baseline covers 96-97 % of the assignments with
    78-111 experts.
  - **R2: robustness to dropping experts.** Dropping the lowest 1 of the top-5 costs +0.004-0.015 NLL; dropping 2
    costs +0.018-0.052. Long 64k and multilingual are the most sensitive.
  - **R2: policies** (ΔNLL; miss rate; available experts per layer):

    | policy | ΔNLL | miss | available |
    |---|---|---|---|
    | d = 1 prefetch top-10, per stream, skip | +0.002-0.008 | 5-7 % | 10 |
    | same, substitute | +0.001-0.004 | 5-7 % | 10 |
    | d = 1 top-10 + the last 16 positions' experts, per stream, substitute | +0.000-0.002 | 3-4 % | 40-49 |
    | batch 16: combo_d1 top-10 union + the last 4 positions, substitute | ≤ +0.002 | 2 % | 138-182 of 360 |
    | d = 2 top-10, substitute | -0.003-+0.009 | 10-14 % | 10 |
    | caches alone (window 64, bs1) | +0.04-0.14 | | ~75-103 |
    | token-ahead-only prefetch (top-16) | +0.3 to +1.1 | 48-53 % | 16 |

  - **R2: substitute (re-route within the available set) beats skipping everywhere,** roughly halving the cost.
  - **Consequences:**
    1. Layer-ahead prefetch (d = 1, top-10, plus a small recency cache) with substitute-on-miss is already within
       noise of the full model on the unchanged v4. **R1 (miss-drop training) is not needed for d = 1.**
    2. **Token-ahead prefetch is the gap.** It decides how much latency a 3090 / 4090 can hide beyond one layer.
       It is where training could help: R3 locality, pre-gating, or a learned predictor done properly.
    3. Batch 16 needs ~40-50 % of the experts resident per step at 2 % misses. That makes the GPU cache size and
       the host-side computation of misses the real budget (§1.1).

- **2026-11-05, prefill-informed (non-causal) expert prediction** (user question;
  `scripts/korbi/campaign_midtraining/prefill_working_set.py` on the R0 traces):
  - **Setup:** per layer, pin the C experts most used in the prompt (first half of a row) for the continuation
    (second half). Compare with an LRU cache of C warmed on the prompt, and with global popularity (other domains).
  - **Results** (hit rate of the continuation's assignments, C = 64 of 360 per layer):

    | policy | short domains (1k → 1k) | long_64k (32k → 32k, one document) |
    |---|---:|---:|
    | prompt-pinned | 0.52-0.72 | **0.24** |
    | LRU | 0.72-0.85 | 0.78 |
    | global popularity | 0.07-0.21 | — |

    - Global popularity falls *below* random (0.18): balancing flattens it, and the experts are domain-specific.
    - With one pinned set for 16 streams, C = 192 reaches 0.86-0.94.
  - **Reading:** the working set drifts within a document. Locality is temporal, not prompt-global, so a whole-
    prompt summary does not beat recency. The lower layers (0-2) are the worst for every cache policy (LRU at
    C = 64: 0.58-0.75).
    - Caveat: the short rows concatenate documents, which favours LRU a little. The long_64k test is clean.
  - **Prefill itself needs no expert prediction.** A chunk of a few thousand tokens touches nearly every expert,
    so prefill streams all experts per chunk anyway. The lever there is the chunk size (§1.1).
  - **Training-side non-causal routing is excluded.** A routing choice that depends on future tokens leaks them
    into the LM (top-5 of 360 per layer is ~35 bits per layer per token). V4.1's CED is causal as well: it is a
    prefill-compute trick (decoder global KV from the middle layer).
  - **The causal analogues go to the ablation list (§5):**
    - **R5, CED-style pre-routing:** decoder-half MoE layers route from the middle layer's hidden state of the same
      token. That gives ~24 layers of exact lookahead for half the experts. (R0's untrained d = 4 baseline
      already drops to R@10 0.74-0.83; trained routing would be exact by construction, at some quality cost.)
    - **R6, chunk-causal expert subsets:** per chunk of C tokens, a selector on the prefix picks M experts per
      layer, and the tokens route within them. This gives exact per-chunk prefetch, e.g. M = 64 → ~7.7 GB
      resident on a 24 GB card. It helps batch 1 (the 10M point) more than batch 16, where the subsets of 16
      streams union up.
    - **R7, hash / token-id routing for MoE layers 0-2:** the layers that neither layer-ahead (R@10 0.47-0.64) nor
      caching predict. Routing is known as soon as the token (or a drafted token) is. Inference-only fallback:
      keep layers 0-2 resident (3 × 360 experts ≈ 2.7 GB at Q4).

- **2026-11-05, decision (user):** retrofits only, with no pretraining restart.
  - Expert offload follows the **practical recipe**:
    - one-layer-ahead prefetch (the free router-on-previous-layer baseline, top-10);
    - plus a recency cache;
    - substitute-on-miss;
    - MoE layers 0-2 resident.
  - R1 and R3-R7 change the model from scratch; they are out of scope for v4 and kept only as notes for a future
    model.
- **2026-11-05, DS-V4.1 Single-Pass mHC and CED vs ours** (report §2.2, §2.4.1):
  - **Our mHC is DS-V4's formulation** (A = h_pre, B = h_res Sinkhorn, C = h_post, all from X_l). Ours runs per
    sublayer (98 per forward) as two kernels: read (coefficients + mixing + input norm, 5d) and write (5d read,
    4d write), i.e. **(3n + 2)d = 14d per sublayer**.
  - **Single-Pass** feeds block l with A_{l-1}, so the write of l-1, the mixing of l and the coefficients of l
    become one pass: (2n + 2)d = 10d. That changes the function, so a retrofit would need fine-tuning.
  - **For us an exact fusion is likely possible** (to be verified with a kernel): one token's streams are only
    4 × 2048 bf16 = 16 KB. They can stay on-chip between the write of sublayer l-1 and the read of l, which
    removes the second read without the shift.
  - **Gain:** fewer launches in decode (196 → 98 mHC kernels), and ~5.6 → ~4 MB of activation traffic per token
    in prefill (~13 % → ~9 % of a 1M prefill on a 4090) [est.].
  - **CED retrofit:** the prefill skip requires that every cross-token path of the upper half come from the
    encoder output or a bounded window. For us that means 3 attention layers' KV **and 21 CKDA layers' k / v /
    β / gates** projected from H_24, plus a replay of the last few tokens for the short convolutions.
    - The gain would be ~45 % of prefill compute.
    - It is large surgery on the information flow, untested as a conversion (DS trained CED from scratch), and
      would need a cheap offline probe first (regress the upper layers' k / v / gates from H_24: R², ΔNLL), then
      an s3 conversion ablation.
    - Not before the sparse-attention path.

- **2026-11-05, CED retrofit probe** (`scripts/korbi/campaign_midtraining/ced_probe.py`, dev queue job
  48550946, 11 min; `docs/campaign_3b_hybrid_midtraining_results/ced_v4_2ph_8000/`):
  - **Setup:**
    - per-layer ridge maps from the 4 mHC streams entering sublayer 49 (8192 dims, 1.84M fit tokens) to the
      24 upper mixers' inputs (3 attention, 21 CKDA);
    - evaluated on held-out documents;
    - the predicted input is used **only on the state-writing path** (attention k / v; CKDA k / v / β / f), while
      queries and gates keep the true input.
  - **Held-out R²:**

    | target | R² |
    |---|---|
    | the input x itself | 0.97 → ~0.5, falling with depth |
    | attention k | 0.84-0.98 |
    | CKDA k | 0.80-0.94 |
    | v | 0.64-0.98 |
    | CKDA β | 0.92-0.98 |
    | CKDA f | 0.89-0.98 |

  - **ΔNLL, one layer substituted at a time (web_en):** mostly ≤ +0.002. The outliers are sublayer 56 (+0.016),
    60 (+0.005) and attention 66 (+0.004).
  - **ΔNLL, all 24 substituted together:** +0.05 (code) to +0.14 (reasoning); +0.10 web_en, +0.11 long_64k.
    - CKDA accounts for +0.04-0.11 of that, and the 3 attention layers for +0.006-0.009, or **+0.036 on long_64k**
      (retrieval).
    - The first and second halves of the upper mixers add roughly.
  - **Reading:**
    - A linear CED initialization keeps most of the state-writing signal: k, β, f are very predictable, v less so.
    - But the zero-training gap of ~0.1 nats would have to be healed by continued training. Long context is the
      most sensitive part.
    - Whether healing works, and how fast, needs an s3 conversion ablation (linear init from these maps → heal N
      tokens → loss and long-context retrieval vs the unconverted model). A small MLP map or jointly trained
      projections would start closer.
    - Priority stays behind the sparse-attention path.
- **2026-11-05, fused mHC kernel** (`experiment/014_fused_mhc/log.md`):
  - **Exact write(l-1) + read(l)** in 3 tiled kernels (write with partial coefficients + Gram, coefficients /
    Sinkhorn, mix); streams bitwise equal, x within 1-2 ulp.
  - **Decode critical path:** ~11 µs vs ~18-20 µs per sublayer (3090 Ti), mostly from moving Sinkhorn off the
    critical path (vLLM's split path already does that). Prefill is slower: keep the unfused path there.
  - **Not wired into the model yet.** Next: a faster Sinkhorn, vLLM decode-graph integration, epilogue / prologue
    fusion.
  - **Side finding:** at 16-256 tokens, cuBLAS's bf16 coefficient GEMM uses split-K with bf16 partials, so 40 % of
    the coefficients are mis-rounded vs fp64.

- **2026-11-05, budget and order (user):**
  - ~1k GPU-h for tests first, then a few thousand for C1 / C2.
  - **Proposal: sparsify first at 64k (C2), then extend sparse.**
    - The attention layers have no RoPE to adapt.
    - Sparse attention bounds the number of attended keys, so the softmax entropy no longer grows with length.
      That was the reason for log-n scaling beyond 8k.
    - Training beyond 64k is then cheap sparse training, mainly for the indexer and the CKDA states.
    - V4.1 likewise trains sparse at 64k and extends later.
    - Context parallelism is still needed for activations at ≥ 256k.
  - **Ablation:** sparse@64k → extend vs a short dense extension → sparsify. Zero-shot length generalization
    (128k-512k) of each sparse variant is the first measurement.
  - **Candidate variants:** dense; SWA-only (lower bound); page selection on mean-pooled keys (MoBA / InfLLM-v2
    style, no new parameters); a learned page indexer with KL warm-up; CSA2-like (compressed main KV m = 4 +
    indexer + SWA); DSA token top-k (selection-quality ceiling, offload-unfriendly); LSA. LSA is screened
    post-hoc (T2) before any training implementation.

- **2026-11-05, packed-sequence context parallelism survey** (read-only):
  - **FLA already does it for CKDA:** `fla.ops.cp`, varlen via `build_cp_context`, the sign gauge (`fla/ops/cp/sign.py`), β up to 2, short-conv halos. Tests in `flash-linear-attention/tests/context_parallel/test_cp_kda_layers.py`.
  - **Megatron glue is missing:**
    - the explicit reject at `complex_kda.py:206-210`;
    - `fla_cp.full_sequence` / `cp_context` assume unpacked samples, and `fla_cu_seqlens` uses the local token count.
    - Work: a THD per-document-zigzag ↔ FLA stream relayout (one all_to_all, computed once per micro-batch) and a
      CP context built from the document `cu_seqlens`, ~2-3 days.
  - **Engram is the real gap:** it gathers the full sequence under CP and rejects packed + CP
    (`engram.py:652-698`). A proper split (token-id halo, global positions, a conv halo) is ~2-3 days; a
    THD-aware full-gather fallback is ~0.5 day.
  - **Softmax attention:** TE 2.12 THD + CP works with p2p (ours) and a2a.
  - **Per-token parts:** mHC and MoE need nothing.
  - **Document boundaries:** they are snapped to multiples of 2·cp, which leaks ≤ 2cp-1 tokens per boundary into
    CKDA. The unsnapped `cu_seqlens` can be carried separately (optional).
  - **Speed:** under CP, FLA falls back to the `auto` KDA kernels, so expect slower CKDA than at CP=1.
  - **Tests:** CP=2/4 THD tests for ComplexKDA and Engram, plus a hybrid CP-vs-CP1 loss parity test, ~2 days.
  - **Total ~1-1.5 weeks.**

## References

- Soft-MTP: https://arxiv.org/abs/2610.38331 (code: github.com/JaydenTeoh/Soft-MTP)
- DFlash: https://arxiv.org/abs/2602.06036 (github.com/z-lab/dflash); DSpark: https://arxiv.org/abs/2607.05147
- DeepSeek-V4.1 report: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf
- LSA: https://arxiv.org/abs/2610.42147 (github.com/Prefrontal-AI/latent-sparse-attention)
- NSA: https://arxiv.org/abs/2502.11089; TreeWY (GDN speculative verification): https://arxiv.org/abs/2608.20961
