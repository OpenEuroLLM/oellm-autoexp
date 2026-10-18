# Model size for local / on-premises inference

**Status:** analysis, 2026-10-18 (revised the same day: target Qwen3.5-35B-A3B, Q4 or lower,
600k-1M GPU-h). Companion to
[`hybrid_moe_3b_campaign.md`](hybrid_moe_3b_campaign.md). The question: which total / active size
should the production hybrid MoE have so that people actually run it at home or on premises? A
second question: what should the political Qwen3.5-architecture baseline that runs in parallel
look like? Numbers for our shapes come from `scripts/korbi/campaign/inference_sizing.py`. It is
analytical, and calibrated to within ~1-3 % of Megatron's counts (m48a3 78.3B vs ~79B, A3B-30B
30.9B, Qwen3.5-35B-A3B 35.3B, Qwen3-Next-80B 81.1B). Market data comes from a web survey on
2026-10-18. Sources are at the end; "(agg)" marks aggregator or SEO pages, whose numbers are
approximate.

## 0. Answer

Revised 2026-10-18. The target is now set: beat Qwen3.5-35B-A3B on inference cost and quality
after the same number of tokens, at Q4 or lower. The budget is ~600k GPU-h on MN5, possibly
1M. The kernels and runtime integrations will be written in-house, so only the physical limits
count: FLOPs, dtypes, memory capacity and bandwidth. The first version of this doc recommended
~80B-A3B; that is superseded, and its analysis remains as the unified-memory option (§3, §6).

1. **Recommended: ~39B total / 2.7B active.** The shape:
   - 160 x 768 experts, top-5 + 1 shared: 32x routed sparsity, like Qwen's 256 / 8;
   - hidden 2048 x 48 blocks, CKDA 5:1, 1 MTP layer (`ours_e160k5` in the calculator);
   - active FFN per layer 6 x 768 = Qwen's 9 x 512, so the non-head active parameters are equal
     (2.1B vs 2.2B).

   It fits:
   - **24 GB at 4.25 bpw** (MXFP4 / IQ4_XS: 20.8 GB), with a 256k+ context: the KV at 256k is
     1 GB at fp8;
   - **32 GB at Q5_K_M** (27.9 GB), or at Q4_K_M with a 1M context;
   - **a 16 GB GPU + 32 GB RAM** with the experts offloaded;
   - anything from 64 GB upward at Q8.

   Against Qwen3.5-35B-A3B at the same quantization (computed, §3):

   | metric | ours vs Qwen3.5-35B-A3B |
   |---|---|
   | decode bytes per token | equal at short context, -25 % at 128k, -35 % at 256k, ~-50 % after the sparse-attention stage |
   | prefill FLOPs per token | -23 / -42 / -50 % at 32k / 128k / 256k |
   | KV at 1M (fp8) | 4.3 vs 10.7 GB |
   | weight memory | +11 % (48 layers and the 262k vocabulary) |

2. **Why not 80B any more.**
   - At Q4 the 80B does not fit the 24 / 32 GB GPUs where Qwen3.5-35B-A3B lives.
   - On 600k GPU-h it gets only ~6T tokens, against ~8.6T for the 39B (§6).
   - The campaign's sparsity curve is flat at the top: at s1, 32x -> 64x gains -0.008 nats at
     equal active size (§10.7). The extra tokens the 39B gets are worth about the same:
     Chinchilla's data term gives -0.010 nats for 6T -> 8.6T.

   In loss it is a wash; in deployment it is a clear loss.
3. **Do not cut the active size; raise the total only on evidence.**
   - **Active:** top-4 instead of top-5 (2.5B, -9 % active) saves only 7 % of the decode bytes,
     because the head, mixers and shared expert are fixed. By the s1 -> s2 dense slope (0.126
     nats per 3x parameters) it costs ~0.01 nats, about the whole 32x -> 64x sparsity gain.
   - **Total:** go to 192 experts (46.6B; 32 GB at Q4_K_M, 24 GB only at ~3.5 bpw) only if round
     4's s2 ladder shows more than 0.01 nats for 32x -> 48x.
4. **Quality against Qwen at equal tokens.** The inference win is certain. The loss margin comes
   from the residual stream and the optimizer, not from the mixer: GDN 3:1 beats CKDA 3:1 by 0.012
   under AdamW at s1 (§5). Pre-register the comparison with a `qwen35_moe` arm in round 4 (s1 +
   s2, ~30 GPU-h) before committing the baseline run.
5. **Below 4-bit.** Ship a 4-bit QAT checkpoint (MXFP4 or NVFP4 experts; attention, CKDA, the
   shared expert and the head at 8 / 6 bit). Whether 768-wide experts quantize better than
   Qwen's 512 is open: Gemma 4 26B-A4B argues for it, Unsloth's Qwen3.5 study (512 experts fine
   at 3-bit) against (§8.4). Test it at s2: PTQ and QAT KL divergence for both architectures. 2-bit is not a target; on 16 GB GPUs, CPU
   offload at Q4 beats IQ2.
6. **Training shape** (§8): 8k sequences with document masking (CKDA varlen still to be wired);
   a constant GBS of 32M tokens (user decision; the b*(D) law gives 17-34M for an 8.6T
   budget); PP2-4 x VPP2. Expert width 768 vs 512 (G 5.3 vs 8, equal total)
   is decided by an s2 loss pair and an A3B speed pair.
7. **Unchanged:**
   - one MTP layer;
   - the sparse-attention stage (InfLLM-v2-style block selection fits our GQA-2 layout);
   - a 6-8-bit head;
   - CKDA kernels for llama.cpp / MLX / vLLM, now in-house. The physical constraints there are a
     sign-aware chunked decay instead of KDA's log-space cumsum, and fp32 state.

## 1. Hardware people use (prices October 2026)

The 2026 DRAM shortage raised every memory-heavy option 1.5-4x over its launch price.

| Device | Memory | BW (GB/s) | FP4 | Street price, Oct 2026 | Notes |
|---|---|---:|---|---:|---|
| RTX 3060 12 GB / 16 GB cards + 32-64 GB DDR5 | 12-16 + RAM | ~360 / 70-90 (RAM) | 50-series only | (installed base) | **The most common setup**: RTX 3060 12 GB is the top single GPU on HF; 16 GB is the top VRAM tier on Steam |
| RTX 3090 / 4090 | 24 GB | 936 / 1008 | no | ~$1.2k used / ~$2.3-3k | 3090: +24 % over its 12-month average |
| RTX 5090 | 32 GB | 1,792 | NVFP4 | MSRP $1,999; street ~$4.3-5.3k | |
| Radeon AI PRO R9700 | 32 GB | 640 | no | ~$1.3-1.5k | |
| RTX PRO 6000 Blackwell | 96 GB | 1,792 | NVFP4 | $16,000 (launch $8,565) | |
| **DGX Spark (GB10)** | 128 GB unified | 273 | NVFP4 | **$6,950** (launch $3,999; $4,699 in Feb) | 2x via ConnectX-7 200 Gb (~111 Gb/s measured RoCE) |
| DGX Spark 64 GB | 64 GB | 273 | NVFP4 | $4,999 (ships 2026-10-23) | new SKU |
| GB10 OEMs (Dell, HP, Lenovo, ASUS, MSI) | 128 GB | 273 | NVFP4 | ~$5.7-8.2k | |
| Strix Halo (Ryzen AI Max+ 395) | 128 GB | 256 (~212 measured) | no | ~$3.4-3.5k (launch $1,999) | prefill ~5x slower than Spark |
| Mac Studio / MBP M5 Max | up to 128 GB | 614 | no (MLX dequant) | ~$2.5k + / ~$4.5k for 128 GB | |
| **Mac Studio M5 Ultra** | 96-512 GB | 1,200 | no | from $5,499; 256 GB $9.5-10.8k; 512 GB price TBA | TB5 RDMA clustering |
| 2x DGX Spark | 256 GB | 2 x 273 | NVFP4 | ~$14k | the user's setup |
| DGX Station (GB300) | 252 GB HBM3e + 496 GB LPDDR5X | 7,100 (HBM) | NVFP4 | ~$85-115k | out of the "at home" class |

Takeaways:

- **Bandwidth sets decode speed; capacity sets total size.** The unified-memory boxes (Spark,
  Strix, Mac) have plenty of capacity and little bandwidth, which is ideal for high-sparsity MoE
  with few active parameters. On discrete GPUs, a ~3B-active model is not bandwidth-bound at all:
  Qwen3.5-35B-A3B decodes at 194 tok/s on a 5090, ~22 % of the bandwidth ceiling, so the
  bottleneck is launch / latency overhead.
- **128 GB is the enthusiast standard** (Spark, Strix, M5 Max). 256 GB costs $10-14k now.
  Anything above that, or a 96 GB GPU, is a business purchase.
- **NVFP4 runs natively only on Blackwell.** Apple, AMD and Intel dequantize 4-bit in software,
  and GGUF k-quants are the lingua franca there.

## 2. What people run, and how fast

| Tier | De-facto model(s) | Measured (decode / prefill tok/s) |
|---|---|---|
| 24 GB | Qwen3.6-35B-A3B Q4_K_M (~20 GB + q8 KV); Qwen3.6/3.8-27B dense Q4 | 35B-A3B: 3090 ~145-158, 4090 ~120-170 (agg) |
| 32 GB (5090) | Qwen3.5/3.6-35B-A3B Q4; Qwen3.8-27B Q4 / NVFP4 | 35B-A3B Q4_K_XL: 194 / 6,960; 27B: 78 / 3,000-5,100 |
| 12-16 GB + 64 GB DDR5 | gpt-oss-120b, Qwen3.6-35B-A3B with `--n-cpu-moe` | gpt-oss-120b on an RTX 4070 + DDR5-6000: 25 / 428 at 32k; decode tracks RAM bandwidth |
| 128 GB unified | **gpt-oss-120b MXFP4** (117B-A5.1B, 59 GiB) is the default; Qwen3.5-122B-A10B is "the smart one"; Mistral Small 4 (119B-A6.5B); Nemotron 3 Super (120B-A12.7B) | Spark gpt-oss-120b: ~60 / ~2,000 at 0 depth, ~40 at 32k. Spark Qwen3.5-122B: 17 (vLLM NVFP4) to 51 (INT4 + MTP). Spark Qwen3.6-35B-A3B NVFP4 + MTP: 75-102. Strix gpt-oss-120b: 34-56 / ~340. M5 Max gpt-oss-120b: 88 at 4k, 65 at 32k |
| 256 GB (2x Spark, M5 Ultra 256) | Qwen3.5-397B-A17B at INT4 / 6-bit; DeepSeek V4.1-Flash at ~3 bpw | 397B: 27 (2x Spark) vs 29 (Mac). V4.1-Flash on 2x Spark: 30-41 / 1,000-2,050, up to 1M ctx |
| 512 GB Mac | DeepSeek V4-Flash q4, V3 q4 | ~35 / ~20 |

- **Dense 70B is dead locally.** Llama-3.3-70B AWQ runs at 6 tok/s on a Spark, while MoEs with
  3-12B active run at 20-100.
- **Long-context decode degradation**, measured on Strix Halo (4k -> 128k):
  - Qwen3-Next-80B-A3B: 55.6 -> 29.1 (52 % retained);
  - Qwen3.6-35B-A3B: 57.8 -> 32.2 (56 % retained).

  On Spark at 32k, full-attention Qwen3-Coder-30B loses 37 % and gpt-oss-120b 31 %. With ~2 GB of
  weights read per token, even the hybrids' 20-25 KB/token of KV dominates the traffic by 128k.
- **Prefill is the complaint on Strix / Spark.** Time to first token, not decode speed, is "what
  feels slow" for agents and RAG. Qwen3.8-Flash-Next took 17 minutes to first token for a
  200k-token prompt on Strix Halo: a tiny KV cache does not fix compute-bound prefill.
  DeepSeek V4.1-Flash's causal encoder-decoder (8B active in prefill, 16B in decode, decoder KV
  projected from the encoder) attacks exactly this. We will not copy that split (user decision).
  Our levers are the 5:1 ratio, the small attention width, and the sparse-attention stage (§3).
- **Quantization practice:**
  - **Formats by platform:** GGUF Q4_K_M / UD-Q4_K_XL / IQ4_XS everywhere; NVFP4 / AWQ /
    INT4-AutoRound on Spark (AutoRound is often faster than NVFP4 there); EXL3 at ~3 bpw on
    multi-Spark setups; MLX at 4-6 bit on Mac.
  - **Large MoEs are near-free at 3-4 bit.** DeepSeek V3.1 Aider Polyglot: 71.6 at full precision,
    69.7 at 4-bit, 68.4 at 3-bit, 65.8 at 2-bit.
  - **Qwen3.5 sensitivity:** experts tolerate 3-bit, while `attn_*` and the GDN output
    projections are "especially sensitive", and MXFP4 did worse than Q4_K on many tensors.
  - **Small experts are a 4-bit risk.** Gemma 4 26B-A4B ships no 4-bit QAT checkpoint because
    its small experts lose too much at 4-bit (Google recommends INT8). Our 768-wide experts sit
    in that range, so plan QAT rather than relying on post-training quantization.
  - **Native QAT releases are now expected:** gpt-oss MXFP4, Kimi K2 INT4, Nemotron 3 Ultra
    NVFP4.
  - **KV quantization:** q8_0 KV is the default. q4_0 KV is reported as near-lossless on
    hybrids (single report).
- **MTP / speculative decoding is now expected.**
  - Qwen3.5/3.6/3.8, Nemotron 3 Super, GLM, MiniMax-M2 and DeepSeek V4 all ship MTP.
  - llama.cpp has merged MTP (May 2026) and EAGLE-3.
  - Measured on a 3B-active hybrid MoE (Qwen3.6-35B-A3B on an RTX 3090): MTP with 2 drafts gives
    +17-27 %; external draft models are -75 %. Spark reports are higher (51 vs 17 tok/s for the
    122B with INT4 + MTP).
  - The gain on MoE is modest because each draft token can route to other experts, raising the
    verification traffic.
  - On Apple Silicon, 3 of 5 speculative setups slowed down.

## 3. Our shapes: memory, KV, decode, prefill

All shapes: hidden 2048, 48 blocks, CKDA 5:1 (40 CKDA layers of 16 x 128 x 128, 8 attention
layers of 16 x 128 q / 2 x 128 KV), untied 262k vocab.

**Memory and cache:**

| shape | total / active (B) | 4-bit (Q4_K_M) | 8-bit | KV bf16 / fp8 per token | KV at 1M (fp8) | state per sequence |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B arch | 35.3 / 2.7 | 21 GB | 37 GB | 20 / 10 KB | 10.7 GB | 61 MB |
| Qwen3-Next-80B-A3B arch | 81.1 / 3.3 | 49 GB | 86 GB | 24 / 12 KB | 12.9 GB | 74 MB |
| ours 30B-A3B (128 x 768, top-8) | 30.9 / 3.2 | 19 GB | 33 GB | **8 / 4 KB** | **4.3 GB** | 41 MB |
| **ours e160k5 (160 x 768, top-5 + 1, MTP; recommended)** | **39.2 / 2.7** | **24 GB** (MXFP4 21) | **42 GB** | 8 / 4 KB | 4.3 GB | 41 MB |
| ours e192k5 (192 x 768, top-5 + 1, MTP) | 46.6 / 2.7 | 28 GB (MXFP4 25) | 49 GB | 8 / 4 KB | 4.3 GB | 41 MB |
| ours m32a3 (224 x 768, top-7 + 1) | 52.9 / 3.2 | 32 GB | 56 GB | 8 / 4 KB | 4.3 GB | 41 MB |
| ours m48a3 (336 x 768, top-7 + 1; first-version pick) | 78.3 / 3.2 | 47 GB (MXFP4 42) | 83 GB | 8 / 4 KB | 4.3 GB | 41 MB |
| ours m64a3 (448 x 768, top-7 + 1) | 103.6 / 3.2 | 63 GB | 110 GB | 8 / 4 KB | 4.3 GB | 41 MB |
| ours ~120B-A5B (hidden 2560) | 129.8 / 5.0 | 79 GB | 138 GB | 8 / 4 KB | 4.3 GB | 52 MB |

For comparison, KV per token at bf16 for other models:

| model | KV per token (bf16) |
|---|---:|
| gpt-oss-120b | 36.9 KB |
| GLM-4.5-Air | 188 KB |
| MiniMax-M2 | 254 KB |
| Kimi Linear 48B-A3B | 8.1 KB |
| Nemotron 3 Nano | 6.1 KB |
| DeepSeek V4.1-Flash (FP4, cross-layer reuse) | 0.89 KB |

Ours is at the efficient end of the field: 1/2.5 of Qwen3.5 per token.

The decode and prefill tables below were computed for the 3.2B-active shapes of the first
version. The recommended e160k5 (2.7B active) is compared directly with Qwen3.5-35B-A3B in §5.

**Decode bytes per token at Q4_K_M (GB).** Attention, the LM head and the router are counted at
8-bit:

| shape | weights | KV, 32k (dense, fp8) | KV, 128k (dense) | KV, 256k (dense) | KV, 128k (DSA top-2048, indexer included) | recurrent state, read + write |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Next-80B arch | 2.15 | 0.40 | 1.61 | 3.22 | 0.23 | 0.16 |
| ours m48a3 | 2.19 | 0.13 | 0.54 | 1.07 | 0.15 | 0.09 |

- **The 262k untied LM head (0.54B) is ~25 % of the per-token decode bytes at 4-bit**: 0.57 GB
  at 8-bit out of 2.19 GB. That is the price of the multilingual vocabulary. Ship the head at
  Q6 rather than Q8, and keep this in mind if the vocabulary is ever reconsidered.
- **The recurrent state** (fp32, read and written each step) is ~4 % of the bytes and is
  irrelevant for single-stream use. For batched serving, 41 MB per sequence equals the KV of
  ~5k tokens at bf16.

**Decode ceiling, single stream** (bandwidth / bytes, 32k / 256k context, dense attention, no
MTP). Measured efficiency is ~0.8 of the ceiling on Spark with llama.cpp (gpt-oss-120b and
Qwen3-Coder-30B both land at ~0.8) and ~0.55-0.9 on Strix Halo:

| shape @ quant | Spark 64 | Spark 128 | Strix 128 | M5 Max 128 | M5 Ultra 256 | 2x Spark (TP2) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Next-80B @ Q4 | 101 / 49 | 101 / 49 | 78 / 38 | 227 / 111 | 444 / 217 | 202 / 99 |
| ours m48a3 @ Q4 | — (*) | 113 / 82 | 88 / 63 | 255 / 183 | 498 / 358 | 227 / 163 |
| ours m48a3 @ Q8 | — | 75 / 60 | 58 / 46 | 169 / 134 | 331 / 263 | 151 / 120 |
| ours m48a3 @ Q4 + DSA | — (*) | 113 / ~110 | 88 / ~85 | | | |

(*) At MXFP4 (42 GB) m48a3 fits the 64 GB Spark with a 256k context; the calculator's Q4_K_M
(47 GB) plus its 55 GB usable budget is borderline.

- **Estimated real speed on one Spark:** ~90 tok/s at 4-bit / short context. At 256k it is ~65
  dense and ~85 with the sparse stage. That compares with ~80 -> ~40 for Qwen3-Next-80B and
  ~60 -> ~40 for gpt-oss-120b. MTP adds +20-50 % on top.
- **The architecture argument at long context:** 1/3 of the KV traffic of the Qwen
  architecture means **+67 % decode speed at 256k on the same box** before any sparse attention.

**Prefill FLOPs per prompt token** (prompt length L, mean position L/2):
2 (N_active - N_head) + 4 n_attn d_q L/2. This leaves out the linear-mixer chunk FLOPs, which
are smaller for our 16-head CKDA than for Qwen's 32-value-head GDN:

| L | ours (8 attention layers x 2048) | Qwen3.5-35B (10 x 4096) | Qwen3-Next-80B (12 x 4096) |
|---:|---:|---:|---:|
| 32k | 6.4 GF | 7.1 | 9.2 |
| 128k | 9.6 GF | 15.1 | 18.9 |
| 256k | 13.9 GF | 25.9 | 31.7 |

So at long prompts our prefill needs 0.5-0.6x the compute of the Qwen architecture at a similar
size, which roughly halves time to first token on compute-bound boxes (Strix Halo, Spark). With sparse attention, the attention term (8.6 GF at 256k) drops
to ~0.1 GF plus the indexer. The caveat is that the CKDA kernels themselves must be fast
on consumer runtimes. On H100 the speed suite has KDA / CKDA at 0.66-0.67x of attention at small
scale, which is irrelevant at A3B training scale (the MoE dominates) but matters for the prefill
kernels.

## 4. The NSA / DSA stage: what it buys, how to do it

- **Our KV is not memory-bound.** 1M context costs 4.3 GB at fp8. The stage buys **decode
  bandwidth** (the dense KV read at 256k is ~50 % of the 4-bit weight read) and **prefill
  attention FLOPs**. Storage stays: DSA keeps the full cache and only reads less.
- **The indexer is the new long-context cost.**
  - DSA's indexer key (fp8, 128-dim, ~132 B per layer-token) is +13 % storage on top of our
    1,024 B bf16 KV per layer-token.
  - At 128k it is the dominant read: 0.14 GB of the 0.15 GB, so the reduction is capped at
    ~3.7x.
  - DeepSeek V3.2 has the same problem: 1.0 GB indexer vs 0.14 GB selected KV at 128k.
- **For our GQA-2 x 128 layout, an NSA / InfLLM-v2-style block selection fits better than
  token-level DSA.** InfLLM-v2 (MiniCPM4):
  - adds no new parameters and reuses the dense KV;
  - scores block-mean keys over blocks of 64, so the "indexer" read is 1/64;
  - shares the selection within a GQA group;
  - converted in **5B tokens** at 98-99.7 % of dense quality, with 2.1-2.3x end-to-end prefill
    and decode speedups.

  If DSA is used, use an FP4 indexer (as DeepSeek V4 does) or d_I = 64.
- **Conversion budget**, from published recipes:
  - indexer warm-up by KL against dense attention with the model frozen: ~2-3B tokens (DeepSeek
    V3.2 2.1B, GLM-5 2.8B);
  - joint sparse training: 5-20B tokens (InfLLM-v2 5B, GLM-5 20B; DeepSeek's 944B is an outlier).

  That is <1 % of the 10T run.
- **Order:** after long-context extension, and before RL. RL needs a deterministic top-k and a
  frozen indexer, as in GLM-5. GLM-5's ablation on GLM-9B at 128k RULER: DSA near-lossless, a
  searched sliding-window pattern -5.7, GDN linearization -8 to -11. So the sparse stage should
  touch the attention layers only, which matches the plan.
- DeepSeek V4's compressed-plus-sparse attention (CSA ×4 + HCA ×128, ~2 % of a GQA8 baseline)
  and V4.1's cross-layer KV reuse (CSA2) are the next step if we ever want sub-KB KV. With 8
  attention layers at 1 KB each there is little left to gain there.

## 5. The political baseline: Qwen3.5-35B-A3B architecture

**The architecture.** Qwen3.5 / 3.6-35B-A3B:
- 40 layers;
- GDN : gated attention = 3:1;
- attention 16 q x 256 / 2 KV x 256, partial RoPE;
- GDN 16 qk + 32 v heads x 128;
- 256 x 512 experts, top-8 + 1 shared;
- zero-centred norms, 1 MTP layer.

Train it on our data, data order and tokenizer (262k instead of Qwen's 248k: +0.05B
embedding, so the comparison is about the architecture). Run the same token count for the
overlap, with paired evaluations at equal D.

**Expected loss gaps, ours vs Qwen3.5-arch at equal tokens** (nats; negative = ours better). These
are small-scale numbers from the campaign (s1 dense or MoE, 2-4BT). Hybrid gains shrank from s1
to s2, so read them as a direction, not a forecast:

| lever | both AdamW | both angular MuOwn | source |
|---|---:|---:|---|
| block: CKDA 3:1 + NoPE / XSA / gate attention vs GDN 3:1 + gated attention | +0.012 | -0.027 | campaign §10.5 (h3 vs qwen35, dense s1) |
| 5:1 instead of 3:1 | +0.005 to +0.007 | +0.005 to +0.007 | §10.5 |
| residual stream: routed-4 / mHC-4 (mixer lr x 0.25 under MuOwn) | -0.052 / -0.051 | -0.022 / -0.048 | §10.7 |
| total 39 vs 35B at equal active and sparsity | ~0 | ~0 | |
| **net (routed residual)** | **~-0.035** | **~-0.040** | |
| optimizer, if the baseline keeps Qwen's AdamW: angular MuOwn vs AdamW | -0.07 to -0.08 | | §10.5 (MoE s1) |

- **Same optimizer.** The win rests on the residual stream; without it, ours is behind under
  AdamW. This is why the `qwen35_moe` pre-registration matters. Round 4 has `attn_moe` and
  `h5_moe` but no Qwen arm. Add it at s1 and s2, under both optimizers if the budget allows
  (~30-60 GPU-h).
- **Which optimizer for the baseline.**
  - **Recommended: Qwen's recipe class (AdamW, tuned by our lr / bs laws).** It is the fair
    "their recipe" baseline, and nobody can say we mistuned it under an optimizer it was not
    designed for.
  - The attribution then comes from the s2 pairs: architecture-only under the same optimizer,
    and optimizer-only.
  - Training both with angular MuOwn makes an architecture-only claim, but GDN under MuOwn has
    not had its own lr tuning: the residual-mixer lr lesson of §10.7 may apply.
- **Length:** 1T tokens (~41k GPU-h, estimated), compared at equal D every ~100B tokens. Extend
  to 2T only if the gap is inside its noise (data-order noise ±0.004 at s1).

**Inference comparison at Q4_K_M** (computed by the calculator; to be measured on Spark / 5090
once both exist):

| | Qwen3.5-35B-A3B arch | ours e160k5 |
|---|---:|---:|
| total / active | 35.3 / 2.73B | 39.2 / 2.74B |
| weights Q4_K_M / MXFP4 / 3.5 bpw | 21.4 / 18.7 / 15.4 GB | 23.7 / 20.8 / 17.2 GB |
| KV per token, bf16 | 20 KB | 8 KB |
| decode bytes per token at 4k / 128k / 256k | 2.06 / 3.36 / 4.70 GB | 2.01 / 2.53 / 3.06 GB |
| the same, with the sparse stage (DSA-like top-2048) | — | ~2.2 / ~2.3 GB at 128k / 256k |
| prefill GFLOP per token at 32k / 128k / 256k | 7.1 / 15.1 / 25.9 | 5.5 / 8.7 / 13.0 |
| decode ceiling on Spark 128 GB, 32k / 256k | 116 / 58 tok/s | 128 / 89 tok/s |
| largest context on a 24 GB card (4-bit weights) | ~150k (Q4_K_M + fp8 KV) | >1M (MXFP4 + fp8 KV) |

## 6. Recommendation, budget and training plan

**Training cost** (MN5, bf16; tok/s/GPU from §7.2 / §11.1 of the campaign doc; the 39B and
Qwen shapes are estimates to confirm with a 25-step speed arm each):

| shape | tok/s/GPU | GPU-h per 1T tokens | basis |
|---|---:|---:|---|
| Qwen3.5-35B-A3B arch (no residual stream) | ~6.5-7k (est.) | ~41k | A3B-30B recipe: 6,908 |
| **ours e160k5, routed residual** | ~5.0-5.3k (est.) | **~54k** | A3B-30B + routed-8: 5,234; m48a2: 5,359 |
| ours e160k5, no residual stream | ~6.9k | ~40k | |
| ours m48a3 (80B) | 3.4-3.9k | 72-81k | |

The 39B should fit the 8-node PP4 x EP4 layout, as the 31B shapes do at 65-82 % memory. The 47B
probably needs PP8 and 16 nodes, as the 53B does, which is one more reason to stay at ~40B on
MN5's 64 GB GPUs.

**Budget split** (GPU-h):

| item | 600k | 1M |
|---|---:|---:|
| remaining architecture decisions (campaign cap 10k) | 7k | 7k |
| Qwen3.5-arch baseline, 1T tokens | 41k | 41k (or 2T: 82k) |
| long-context extension + sparse-attention stage + QAT | ~25k | ~25k |
| SFT / RL (placeholder) | ~30k | ~60k |
| reserve (restarts, failures), ~5 % | 30k | 50k |
| **pretraining, ours e160k5** | **~467k -> ~8.6T tokens** | **~817k -> ~15T tokens** |

- **Engineering upside.** Throughput levers would give ~+25 %: VPP for the routed residual (the
  hang of §7.2), internode HybridEP (EP8), fused routed-residual kernels. That is ~10.7T tokens
  on 600k and ~19T on 1M. With AI-written kernels these are cheap to attempt and the largest
  return per hour of work.
- **If the extension comes**, more tokens for the same model is the safe use. D/N_active is
  ~3,200 at 8.6T; DeepSeek V4-Flash trained 13B active on 32T (~2,500), and Qwen-class small
  models go far beyond. The alternative is a second size for the 128 GB tier (~120B-A5B, §3:
  4-bit 69-79 GB, Spark ~60 tok/s), the most-requested gap in the market. One run at
  10T x 5B active costs roughly 1.7x the 39B's GPU-h per token, more with EP overhead, so it
  does not fit next to a full 39B run even at 1M. It is a next-campaign item.
- **Larger, if the data say so.** The round-4 s2 sparsity ladder decides between 160 experts
  (39B) and 192 (47B). The threshold is a gain of >0.01 nats for 32x -> 48x at equal active
  size, enough to pay for the 24 GB tier dropping to 3.5 bpw and the 16-node layout.

## 7. Open points

1. Round 4 additions:
   - a `qwen35_moe` arm at s1 and s2, AdamW and angular MuOwn, paired with `h5_moe` + routed
     residual;
   - the s2 sparsity ladder at fixed active size (32x / 48x);
   - expert width 512 vs 768 at equal active FFN, for loss, PTQ / QAT robustness and speed.
2. 25-step MN5 speed arms: e160k5 with the routed residual (PP4 x EP4, 8 nodes) and the
   Qwen3.5-35B-A3B architecture.
3. PTQ KL divergence (Q4_K_M, MXFP4, Q3_K_M) of an s2 MoE checkpoint for both architectures:
   are 768-wide experts more robust than 512?
4. CKDA kernels: chunked prefill and decode with fp32 state on CUDA (sm_121 GB10, sm_120 5090),
   Metal and Vulkan. Then decode / prefill on Spark and a 5090 against Qwen3.5-35B-A3B.
5. The baseline's optimizer (recommended: AdamW, Qwen's recipe class) and length (1T, extend if
   unresolved).
6. Prices move monthly (DRAM shortage). Re-check before positioning the release.

## 8. Training shape: sequence length, batch size, expert granularity (2026-10-18)

Setting: 128-256 MN5 nodes (512-1,024 H100), a GBS of ~32M tokens, angular MuOwn, and the
e160k5 shape (§0).

### 8.1 Sequence length: 8k, yes

- **Cost: ~+2.5 % FLOPs vs 4k.**
  - Per token, the forward attention FLOPs of the 8 attention layers are 4 x 8 x 2048 x L/2:
    0.13 GF at 4k, 0.27 GF at 8k, against ~5.5 GF for the rest of the model.
  - CKDA is linear in L, so the chunked kernels cost the same per token.
  - Activation memory per microbatch depends on its token count, not its sequence count.
- **Benefits:**
  - The CKDA layers learn to carry state over spans twice as long. A 5:1 hybrid relies on its
    recurrent state between the sparse attention layers, so this matters more than for a
    transformer.
  - The long-context extension later needs fewer tokens.
  - A 32M GBS becomes 4,096 sequences instead of 8,192.
- **Common practice:** Llama 3 pretrained at 8k. DeepSeek-V3 and Qwen3 used 4k for the bulk,
  then 32k.
- **Document masking. This needs implementation work.**
  - Recommended: masked attention within each document, plus a CKDA state reset at document
    boundaries (varlen / `cu_seqlens`). Otherwise the state at 8k runs across unrelated
    documents, a regime that never happens at inference. Without masking, 8k mainly helps the
    few documents longer than 4k (code, books, math), so upsample those.
  - Today `complex_kda.py` and `fla_hybrid.py` raise on `packed_seq_params`. The FLA chunk
    kernels take `cu_seqlens`, so the work is threading the THD / packed metadata through the
    Megatron adapters, the attention (TE THD) and CP.
  - Test that a packed sequence equals the documents run separately: loss and every gradient
    must match.
  - The small-scale screens used 4k without masking (`reset_attention_mask: false`). Changing
    both is a regime change, so do a quick s1 check: 4k vs 8k, masked vs unmasked, at equal
    tokens and GBS in tokens, with validation on fixed 4k windows (~10-20 GPU-h).

### 8.2 GBS 32M (4,096 x 8k): decided, constant

User decision 2026-10-18: constant GBS of 32M tokens. The campaign's b*(D) (§7.8 / §7.9) is the
loss-optimal *constant* batch for a run with total budget D: b* ≈ 0.26-0.52M tokens at 1BT,
∝ D^0.46. For an 8.6T-token run that gives **b* ≈ 17-34M**, so 32M is inside the extrapolated
optimum for the whole-run budget.

An earlier draft of this section applied the law step by step, as an instantaneous b* at the
tokens seen so far, and argued for a batch ramp. That reads the measurement as something it is
not: it was measured with a constant batch per run. Caveats that remain:
- the extrapolation spans four orders of magnitude in D, from an 85M model;
- b* may also grow with N.

The s2 check at two token budgets (~100-150 GPU-h) can still pin the exponent, but it does not
block the decision.

Large constant GBS also suits the layout: 128k tokens per dense-DP rank at PP4 on 1,024 GPUs
(16 microbatches of 8k), so deep PP is cheap (§8.3).

### 8.3 Pipeline parallelism at 32M GBS

- 39B is small for 1,024 GPUs: PP serves throughput and per-GPU memory, not fit.
- Tokens per dense-DP rank per step = 32M / (1,024 / PP):

  | layout | per DP rank | microbatches (8k, mbs 1) | bubble |
  |---|---:|---:|---:|
  | PP4 | 128k tokens | 16 | ~9 % (VPP2) |
  | PP2 | 64k tokens | 8 | ~6 % (VPP2) |

  PP1 needs EP ≥ 16, i.e. internode HybridEP, which is broken on MN5 (`internode_doca.cu:124`).
- **Speed arms:** PP2 x VPP2 and PP4 x VPP2 at EP4, 8k, mbs 1 / 2, full vs selective
  recompute. Large GBS makes deeper PP cheap, so the best layout is whichever frees the most
  memory for mbs 2 / selective recompute.
- **The routed residual's VPP hang (§7.2) becomes the blocker here.** Fix it before the
  production layout is fixed.

### 8.4 Why 768-wide experts, and should we go finer (more experts)?

- **Why 768:** it was inherited. The width comes from the speed-tuned A3B-30B shape
  (Qwen3-30B-A3B's 128 x 768, exp_irene's MN5 layout). It is not a loss decision.
- **In granularity terms** (G = 2 x hidden / expert width):

  | model | G |
  |---|---:|
  | ours (768) | 5.3 |
  | Qwen3-30B-A3B | 5.3 |
  | **Qwen3.5-35B-A3B (512)** | **8.0** |
  | DeepSeek-V3 / Kimi K2 (2 x 7168 / 2048) | 7.0 |
  | gpt-oss | 2.0 |

- **Our s1 data** (round 3B, hidden 768, equal active, §10.5 / §10.7) tested G = 1.5 / 3 / 6:
  - G 1.5 is worse than 3 by 0.01-0.02;
  - G 6 beats 3 by 0.012 under AdamW and by 0 under angular MuOwn at 16x, and by 0.005 at 64x.

  So G 5.3 is already in s1's "fine" range, and 5.3 -> 8 should be worth ≤ ~0.005, possibly 0
  under angular MuOwn.
- **The literature says the optimal granularity grows with compute** (fine-grained MoE scaling
  laws; DeepSeek, Kimi and Qwen all at G ≈ 7-8). At a 10T-token budget G ≈ 8 is the likelier
  optimum.
- **"More experts" at fixed memory means finer experts.**
  - E240 x 512, top-8 + 1 shared: 39.1B / 2.74B. Same total, active and memory as e160k5
    (MXFP4 20.8 GB).
  - E256 x 512: Qwen's exact expert layout, 41.6B; MXFP4 22.1 GB is tight on 24 GB. EP-friendly
    up to 64.

  More experts at the same width means more total, which the 24 GB-at-4-bit budget does not
  allow (§0).
- **Costs of G 8:**
  - 1.6x the routed dispatch volume per token (top-8 vs top-5);
  - smaller grouped GEMMs (2048 x 512);
  - more router / permute work.

  At s1, G 3 -> 6 cost 21-32 % throughput, but s1 is too small to be representative. At A3B it
  is untested in a clean pair. At 1,024 GPUs the all-to-all is the most likely bottleneck, so
  this is a real cost.
- **Inference:**
  - Decode bytes are identical, and CPU offload does not care.
  - MTP verification and batched serving touch more distinct experts per step, which is mildly
    worse for fine experts.
  - **Correction to §0.5:** Unsloth's Qwen3.5 study found its 512-wide experts tolerate 3-bit.
    "768 quantizes better" is therefore not established; the s2 PTQ test (§7.3) decides it.
- **Decision rule:**
  - **Loss:** E240w512k8 vs e160k5 at s2, equal active and total (~40 GPU-h with the
    `qwen35_moe` arm).
  - **Speed:** both at A3B, same layout, 8k (25 steps, ~10 GPU-h).
  - Take G 8 if s2 gains ≥ 0.004 (2x the data-order noise) and A3B speed is ≥ 0.9x. Otherwise
    keep 768.
  - My prior: about equal loss under angular MuOwn and 10-20 % slower, so 768 stays unless the
    data say otherwise. If the a2a overlap hides the dispatch at scale, G 8 is the safer bet for
    10T tokens.

## Sources (accessed 2026-10-18)

Hardware and prices:
- DGX Spark price, 64 GB SKU: [The Register 2026-10-02](https://www.theregister.com/systems/2026/10/02/nvidia-debuts-4999-dgx-spark-with-half-the-ram-and-storage-amid-memory-crunch/5300622), [HWBusters](https://hwbusters.com/news/nvidia-dgx-spark-64gb-arrives-at-4999-as-the-128gb-model-jumps-to-6950/), [Tom's HW (Feb hike)](https://www.tomshardware.com/desktops/mini-pcs/nvidia-dgx-spark-gets-18-percent-price-increase-as-memory-shortages-bite-founders-edition-now-usd4-699-up-from-usd3-999), [pi3g OEM prices](https://pi3g.com/nvidia-dgx-spark-price/), [NVIDIA spec](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- RTX PRO 6000 at $16k: [TechPowerUp](https://www.techpowerup.com/351549/nvidia-rtx-pro-6000-blackwell-96-gb-gpu-now-costs-usd-16-000), [VideoCardz](https://videocardz.com/newz/nvidia-raises-rtx-pro-6000-blackwell-price-to-16000-now-87-above-original-msrp)
- DGX Station: [Tom's HW](https://www.tomshardware.com/desktops/nvidias-gb300-powered-dgx-station-desktop-tower-listed-for-nearly-usd100-000-online-enterprise-ai-powerhouse-now-available-to-buy-for-mere-mortals-with-lots-of-cash), [flopper](https://flopper.io/desktop/nvidia-dgx-station-gb300)
- Mac Studio M5 Ultra: [Apple 2026-08](https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/), [MacStories review](https://www.macstories.net/stories/m5-ultra-mac-studio-review-the-dream-mac-for-local-ai-agents/)
- Strix Halo: [llm-tracker](https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance), [compute-market (agg)](https://www.compute-market.com/blog/strix-halo-mini-pc-local-ai-2026), [soothill.io benchmarks](https://www.soothill.io/blog/2026/08/03/llamacpp-vulkan-vs-rocm-strix-halo/), [prefill / TTFT](https://datahardware.ai/blog/strix-halo-prefill-time-to-first-token-2026)
- RTX 5090 street price: [tech-insider (agg)](https://tech-insider.org/rtx-5090-shortage-price-surge-2026/)
- Used 3090 / 4090: [bestvaluegpu](https://bestvaluegpu.com/history/new-and-used-rtx-3090-price-history-and-specs/), [dealhunter](https://dealhunter.io/prices/rtx-4090)
- Installed base: [HF hardware census 2026-05](https://huggingface.co/blog/clem/hardwaresetupsonhf), [Steam survey Sep 2026](https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam)

Benchmarks and models:
- Spark llama.cpp: [ggml-org #16578](https://github.com/ggml-org/llama.cpp/discussions/16578)
- 5090 / R9700 on Qwen3.5-35B-A3B: [#19890](https://github.com/ggml-org/llama.cpp/discussions/19890)
- gpt-oss-120b CPU-MoE offload: [carteakey](https://carteakey.dev/blog/local-inference/optimizing-gpt-oss-120b-local-inference/), [#15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- Qwen3.5-122B on one Spark: [NV forum](https://forums.developer.nvidia.com/t/qwen3-5-122b-a10b-on-single-spark-up-to-51-tok-s-v2-1-patches-quick-start-benchmark/365639)
- Spark model guide: [gpuwerk](https://gpuwerk.com/dgx-spark/guides/best-models-2026/)
- 2x Spark Qwen3.5-397B: [alooftwaffle](https://alooftwaffle.substack.com/p/96-hours-with-dual-dgx-sparks-and)
- DeepSeek V4.1-Flash on 2x Spark: [0xSero](https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks), [coolbho3k](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark)
- M5 Max: [hardware-corner](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
- Long-context decode on Strix: [strix-halo-guide](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/BENCHMARKS.md)
- Model cards and blogs: [gpt-oss](https://openai.com/index/introducing-gpt-oss/), [Qwen3.6-35B-A3B](https://qwen.ai/blog?id=qwen3.6-35b-a3b), [Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json), [Qwen3-Next config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json), [Qwen3.8 lineup (agg)](https://codersera.com/blog/qwen-3-8-model-lineup-2026/), [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), [Nemotron 3 Super](https://arxiv.org/html/2604.12374v1), [Nemotron 3 Nano](https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models), [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4), [Kimi Linear](https://arxiv.org/abs/2510.26692)
- Community requests: [GLM-5.2 #3](https://huggingface.co/zai-org/GLM-5.2/discussions/3), [GLM-5 #3](https://huggingface.co/zai-org/GLM-5/discussions/3), [Qwen3.6-27B #7](https://huggingface.co/Qwen/Qwen3.6-27B/discussions/7), [Qwen3.8 #161](https://github.com/QwenLM/Qwen3.8/discussions/161)

Quantization:
- Bit width vs quality: [Unsloth Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- Qwen3.5 tensor sensitivity: [Unsloth Qwen3.5 KLD](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- Gemma 4 QAT: [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- KV quantization on hybrids: [llama.cpp #21385](https://github.com/ggml-org/llama.cpp/issues/21385)
- Kimi K2 Thinking INT4 QAT: [HF](https://huggingface.co/moonshotai/Kimi-K2-Thinking)

Sparse attention and MTP:
- DeepSeek V3.2 DSA: [arXiv 2512.02556](https://arxiv.org/pdf/2512.02556)
- DeepSeek V4: [arXiv 2606.19348](https://arxiv.org/abs/2606.19348), [vLLM blog](https://vllm.ai/blog/2026-04-24-deepseek-v4)
- DeepSeek V4.1-Flash: [arXiv 2609.19969](https://arxiv.org/pdf/2609.19969), [DeepSeek news](https://api-docs.deepseek.com/news/news260910/)
- NSA: [arXiv 2502.11089](https://arxiv.org/abs/2502.11089)
- InfLLM-v2: [arXiv 2509.24663](https://arxiv.org/html/2509.24663)
- MoBA: [arXiv 2502.13189](https://arxiv.org/abs/2502.13189)
- GLM-5 (DSA conversion): [arXiv 2602.15763](https://arxiv.org/html/2602.15763v1)
- MTP on Qwen3.6-35B-A3B / 3090: [thc1006](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)
- MoE speculative-decoding traffic: [arXiv 2506.20675](https://arxiv.org/html/2506.20675v1)
- Speculative decoding on Apple: [arXiv 2607.17283](https://arxiv.org/abs/2607.17283)

No primary source was found for "NSA2". If it is an internal name, the closest published recipes
are InfLLM-v2 and GLM-5's DSA conversion.
