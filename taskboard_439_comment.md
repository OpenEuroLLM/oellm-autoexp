## Architecture search status (MN5, rounds 1–5)

All comparisons are at small scale. s1 is hidden 768 × 12 blocks (~85M non-embedding) and s2 is hidden 1024 × 18–20 blocks (~0.25B); MoE rounds use the same widths with 48–64× sparse FFNs. Arms are **parameter-matched**: non-embedding parameters within 1 %, d_ffn trimmed per arm. Unless noted, lr, batch and data order are equal and gaps are in nats vs the round's reference. Full log with every table: [`docs/hybrid_moe_3b_campaign.md`](https://github.com/OpenEuroLLM/oellm-autoexp/blob/feat/hybrid-v0.19/docs/hybrid_moe_3b_campaign.md). Throughput work is in #438.

### How we decided: `loss_prediction` for stability and decisions

Every run goes through `loss_prediction` (losspred) automatically (`scripts/korbi/campaign/curves.py`, `analyze.py`, `joint_fit.py`, `cut_study.py`, refreshed every 2 h on MN5):
- **Health.** Spike detection (hysteresis detector) with the size and lasting offset of every spike, growth / stall regimes (monotonicity detector), and a warmup-instability scan. On top: grad-norm and non-finite-gradient monitors that cancel a run early.
  - Two real problems were found this way: the dense-mHC NaN, and a diverging free-routing arm (grad norm 1e2 → 1e8 with a finite loss).
- **Learning-curve fits.** L = E + A (τ + t0)^−α in LR-area time on the stable phase, per run, with paired moving-block bootstrap intervals for the gap of every arm to its reference. Comparisons are at equal seen tokens.
- **When is a short run decision-grade?** Cut studies on runs that went to the full 4BT:
  - AdamW fits from 40 % of the target predict the final gaps to 0.002.
  - Angular MuOwn needs the fit to start after step ~1500 (the early transient); with that, 0.0008 at 54 %.
- **Noise.**
  - Seed pairs (other init and other data order) agree at step 4000 to 0.0003, so the step-4000 gaps below are resolved.
  - Single-run extrapolations to 10BT differ by up to 0.037 between seeds, so the final decision uses joint fits (shared exponent prior) and paired seeds.
  - Early in MoE runs, nondeterministic kernels give ~0.02 run-to-run noise at steps 300–900. That is why we read gaps at 4000.
- **Feature gate.** A feature is kept only if it beats its matched base beyond the noise at equal lr / batch, and costs ≤ 30 % throughput at small scale.

### Rounds: what was tested where

| Round | Setting | Question |
|---|---|---|
| 1 | s1 + s2 dense, datamix4, 2BT WSD | Megatron vs torchtitan (Megatron: 1.08× at 130M, ~1.8× at A3B-30B); A3B speed; feature screen (attention + XSA / output gate; KDA / CKDA / GDN 3:1 hybrids); LR × BS sweeps for AdamW, MuOwn, angular MuOwn, nGPT; warmup studies |
| 1b | s1 + s2, 4BT stable, AdamW with a 1000-step warmup | the valid setup (round 1's AdamW warmup was too short); CKDA with its standard inits; parameter-matched Qwen3.5-like GDN; every run curve-fitted |
| 3 | s1 dense, both optimizers | the hybrid recipe (CKDA 3:1 / 5:1, triadic CKDA, XSA inside CKDA) and residuals / memories (mHC-4, routed residual, block AttnRes, Engram, nGPT) |
| 3B | s1, MoE on the attention base | sparsity 4–64×, granularity, shared expert, router, MoE lr / batch |
| 3C | s1 + s2 | residual variants under angular MuOwn (mixer lr), pure-linear stacks, nGPT at its own lr |
| 4 / 4R / 4G | s1 + s2, CKDA 5:1 + 48× MoE | residuals on the MoE hybrid (routed 2/3 of 4/8, keyed, mixing, balancing, mHC-2 / mHC-4), LatentMoE, granularity; repeats with stream diagnostics and CUDA graphs |
| **5 (running)** | s1 / s2 / s3, flag mix, 8k, document masking | the final architecture choices on the production data setting |

### Linear attention: CKDA

- **CKDA with its standard inits:** shipped alpha, bias-free beta, low-rank output gate, beta_max 2.
  - An early screen used a non-standard CKDA config (spread alpha init, linear output gate), which made it look worse than it is. This was corrected in round 1b.
- **Hybrid recipe:** CKDA 5:1 (3:1 is 0.005–0.007 better, at twice the attention layers / KV cache). The attention layers use NoPE + qk-norm + XSA + output gate.
- **Qwen3.5-like GDN** (gated attention, partial RoPE, zero-centred norm, parameter-matched):
  - It wins under AdamW (−0.022).
  - Under angular MuOwn its lead shrinks linearly with tokens and is gone by 4BT (s1 and s2: −0.030 at 1k steps → +0.001 at 7.6k).
  - The CKDA hybrid's lead under MuOwn is stable (−0.020 at 1k → −0.021 at 7.6k).
- **Triadic CKDA:** gains under AdamW (−0.021 over CKDA), but equal under MuOwn at −10 % speed. Optional.
- **Pure-linear stacks:** trail attention by ~0.045. One triadic layer in six recovers a third of that.
- **Hybrid vs all-attention:** dense, the hybrid gains −0.022 (s1) / −0.010 (s2) under MuOwn. With MoE at 4k the gain is small (−0.002 at s1). The hybrid's main argument is now inference efficiency (6× less KV cache at 5:1). Round 5 measures the loss gap at 8k with document masking.

### XSA (exclusive self-attention)

XSA ([Zhai 2026, arXiv:2603.09078](https://arxiv.org/abs/2603.09078)) orthogonalizes every attention output head against the token's own value vector, so a head cannot just copy the token itself. It costs ~3 % throughput (fused Triton kernel, together with the output gate). Ablations:

| Where | AdamW | angular MuOwn |
|---|---|---|
| attention, s1 / s2 (round 1) | −0.051 / −0.055 | −0.018 / −0.020 |
| attention, s1, paired bootstrap fit (round 1) | | −0.020 (± 0.001) |
| on top of a 3:1 hybrid's attention layers, s1 / s2 (round 1) | −0.004 to −0.012 / +0.006 to −0.012 | −0.004 to −0.008 / −0.005 to −0.008 |
| inside the CKDA layers (`linear_xsa`, round 3 ablation) | +0.023 (worse) | ±0 |

- **XSA is the largest single attention feature** and holds under both optimizers and at both sizes. For comparison, the attention output gate gains −0.022 under AdamW but fails the gate under MuOwn.
- **On a hybrid it adds little.** The linear mixer already covers much of what XSA fixes; the gains there are within ~3× the seed noise.
- **Kept in every attention layer,** including the hybrid's (and in the all-attention reference). **Dropped inside the linear layers.**

### Optimizer: angular MuOwn

- **Gain:** −0.07 to −0.11 vs the screen's AdamW (~−0.09 vs tuned AdamW), on dense, hybrid and MoE.
- **Schedule:** no warmup. lr 4e-3 at s1, transferred by (N2/N1)^−0.417 (s2 2.5e-3, s3 1.8e-3); the transfer keeps the ranking.
- **Batch:** b* grows with D (gbs 128 best at 3BT, consistent with b* ∝ D^0.46).
- **It roughly halves every architecture gain.** Part of what the mixers fixed under AdamW the optimizer fixes too, so features are gated under MuOwn.
- **Residual mixers need `residual_mixer_lr_mult 0.25`.** At the shared lr they trained at ~4× Adam's optimum, which hid the mHC / routed / AttnRes gains under MuOwn. With it, every residual variant beats the plain hybrid under both optimizers.
- **nGPT dropped:** 0.13–0.18 behind after cooldown, diverges at 2× lr.

### MoE sparsity and shape

At equal active parameters and tokens, vs the dense twin (angular MuOwn, raw gap at step 4000, s1):

| Sparsity | 4× | 8× | 16× | 32× | 64× |
|---|---|---|---|---|---|
| gap | −0.077 | −0.104 | −0.125 | −0.135 | −0.143 |

Gains diminish to ~0.01 per doubling at the top; rounds 4 / 5 use 48×.

- **Shared expert: yes.** top-3 + shared beats top-4 by 0.010 at no speed cost, and the router balance is better.
- **Router:** sigmoid scores + expert bias (rate 1e-3) + aux 1e-4. Best loss under both optimizers, and the best balance (AdamW, worst-layer max/mean − 1: 0.6 vs 6.9 with softmax + aux 1e-3).
- **Granularity:** finer is slightly better (−0.005 to −0.012) but slower. Production shape: 160 × 768 top-5 + 1 shared (~39B total / 2.7B active).
- **LatentMoE at equal cost:** −0.021 (round 4, s1); its throughput at scale is pending.

### Residual streams: mHC and our routed residual

Both widen the residual into n parallel streams.

- **mHC** (DeepSeek, [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)). Every sublayer reads a mix of all n streams and writes its output to all of them, with the streams mixed by a doubly stochastic matrix (Sinkhorn). It is norm-preserving through depth, but every sublayer moves ≥ 2n hidden vectors per token, and activation memory and the pipeline payload grow n×.
- **Routed residual (new, ours: `mhc_routed_topk`).** The same widened residual, made sparse like an MoE over streams. Per token and sublayer:
  1. A small router on the main stream 0 scores the other n − 1 streams.
  2. The sublayer reads stream 0 plus the top k − 1 of the others, mixed by the router's softmax weights.
  3. It adds its output back into exactly those k streams, each with a learned write gate. The other streams are untouched.
  - **Cost:** a sublayer moves (2k + 1) hidden vectors instead of ≥ 2n, and a decode step touches k + 1 of the n streams.
  - **Content keys** (`mhc_routed_key_dim`): the router also reads a small tag (64 channels) of every stream, so it can address streams by content and not only by position.
  - **Mixing** (`mhc_routed_mix`): an mHC-style k × k Sinkhorn mix among the picked streams.
  - **Balancing** (`mhc_routed_balance_rate`): an aux-loss-free bias, as in DeepSeek-V3's MoE router, that spreads the picks.
  - **Reversible** (`mhc_routed_reversible`): the backward undoes the writes instead of saving the gathered rows, which saves memory.
  - Fused kernels; works under per-layer CUDA graphs.

Results (round 4, s1, vs the CKDA 5:1 MoE base at step 4000; s2 in brackets):

| Residual | Active streams per sublayer | Gap |
|---|---|---|
| **mHC-4** | 4 of 4 (dense) | **−0.049** |
| routed 3-of-8, keyed + mixing | 3 of 8 | −0.030 |
| routed 3-of-8, keyed | 3 of 8 | −0.019 (s2: −0.030) |
| mHC-2 | 2 of 2 | −0.025 (s2: −0.025 / −0.029) |
| routed 2-of-8, keyed | 2 of 8 | −0.013 (s2: −0.007) |
| routed 2-of-8, plain | 2 of 8 | −0.004 |
| free routed 1-of-8 (no forced main stream) | 1 of 8 | +0.40 at step 900 (stopped) |

What we learned:
- **The gain follows the number of active streams, not the stream count.**
- **Without keys the router collapses onto 1–2 streams** (pick entropy 0.14). Keys and balancing de-collapse it (0.8), but the loss barely moves.
- **The main stream must stay in every read;** free routing fails.
- **mHC needed a Megatron fix:** Sinkhorn in log space. The clamped normalisation produced NaN gradients with a finite loss once the mixing logits spread. With it, **mHC-4 is the best residual by a wide margin.** The routed variants that come close need 3 active streams, keys and mixing, so they lose the "cheap axis" premise.
- **Throughput decides it.** At s1 the residual arms ran −33 % vs the base without CUDA graphs and −15 % with graphs on the attention / mixer layers (42.6k vs 50.3k tokens/s/GPU). At the production layout (8k, document masking, PP4 × VPP2) routed 2-of-8 costs 30 % of tokens/s, because it currently sends all 8 streams across every pipeline boundary. mHC-4 (4 streams per boundary) does not fit in memory there yet; coarser recompute checkpoints and PP without VPP are being tested (#438).

### Dropped or on hold

- **Engram** (n-gram memory): large gain (−0.07) but not parameter-matched (sparse tables like embeddings). Only as a candidate with its memory cost reported.
- **Block AttnRes:** −0.011 with the mixer-lr fix, still improving at 4k; on hold.
- **Attention output gate alone:** fails the gate under MuOwn (kept in the hybrid's attention layers, where it pairs with NoPE).
- **nGPT:** dropped (above).

### Next

Round 5 is running: the flag mix (452 shards, 73 TB) at 8k with document masking, s1 / s2 / s3 (s3 = hidden 1280 × 24, ~0.5B active / 12B total, 4 nodes).
- **Arms:** all-attention MoE; the Qwen3.5-like proxy (main run A); the CKDA 5:1 base (main run B); + mHC-4 / mHC-2 / routed 1 + 1-of-8 / AttnRes; granularity ×2; and seed pairs.
- **Decision:** by joint curve fits extrapolated to the overtrained regime (10T tokens), together with the production-layout throughput. If mHC is only slightly better at a high compute cost, it is dropped. If mHC is about equal to or worse than the routed residual, the routed residual is taken, in its best setting.
