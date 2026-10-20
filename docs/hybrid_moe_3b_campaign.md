# Campaign: a 3B-active hybrid MoE for ~10T tokens

**Status:** plan, 2026-10-17. Round 1 (≈2k GPU-h) is in execution; decisions are reviewed
before round 2. Companion documents: the general method in
[`optimal_architecture_search_procedure.md`](optimal_architecture_search_procedure.md) (curve
model, HP laws, envelope law, sequential design), the speed state in
`config/experiments/korbi/hybrid_speed/{SPEED_STATE,NOTES,TODO_SPEED}.md`.

## 0. Goal and constraints

Find the best setup for a **~3B-active MoE hybrid** trained on **~10T tokens**: best loss at
the target scale, decent training speed (MFU), high inference efficiency (small KV / state per
token, few active parameters). Constraints:

- **Total budget for the decisions: < 10k GPU-h** on MareNostrum 5 (4× H100 64 GB per node),
  plus a small buffer. The main decisions are made at small scale and validated at one larger
  scale before the production run.
- **Round 1: ≈ 2k GPU-h** (this document's §4), then a review.
- **Schedules:** constant LR (stable) or warmup-stable only; cooldowns only as short branches
  for measurement (WSD). nGPT needs no warmup; MuOwn / angular MuOwn likely neither (tested).
- **Feature gate:** a feature is considered further only if it **improves loss at parameter-
  matched size with equal lr / bs** (beyond the seed noise), and costs ≤ 30% speed at the
  small scale (speed suite ratio ≥ 0.7 of its reference) -- and holds up at scale (§3.2).
- **Prior knowledge:** CKDA scales well and works best in a hybrid (ComplexKDA ladder, §1.2).
  Open: which linear variant, XSA vs plain attention, hybrid ratio, MoE shape, optimizer.
- **Framework:** Megatron-LM unless torchtitan is clearly faster; if it is at 130M, large-scale
  speed tests of both decide (§3.1).

## 1. Inputs

### 1.1 Speed at small scale (done / in flight)

Speed suite (one node, 130M dense / 120M-active 1.2B-total MoE, tok/s vs reference): see
SPEED_STATE.md. As of 2026-10-17, below 0.7: kda / ckda 0.66 / 0.67, tri_kda / tri_ckda
0.57 / 0.56, pkm 0.61, mhc4 0.58, routed_k2_mix 0.63; power now 0.71 (own Triton kernel); the
MoE reference at 0.44 of a dense model of equal active width (communication, dispatch,
optimizer over 370M params / GPU). Megatron dense reference: 320k tok/s/GPU at mbs 8,
334k at mbs 16 (123.6M params, seq 4096).

### 1.2 ComplexKDA ladder (OpenEuroLLM/ComplexKDA, `lm_scaling/`)

- 180 cells: 47M-1.7B × D ∈ {6, 12, 20, 30, 50} BT, Nemotron-CC, GPT-NeoX vocab, seq 4096,
  arms attn, attn-qknorm, KDA, CKDA, KDA / CKDA 3:1 hybrids (gated NoPE attention), matched in
  N by d_ffn. Results `harvest/megatron_ladder.tsv`, geometry `ladder_matched.json`.
- HP law (Ajroldi et al.): b* = 9.9e-5 N^0.145 D^0.460 sequences, η* = 0.3714 N^-0.417 D^0.0862;
  used LRs 2e-3 (47M), 1e-3 (124-302M), 5e-4 (588M-1.7B), gbs 64-128 × 4096, AdamW β2 0.95,
  wd 0.1, WSD (2000 warmup, 20% linear cooldown).
- Mean loss offset vs attention: qknorm -0.028, KDA -0.062, CKDA -0.059, KDA / CKDA hybrid
  -0.073 / -0.071 nats; hybrid vs pure ≈ -0.019 nats, constant in scale; seed noise ≈ 0.002.
- Fit L = E + (A N^-α + B D^-β)^k: per arm E 1.16-1.25, α 0.40 (attn) vs 0.52 (KDA family),
  β 0.38 vs 0.42; held-out RMSE 0.0015 at 1.7B / 50BT.
- Throughput (GH200, torchtitan): Megatron measured at ~0.67× torchtitan at 47M -- to be
  re-measured on our stack (§3.1).

### 1.3 Data

**Round 1 uses datamix4** (`config/backend/megatron/data_datamix4_mn5.yaml`): exp_irene's
multilingual OpenEuroLLM mix on MN5 ("1TT-option-4", = LUMI datamix4), OELLM 256k tokenizer,
156 train / 155 validation prefixes under `/gpfs/projects/ehpc533/preprocessed/
openeurollm-tokenized-256k` as a per-split JSON (copied from exp_irene 2841f4c; all present).
All comparisons of a round use the same data; the switch to "flag" happens between rounds.

**The "flag" mix** (once fully transferred) is prepared:

`/gpfs/projects/ehpc533/training-data-collection/flag/`: Megatron-format shards per source
(`<source>/megatron-lm/...`), mix `datamix.txt` (452 weighted prefixes, relative paths), token
counts `counts.csv`, OELLM 256k SentencePiece tokenizer
(`/gpfs/projects/ehpc533/tokenizers/tokenizer-256k`, as JUPITER's `oellm_tokenizer_256k`).
Several sources land overnight (2026-10-16/17). The JUPITER "flag" config
(`config/backend/megatron/oellm_32b_dense/data/oellm_256k_15TT_jupiter_train.yaml`: weighted
`data_args_path`, split 998,2,0, seq 4096) is the template for `data_flag_mn5.yaml`.

**Vocabulary:** 262,144 rows. At small width the embedding dominates (124M body: d 576 →
151M embedding parameters); comparisons therefore match **non-embedding** parameters, use
untied or tied embeddings consistently, and report N_total, N_non-emb and N_active.

### 1.4 Tooling

- Speed suite and kernel tests (`scripts/korbi/hybrid_speed/`), autoexp + monitor on MN5.
- Loss prediction: `../loss_prediction` (losspred, state 2026-10-17 `bd0bc55`): spike
  (`HysteresisSpikeDetector`) and regime (`MonotonicityRegimeDetector`) detection, joint family
  fits of stable + decay branches in schedule-stage time, noise metrics, warmup-instability scan,
  log-linear relations of fitted parameters to HPs. Its findings (`docs/findings.md`) that the
  plan builds on:
  - step noise is white and ∝ gbs^-0.5: comparisons use a known noise model, not extra seeds;
  - cooldown branches are predicted from the schedule alone to ~0.2 % (trend error), and the
    drop follows lr^0.47 gbs^-0.32 size^0.18: **stable trunks + predicted cooldowns** are a sound
    basis for decisions, with a few measured branches as checks;
  - late-warmup blow-ups are frequent at high LR (confounded with β2 0.99) and recover: they get
    an **instability score** and are excluded from the stable fit rather than rejecting the run;
  - stable-phase parameters (E, A, α) are only partly identifiable per run: share them across
    families (per size / arm) before regressing on HPs.
- `optimization_procedure/` prototype (hierarchical fit, curve QA, design, ledger) and
  `scripts/scaling_laws/`.

## 2. Decision framework

### 2.1 What is compared

1. **Speed at scale** (MFU / tok/s / GPU at the target layout) for "similar" variants -- the
   same family at the same parameter budget (e.g. the linear mixer of a 3:1 hybrid), not across
   unrelated shapes.
2. **Loss at parameter-matched size**, equal lr / bs, on a 6BT reference training (flag data).
3. **Inference efficiency**: KV + recurrent state bytes per token at 32k context, active
   parameters per token, decode FLOPs (computed, not measured).
4. **Health**: stability at high LR, spike rate, grad-norm / logit / attention-entropy drift
   (the speed suite's arch metrics), loss-prediction residuals.

The combined criterion is the **effective compute multiplier** of §7.3 of the method doc (iso-loss
compute ratio × throughput ratio) in the target regime (3B active, 10T tokens: D/N_active ≈ 3000),
reported with its uncertainty; inference efficiency is reported alongside, not folded in.

### 2.2 Feature gate (promotion)

A feature f enters the next stage iff, against its matched base:

- loss(f) < loss(base) - 2σ (σ ≈ 0.002 nats at 6BT, from the ladder's seeds; re-estimated with
  2 seeds of the base), at equal lr / bs **and** at its own best LR from a 3-point LR probe
  (a feature that only wins at a shifted LR is noted, not rejected);
- small-scale speed ≥ 0.7 of the base (speed suite), or a credible kernel path to it;
- healthy at 2× the base LR (no divergence, spike rate not above base).

### 2.3 Health metrics and loss prediction instead of full runs

- Runs are **stable (constant LR)** trunks with optional short decay branches (WSD); no
  cosine. nGPT / MuOwn arms start without warmup (verified by an A/B with 1% warmup).
- From every trunk we extract (losspred): spike events and their lasting offset, regime
  changes (growth / stall), the warmup-instability magnitude, noise level vs its gbs^-0.5
  expectation, grad-norm trend, fit residuals, and the **predicted cooled loss** at D (family fit,
  cooldown drop from the fitted drop law). A run is **healthy** if no lasting spike or growth
  regime after warmup and residuals are within noise; unhealthy-but-recovering runs keep their
  data with an **instability penalty** (the loss excess integrated over the event), which enters
  the HP response as a separate stability term: an HP optimum is the best predicted loss
  *subject to* an instability budget, so a sharp optimum next to divergence is avoided.
- Early extrapolation: HP optima and scaling parameters are fitted from short trunks (≈ 10-20 %
  of D) extrapolated with the shared-parameter family model; the extrapolation error is tracked
  on the runs that are continued (ledger), and its spread widens the decision intervals.
- Decisions use the *predicted* cooled loss at the comparison budget, validated on a subset of
  branches that are actually cooled (prediction error tracked in a ledger).
- Scaling parameters and optimal HPs come from the fits (method doc layers B / C) once ≥ 2
  sizes per arm exist; LR / BS optima are fitted as power laws in (N, D).

### 2.4 Hyperparameters

- **LR × BS** short sweeps at small scale (0.5-1BT stable + predicted cooldown), per
  optimizer: AdamW (reference), MuOwn, angular MuOwn, nGPT (its own normalisation, no warmup).
  Grid: 5 LRs (×2 spaced around the ComplexKDA law's η*) × 3 batch sizes (b*/2, b*, 2b*).
- **Fixed** unless a sweep says otherwise: β2 0.95, wd 0.1 (AdamW), clip 1.0, z-loss off,
  bf16, qk-norm on.

## 3. Phases

### 3.0 Small-scale speed, mbs tuning (round 1, ≈ 120 GPU-h)

- Finish the speed TODO items with large expected gains (TODO_SPEED.md).
- **mbs tuning** for the reference and each promising arm: mbs ∈ {4, 8, 16, 32} with CUDA
  graphs on / off, selective recompute where memory runs out; dense 130M and MoE 120M/1.2B.
  Output: the best mbs per arm and its tok/s (feeds the cost model).

### 3.1 Megatron vs torchtitan (round 1, ≈ 60 GPU-h)

- 130M dense on both, same model / data / node (titan: upstream main, TitanTraining-main image,
  recipes `oellm_autoexp/titan_recipes/hybrid_speed.py`; sweep `titan_main_mn5.yaml`).
- If titan is not faster (≤ 1.05×): Megatron. If it is: titan's 3B-active MoE (Qwen3 MoE
  recipe) at the same multi-node layout in §3.2 before deciding; the decision weighs our feature
  kernels (FLA / CuTe / Triton, all framework-neutral modules) and the porting cost.

### 3.2 Large-scale speed: 3B active, 30B-100B total (round 1, ≈ 450 GPU-h)

Two target shapes (Qwen3.5 / Qwen3-Next-like hybrid MoE; exact widths fixed in the configs):

| | A3B-30B | A3B-80B |
|---|---|---|
| hidden / layers | 2048 / 48 | 2048 / 48 |
| mixer pattern | 3:1 linear : attention | 3:1 linear : attention |
| attention | 16 q heads × 128, 2 kv groups, qk-norm, gated | same |
| linear mixer | CKDA (32 × 128) / KDA / GDN | same |
| experts | 128 × 768, top-8 + 1 shared | 512 × 512, top-10 + 1 shared |
| params total / active | ≈ 30B / ≈ 3.2B | ≈ 80B / ≈ 3.0B |

- Layouts: EP 8-32 across nodes, TP 1-2, PP 1-4, distributed optimizer; 8-16 nodes; 30-50
  steps per arm (the throughput settles after ~15).
- Feature set per shape: linear mixer {CKDA, KDA, GDN}, attention {plain, XSA}, dispatcher
  {alltoall, flex/HybridEP}, EP a2a overlap on / off (now ported for HybridModel), CUDA graphs,
  recompute policy, mbs. "Similar" pairs are compared at the same layout.
- Output: tok/s/GPU and MFU per arm and layout; the best layout per shape; the speed side of
  the decision.

### 3.3 Data on MN5 (round 1, ≈ 10 GPU-h): datamix4 now, flag later

- `scripts/korbi/flag_datamix_mn5.py`: `datamix.txt` → absolute-path weights file restricted to
  shards that exist (with renormalised weights and a report of what is missing), plus a fixed
  validation set (held-out shards); `config/backend/megatron/data_flag_mn5.yaml`.
- Smoke: 130M, 200 steps, data cache built once, loss curve sane.
- Re-run the generator once the data has fully landed; the reference trainings use the
  complete mix only.

### 3.4 Parameter-matched feature screen (round 1, ≈ 650 GPU-h)

At **two sizes** with the ComplexKDA geometry (non-embedding ≈ 96M and ≈ 257M; 256k vocab,
tied), flag data, equal lr / bs from the HP law (checked by §3.5 at the small size), stable
trunks to 2BT with a decay branch at 2BT, then predicted to 6BT:

| group | arms |
|---|---|
| attention | attn-qknorm (base), +XSA, +output gate |
| hybrids 3:1 | CKDA, KDA, GDN, mLSTM (each with plain attention and with XSA) |
| MoE | 3:1 CKDA hybrid MoE vs its dense twin (active-matched), 120M-active / 1.2B-total |
| optimizer | AdamW vs MuOwn (from §3.5) on the base and the best hybrid |

Survivors of the gate (§2.2) at both sizes go to §3.6.

### 3.5 LR / BS sweeps (round 1, ≈ 450 GPU-h)

- Size 1 (≈ 96M non-emb), 1BT stable trunks: AdamW and MuOwn 5 LR × 3 BS (30 runs), nGPT and
  angular MuOwn 5 LR × 1 BS (10 runs), each without and (AdamW) with 1% warmup.
- Size 2 (≈ 257M): AdamW and MuOwn 3 LR × 2 BS (12 runs).
- Output: η*, b* per optimizer and size with uncertainty; the exponent check against the
  ComplexKDA law; the LR used by the screen.

### 3.6 6BT reference trainings (round 1, ≈ 250 GPU-h; more in round 2)

- Size 1 and 2, WSD (stable to 4.8BT + 20% cooldown), base + the top 2-3 arms, 2 seeds of the
  base: the measured anchor for the loss predictions of §3.4.

### Round 1 budget

| phase | GPU-h |
|---|---:|
| 3.0 speed + mbs | 120 |
| 3.1 Megatron vs titan | 60 |
| 3.2 3B-active speed at scale | 450 |
| 3.3 flag data | 10 |
| 3.4 feature screen | 650 |
| 3.5 LR / BS sweeps | 450 |
| 3.6 6BT references | 250 |
| **total** | **≈ 1990** |

## 4. After round 1 (to 10k GPU-h)

1. **Third / fourth size** (≈ 0.6B and 1.0B non-emb) for the survivors, the HP laws and the
   envelope fit (≈ 2.5k GPU-h).
2. **MoE ladder** at 3 active sizes (0.1-0.6B active, 8-16× total) with the chosen mixer:
   expert count / granularity / shared expert / zero-compute experts (≈ 2.5k GPU-h).
3. **Held-out validation run** at ≈ 1B active, 50-100BT, prediction registered beforehand
   (≈ 1.5k GPU-h).
4. **Final pick** of the 3B-active config: layout, mbs, recompute, dispatcher; a short
   production smoke at the full layout (≈ 0.5k GPU-h).

## 5. Execution notes

- All runs go through autoexp + the monitor on MN5 (`scripts/korbi/mn5_autoexp_bg.sh`), synced
  with `./sync_to_marenostrum.sh`; configs under `config/experiments/korbi/campaign_3b/`.
- Every run logs N_total / N_non-emb / N_active and throughput; parameter counts are checked
  before submission (matched within 2%).
- Predictions (cooled loss at 6BT, η*) are written to a ledger before the measuring runs
  finish.
- Speed numbers come from the median step time after step 30 (speed suite convention).
- Speed is compared on **tok/s/GPU** only: Megatron and torchtitan count FLOPs differently (titan
  counts non-causal attention: 1.195 vs 0.968 GFLOP/token at 130M), so their TFLOP/s and MFU are
  not comparable.
- Dataloader stalls on GPFS: the node-local prefetch / windowed shuffle (branch
  `data_prefetch_shuffle`, merged) must be wired into pretrain_hybrid and the per-split blends
  and A/B'd before the 6BT references and the at-scale real-data runs (TODO_SPEED.md §0b).

## 6. Open points for review

- Size of the "6BT reference": with the 256k vocab, is ≈ 257M non-embedding the right top size
  for round 1, or should one rung be ≈ 0.6B?
- Hybrid ratio fixed at 3:1 in round 1 (ComplexKDA: hybrid vs pure constant ≈ -0.019); ratio
  sweeps (7:1, 5:1) deferred to round 2.
- MoE shape at scale (128 × 768 top-8 vs 512 × 512 top-10) is a speed question in round 1 and a
  loss question in round 2 (MoE ladder).

## 7. Round-1 status (2026-10-17, interim)

### 7.1 Megatron vs torchtitan (§3.1): decided, Megatron

130M dense, FineWeb-10BT, one node, best of each:

| framework | setup | tok/s/GPU | MFU |
|---|---|---:|---:|
| Megatron | mbs 32 (CUDA graphs on / off: 343.8k / 342.9k) | 343.8k | ≈ 41 % |
| Megatron | mbs 16 | 331.9k | ≈ 40 % |
| titan main | GraphTrainer (whole-step graph, full Inductor) + FA3 varlen, mbs 32 | 319.5k | 38.6 % |
| titan main | same, mbs 16 | 305.2k | 36.9 % |
| titan main | same, flex instead of FA3, mbs 32 | 292.4k | 35.3 % |
| titan main | FSDP2 + regional compile, FA3, mbs 8 | 252.2k | 30.5 % |

Megatron is 1.08× faster (above the 1.05× bar). At the A3B-30B scale (8 nodes, same shape:
attention-only, 128 × 768 top-8, 262k vocab, gbs 1024; titan: FSDP2 over 32 ranks + EP, full
remat, compiled regions, FA3; `titan_a3b30_mn5.yaml`, suite ta2) the gap widens. Median
tok/s/GPU over steps 6-14: titan HybridEP EP4 ≈ 3,630, all-to-all EP4 ≈ 3,410, all-to-all
EP8 ≈ 1,330. Megatron's tuned attention-only arm with normal routing: ≈ 6,500-6,700. Titan is
≈ 0.55× Megatron, so **the decision is Megatron**. Titan's DeepEP path needs DeepEP v2
(`ElasticBuffer`), which the image does not have; rebuilding it would not close a 1.8× gap.
Titan used ~30 of 63 GiB at mbs 4: selective remat / mbs 8 arms are prepared but not run. FA4
(flex_flash) hangs on H100, as upstream reports. Titan's FA3 path needs the `flash_attn_3`
nightly wheel, which is not in the image yet; for now it sits in `pylibs_titan_fa4`.

### 7.2 A3B-30B / A3B-80B speed (§3.2)

A3B-30B: 48 blocks (3:1 mixer : attention, each followed by a 128 x 768 top-8 MoE), hidden
2048, 262k vocab, 8 nodes, TP1 / PP4 / EP4 (HybridEP), full recompute, mbs 4, gbs 1024.
Median tok/s/GPU over steps >= 5 (`speed_a3b30_mn5.yaml`, suites s6 / s7):

| arm | tok/s/GPU | note |
|---|---:|---|
| CKDA, tuned layout, gbs 4096 | 7,290 | exp_irene's batch: fewer pipeline bubbles |
| CKDA, tuned layout + forced balancing | 7,136 | benchmark-only routing: +3 % |
| **CKDA, tuned layout** | **6,947** | VPP2 7/7/7/6/7/7/6/1 + router fusion, P2P overlap, 8 connections |
| attention-only, tuned layout | ≈ 6,600 | + forced balancing: 8,801 (= exp_irene's 185.8 TFLOP/s) |
| CKDA, VPP2 layout alone | 6,443 | the P2P overlap (VPP only) is where the gain is |
| CKDA gated / EP2 / XSA / GDN / KDA (old layout) | 6,162 / 6,020 / 5,970 / 5,955 / 5,796 | |
| attention-only (old layout) | 6,117 | |
| CKDA reference (old layout) | 6,052 | |
| CKDA mbs 2 / 8 (old layout) | 5,524 / 5,735 | mbs 4 is best |
| CKDA alltoall (old layout) | 4,600 | HybridEP +32 % |
| CKDA EP8 / selective recompute / PP2 | fail | HybridEP internode assert / OOM / OOM |

- The mixers cost 0-5 % at this scale (the MoE dominates).
- The tuned layout gives +15 % (6,052 -> 6,947). exp_irene's 185.8 TFLOP/s is reproduced
  (attention-only, forced balancing: 187.9), so the remaining gap to it is routing balance
  and model, not the stack.
- A3B-80B (512 x 512 top-10 + shared, 16 nodes): EP4 / PP8 gives 3,921 at 91 % memory.
  PP8 x VPP2 tuned OOMs at mbs 4 (s8) and gives 3,941 at mbs 2 (s9, 73 % memory): no gain.
  EP8 fails internode (HybridEP) or OOMs (alltoall). Each GPU holds 128 of the 512 experts per
  layer. On MN5 (64 GB H100, HybridEP intra-node only), **the 80B-total shape trains at 0.57x
  the 30B shape's tok/s/GPU** (3.9k vs 6.9k). Its loss advantage per active parameter would
  have to pay for that: a question for the MoE ladder (§4), not more layout tuning. Levers left
  if it is needed: internode HybridEP (DOCA assert, `internode_doca.cu:124`) or alltoall EP16
  across nodes with optimizer CPU offload for memory.

**Round-3/3B recipe at scale (2026-10-18).** CKDA 5:1 (reference CKDA, no linear XSA), attention
NoPE + qk-norm + XSA + output gate, MoE sigmoid router + expert bias + aux 1e-4 with a shared
expert (top-k - 1 routed + 1 shared of the expert width); tuned layouts, 25 steps:

| model | layout | recipe (CKDA 5:1) | all-attention twin | hybrid cost |
|---|---|---:|---:|---:|
| A3B-30B (128 x 768) | PP4 x VPP2, EP4, mbs 4, full recompute | 6,908 (137 TFLOP/s, 73 % mem) | 7,539 (161, 76 %) | -8 % |
| A3B-80B (512 x 512) | PP8 x VPP2, EP4, mbs 2, full recompute | 3,880 (70 TFLOP/s, 74 % mem) | 4,653 (91, 75 %) | -17 % |

- The recipe costs nothing against the old 3:1 CKDA layout (6,947 / 3,921).
- **mHC / routed residual / AttnRes do not run at this scale.** They raise on full activation
  recompute (hybrid_block.py: "use selective recompute"). With selective recompute (MoE
  activations + norms), A3B-30B OOMs at micro batch 2 **and 1**, even without mHC. So a
  residual-stream scheme at 3B-active scale needs full-recompute support in Megatron (or more
  pipeline stages). That is the engineering item, not a fused kernel.
- EP8 (internode HybridEP) still asserts on MN5 (`internode_doca.cu:124`). A3B-80B runs EP4.

**mHC / routed residual at A3B-30B with full recompute (Megatron, 2026-10-18, local).**
hybrid_block.py now runs mHC (dense and routed) under full activation recompute: the widened
[s, b, n, h] residual goes through the checkpointed chunks, and a routed chunk takes ownership
of its input with one copy (routed writes in place). Tests: loss and every gradient identical
to no recompute (dense mHC-4, routed 2/4, routed 2/8 + content keys); Block AttnRes still
rejects full recompute (its Python block state). Each checkpoint saves n streams, so per-layer
checkpoints OOM at mbs 4 and 2. Checkpointing every r layers (recompute_num_layers = r; r = 2 is
one mixer + MoE block) fits:

| arm (A3B-30B, PP4 x VPP2, EP4) | tok/s/GPU | memory |
|---|---:|---:|
| recipe, mbs 4, r = 1 (best) | 6,908 | 73 % |
| recipe, mbs 2, r = 1 / r = 2 | 6,351 / 6,363 | 66 / 65 % |
| + mHC-4, mbs 2, r = 2 / r = 4 | 4,944 / 4,951 | 72 / 68 % |
| + routed (8 streams, 2 active), mbs 2, r = 4 | 4,755 | 75 % |
| + mHC-4, mbs 4, r = 2 / r = 4 | OOM / CUDA failure at startup | |

- mHC-4 costs **-22 %** at equal mbs and r (4,944 vs 6,363), and -28 % against the recipe's
  best layout, about the same as at s1 (-23 %). Its relative cost does not shrink at scale.
- The simple routed version is slower than mHC-4 and uses more memory: it saves and copies all 8
  streams. The routed design allows saving one stream per sublayer instead
  (mhc_routed_reversible + sublayer-level recompute: keep the gathered input x, recompute f(x)
  and undo the write in the backward), so its residual overhead becomes O(selected streams),
  independent of n. Dense mHC cannot do that: its mixing gradient needs all n streams. Build it
  if routed-8 wins on loss in round 3C.

**Routed-8 with sublayer recompute + fused router at A3B-30B (2026-10-18).**

| arm | tok/s/GPU | memory |
|---|---:|---:|
| recipe, mbs 2, PP4 x VPP2 (reference) | 6,363 | 65 % |
| + mHC-4, mbs 2, VPP2, recompute every 2 layers | 4,944 (-22 %) | 72 % |
| + routed 2 of 8, reversible, mbs 2, **PP4 without VPP** | **5,234 (-18 %)** | 80 % |
| + routed 2 of 8 (reversible or not), mbs 2, with VPP2 | hang: a pipeline P2P times out at the first step | |
| + routed 2 of 8 (reversible or not), mbs 4 | OOM | |

- Routed-8 is already cheaper than mHC-4 at scale, even without the interleaved schedule
  (VPP2 is worth +6-15 % to the plain model).
- The interleaved-schedule hang needs MoE / A3B scale: the same configuration at s1 (PP2 x
  VPP2, P2P overlap, dense FFN) runs for routed (both modes) and for mHC. It happens with and
  without the reversible mode. Next: debug with the NCCL flight recorder on A3B-30B.

### 7.3 Feature screen s1 (§3.4): done

s1: hidden 768 x 12 blocks, datamix4, lr 1e-3, gbs 64 x 4096, WSD (1 % warmup, 20 % linear
cooldown to lr / 10), 2BT. d_ffn is matched per arm (non-embedding within 1 %). There is one
seed per arm; seed noise from ComplexKDA is ≈ 0.002. losspred health: no spikes, no growth
regimes, instability 0 in all runs. (The first array lost 5 arms to a shared data-cache race;
the cache is now pinned per run and those arms were rerun as screen_s1b.)

| arm | non-emb | val loss | vs attn | tok/s/GPU (1 node) |
|---|---:|---:|---:|---:|
| KDA hybrid + XSA | 85.1M | **2.9977** | -0.064 | 130.6k |
| KDA hybrid | 85.1M | 3.0021 | -0.060 | 129.8k |
| attention + XSA | 85.0M | 3.0113 | -0.051 | 162.0k |
| CKDA hybrid + XSA | 84.3M | 3.0190 | -0.043 | 132.6k |
| CKDA hybrid + gated attention | 84.3M | 3.0237 | -0.039 | 132.4k |
| GDN hybrid | 85.2M | 3.0305 | -0.032 | 145.1k |
| CKDA hybrid | 84.3M | 3.0306 | -0.032 | 135.6k |
| attention + output gate | 85.0M | 3.0407 | -0.022 | 161.6k |
| attention (base) | 85.0M | 3.0622 | - | 166.8k |

- Every feature passes the gate at s1. XSA is the largest single attention feature (-0.051).
  On top of a hybrid it adds less (-0.004 on KDA, -0.012 on CKDA): the linear mixer already
  covers much of what XSA fixes.
- KDA (the shipped FLA layer: low-rank output gate, beta 1) is ahead of our CKDA configuration
  (beta_max 2, linear output gate, no qkv silu, spread gate init) by 0.029. This is the
  opposite of the ComplexKDA paper. It may be the CKDA settings rather than the mixer: s2
  checks whether it holds; if so, add a CKDA arm with KDA's gate / beta settings.
- GDN is as good as CKDA here and the fastest hybrid (0.87 of attention vs 0.78-0.81).
- **s2** (hidden 1024 x 20 blocks, 16 heads, 257M non-emb, same settings and arms; done):

  | arm | non-emb | s2 val | vs attn (s2) | vs attn (s1) | tok/s/GPU (s2) |
  |---|---:|---:|---:|---:|---:|
  | KDA hybrid + XSA | 257.4M | **2.8739** | -0.064 | -0.064 | 75.7k |
  | attention + XSA | 256.9M | 2.8831 | -0.055 | -0.051 | 95.9k |
  | KDA hybrid | 257.4M | 2.8859 | -0.052 | -0.060 | 76.2k |
  | GDN hybrid | 257.6M | 2.8877 | -0.050 | -0.032 | 84.1k |
  | CKDA hybrid + gated attention | 256.7M | 2.9016 | -0.036 | -0.039 | 76.3k |
  | CKDA hybrid | 255.4M | 2.9042 | -0.034 | -0.032 | 77.1k |
  | CKDA hybrid + XSA | 255.4M | 2.9106 | -0.027 | -0.043 | 75.7k |
  | attention + output gate | 258.3M | 2.9156 | -0.022 | -0.022 | 96.8k |
  | attention (base) | 256.9M | 2.9377 | - | - | 101.9k |

- **Consistent at both sizes**: every feature beats attention, and no gain shrinks from s1 to
  s2. The order is KDA ≥ GDN > CKDA (our config) for the mixer; XSA on attention -0.05 at both
  sizes; the output gate -0.02.
- **XSA on top of a hybrid is inconsistent**: KDA -0.004 / -0.012 (s1 / s2), CKDA -0.012 /
  +0.006. At 1-2 attention layers in 4, XSA's gain is within ~3x the seed noise; it needs seeds
  before it is kept.
- GDN gains from s1 to s2 (-0.032 -> -0.050) and is the fastest hybrid (0.83 of attention at
  s2, 1 node). At A3B scale all mixers cost only 0-5 % (§7.2).

### 7.4 LR / BS sweep s1 (§3.5): stable trunks done, cooldown calibration in flight

Attention base of s1, 1BT, constant LR after the optimizer's warmup (AdamW 1 %, MuOwn /
angular MuOwn / nGPT none). Val loss at 1BT, **not cooled**:

| optimizer | lr (gbs 32 / 64 / 128 at the best lr) | best |
|---|---|---:|
| angular MuOwn | 1e-3: 3.268, **2e-3: 3.201**, 4e-3: 3.231 (gbs 64) | 3.201 |
| nGPT (AdamW) | **1e-3: 3.293**, 2e-3: 3.297, 4e-3: 3.299, 8e-3: 3.379 (gbs 64) | 3.293 |
| AdamW | 5e-4: 3.425 / **3.348** / 3.409; 1e-3 at gbs 64: 3.378 | 3.348 |
| MuOwn | 1.25e-4: 3.444, **2.5e-4: 3.372**, 5e-4: 3.470 (gbs 64) | 3.372 |

- The optima are inside the grids, except nGPT, whose curve is flat at the low end (5e-4 is
  in the calibration). gbs 64 is best for AdamW and MuOwn at 1BT; that is a stable-phase
  loss, so larger batches are flattered less than after a cooldown.
- **MuOwn needs no warmup**: with vs without 1 % warmup, 3.691 vs 3.697 (1e-3) and 3.979 vs
  3.978 (2e-3).
- No run shows spikes, growth regimes or instability. The health metrics separate nothing at
  this size and length: all runs are healthy, and the LR optimum is a plain loss minimum.
- **Cooldown calibration (`hp_s1w`, WSD 1BT, 20 % linear cooldown to lr / 10, gbs 64):**

  | optimizer | lr | stable val | cooled val | drop |
  |---|---:|---:|---:|---:|
  | angular MuOwn | 2e-3 / **4e-3** | 3.201 / 3.231 | 3.103 / **3.077** | 3.1 / 4.8 % |
  | nGPT | 5e-4 / **1e-3** | - / 3.293 | 3.240 / **3.182** | - / 3.4 % |
  | MuOwn | 2.5e-4 / **5e-4** | 3.372 / 3.470 | 3.223 / **3.204** | 4.4 / 7.7 % |
  | AdamW | 5e-4 / 1e-3 | 3.348 / 3.378 | 3.267 / 3.268 | 2.4 / 3.3 % |

  - The cooldown favours higher LRs: the stable-trunk optimum is biased low (MuOwn and angular
    MuOwn flip to the higher LR). Ranking stable LRs without a cooldown model picks the wrong LR.
  - The drop is optimizer-specific. From the two LRs, drop ~ lr^k with k ≈ 0.43 (AdamW, as
    losspred's AdamW law 0.47), ≈ 0.64 (angular MuOwn), ≈ 0.80 (MuOwn). One cooldown branch
    per optimizer and LR region is needed to use stable-only sweeps.
  - Cooled ranking at 1BT: angular MuOwn 3.077 < nGPT 3.182 < MuOwn 3.204 < AdamW 3.267. Angular
    MuOwn's lead over AdamW (0.19 nats) is larger than any architecture feature here; it has to
    hold on a hybrid and at s2 before it is trusted.
  - `hp_s1w2` (one LR above each): angular MuOwn 8e-3: 3.104, so the optimum is ≈ 4e-3
    (bracketed). MuOwn 1e-3: 3.295, so the optimum is ≈ 4-5e-4 (bracketed). AdamW 2e-3: 3.248
    and nGPT 2e-3: 3.143 are still improving; `hp_s1w3`: AdamW 4e-3: 3.298 (optimum ≈ 2e-3,
    3.248), nGPT 4e-3: **3.132**, 8e-3: 3.156 (optimum ≈ 4e-3).
  - **Cooled optima at s1 (attention, 1BT, gbs 64)**: angular MuOwn 4e-3 **3.077**; nGPT 4e-3
    3.132; MuOwn 5e-4 3.204; AdamW 2e-3 3.248. All four are bracketed.
    nGPT's stable trunks were flat in LR, but cooled it gains 1.1-1.6 % per LR doubling: its
    optimum is far above the stable-trunk pick.
- Procedure consequence (§2.4): sweep LR on stable trunks, and run 2-3 WSD branches per
  optimizer to fit its drop law. Pick the LR on the predicted cooled loss, not the stable one.
- The ranking is not final until the cooldown is accounted for. losspred's drop law is
  AdamW-fitted, so `hp_s1w` runs WSD (20 % cooldown) at the best LR and one neighbour per
  optimizer (8 runs, ≈ 22 GPU-h). These calibrate each optimizer's drop for the family fit
  and show whether the ranking survives the cooldown.

### 7.5 Decisions so far, open questions (for review)

Spent: ≈ 750 GPU-h since 2026-10-16 (speed work included) of the 2k round-1 budget.

Decided:
- **Megatron-LM**, not torchtitan: 1.08x at 130M, ≈ 1.8x at A3B-30B (tok/s/GPU).
- **A3B-30B shape** (128 x 768 top-8) over A3B-80B on MN5: 6.9k vs 3.9k tok/s/GPU.
  Layout: TP1 / PP4 x VPP2 (head + loss on their own segment) / EP4 HybridEP, full recompute,
  mbs 4, router fusion, P2P overlap, 8 CUDA connections.
- **Feature gate**: under AdamW, all five features pass at both sizes. Under angular MuOwn (s1)
  KDA, CKDA and XSA pass; GDN and the attention output gate fail. The s2 re-run decides.
- **No warmup for MuOwn** (A/B identical).
- **LR selection needs the cooldown**: stable-trunk optima are biased low (optimizer-specific
  drop law); the procedure is stable trunks + 2-3 WSD branches per optimizer.

**Optimizer transfer (done, ≈ 22 GPU-h)**: the screen's 2BT WSD setting with the s1 cooled optima
(angular MuOwn 4e-3, nGPT 4e-3, both without warmup); s2 at 2.5e-3 (s1's 4e-3 x the ComplexKDA
law (N2/N1)^-0.417). Val loss, vs the AdamW (1e-3) screen arm:

| model | AdamW 1e-3 | angular MuOwn | nGPT |
|---|---:|---:|---:|
| s1 attention | 3.0622 | **2.9535** (-0.109) | 3.0195 (-0.043) |
| s1 KDA hybrid + XSA | 2.9977 | **2.9217** (-0.076) | 2.9827 (-0.015) |
| s2 attention | 2.9377 | **2.8274** (-0.110) | - |

- **Angular MuOwn carries over** to the hybrid and to s2 (with the law-scaled LR). The screen's
  AdamW LR (1e-3) is ≈ 0.02 from AdamW's cooled optimum, so the gap to tuned AdamW is ≈ 0.09.
- **The hybrid's gain halves under angular MuOwn**: -0.064 (AdamW) -> -0.032. Part of what the
  linear mixer fixed under AdamW the optimizer fixes too. s2 attention with angular MuOwn
  (2.8274) beats every AdamW s2 arm, hybrids included.
- So the architecture ranking from the AdamW screen does not carry over to angular MuOwn. **The
  gate has to be re-run with the chosen optimizer** before any feature is kept.

Open (proposed next, ≈ 600-900 GPU-h):
1. **Optimizer: angular MuOwn** (proposed; to confirm). **s1 screen re-run with it (done,
   `screen_s1_am`, ≈ 48 GPU-h)**: lr 4e-3, no warmup, otherwise as §7.3. The attention arm
   reproduces the transfer run to 0.0004 (2.9531 vs 2.9535):

   | arm | AdamW: vs attn | angular MuOwn val | angular MuOwn: vs attn | gate |
   |---|---:|---:|---:|---|
   | KDA hybrid + XSA | -0.064 | **2.9194** | -0.034 | pass |
   | KDA hybrid | -0.060 | 2.9269 | -0.026 | pass |
   | CKDA hybrid + XSA | -0.043 | 2.9332 | -0.020 | pass |
   | attention + XSA | -0.051 | 2.9350 | -0.018 | pass |
   | CKDA hybrid + gated attention | -0.039 | 2.9367 | -0.016 | pass |
   | CKDA hybrid | -0.032 | 2.9370 | -0.016 | pass |
   | attention + output gate | -0.022 | 2.9520 | -0.001 | **fail** |
   | attention (base) | - | 2.9531 | - | - |
   | GDN hybrid | -0.032 | 2.9661 | **+0.013** | **fail** |

   - All architecture gains shrink to about half or less under the better optimizer. The order of
     the survivors holds (KDA > CKDA; XSA helps on attention and on KDA).
   - **GDN and the attention output gate fail the gate under angular MuOwn**: their AdamW gains
     were things the optimizer also fixes.
   - XSA on hybrids now adds -0.008 (KDA) / -0.004 (CKDA): still small, still needs seeds.
   - Next: the s2 re-run with angular MuOwn (lr 2.5e-3, ≈ 125 GPU-h) for the second size. Then
     the 6BT references with the survivors: KDA hybrid + XSA, KDA hybrid, attention + XSA,
     attention base.
2. Why our CKDA trails KDA: a CKDA arm with KDA's gate / beta settings (1 run per size).
3. Seeds for the small deltas (XSA on hybrids, KDA vs GDN): 2 extra seeds at s1, ≈ 40 GPU-h.
4. **6BT references (§3.6)**: s1 and s2, the top arms (KDA + XSA hybrid, GDN hybrid,
   attention + XSA) + attention base (2 seeds), with the chosen optimizer. They anchor the
   losspred extrapolations of the 2BT screen.
5. The batch-size axis with the chosen optimizer (stable trunks at gbs 32 / 64 / 128 / 256 +
   WSD branches), at s1 and s2, for the b* law exponent.
6. Flag data once it has fully landed (the generator's `--require-complete`); the datamix4
   results stay the reference for round 1.

### 7.6 Running (2026-10-17, angular MuOwn as the working optimizer choice)

Angular MuOwn is used from here on as the working choice. The AdamW results above stay the
comparison; it can be revisited at review.

- `screen_s2_am`: the s2 screen with angular MuOwn (lr 2.5e-3, no warmup), 9 arms, ≈ 125 GPU-h.
- `hp_s1_bs2`: batch-size axis at s1, stable trunks gbs 32 / 128 / 256 x lr 2e-3 / 4e-3 / 8e-3,
  and one WSD branch per gbs at 4e-3 (20 % cooldown; the decay length is written per arm, see
  memory "sweep-entry interpolation"), ≈ 31 GPU-h.
- `ref6_s1` / `ref6_s1_seed2`: **6BT references at s1** with angular MuOwn (WSD, 20 % cooldown):
  KDA hybrid + XSA, KDA hybrid, attention + XSA, attention (2 seeds: 1234, 1235), ≈ 78 GPU-h.
  These anchor the losspred extrapolation of the 2BT screen (predict 6BT from 2BT; check
  against the measured 6BT).
- s2 6BT references follow once `screen_s2_am` has picked the survivors at s2.
- Flag data (`scripts/korbi/flag_datamix_mn5.py`, 2026-10-17 morning): 384 / 452 shards, 90.9 %
  of the mix weight. dclm-1.0 39/100, finephrase 4/5, finemath 0/1, finepdfs 36/39. Still
  transferring, so round 1 stays on datamix4.

Committed after these: ≈ 990 GPU-h of the 2k.

### 7.7 Registered predictions: s1 6BT from the 2BT screen (angular MuOwn)

`scripts/korbi/campaign/predict.py` fits each 2BT run of `screen_s1_am` with losspred's WSD
family model: time = LR area, anchored at the initial loss, drop_exponent held at 0.1 (one
decay start per run). It predicts the training loss at the end of the same schedule stretched
to 6BT. The fit reproduces each run's own 2BT end loss to ≤ 0.07 %. Written before the 6BT
references finished; checked against `ref6_s1` when they finish.

| arm | 2BT end (train) | predicted 6BT end (train) |
|---|---:|---:|
| KDA hybrid + XSA | 2.9531 | 2.8001 |
| KDA hybrid | 2.9609 | 2.8058 |
| attention + XSA | 2.9686 | 2.8064 |
| attention | 2.9869 | 2.8119 |
| CKDA hybrid + XSA | 2.9673 | 2.8137 |
| CKDA hybrid + gated attention | 2.9708 | 2.8177 |
| CKDA hybrid | 2.9711 | 2.8194 |
| attention + output gate | 2.9856 | 2.8209 |
| GDN hybrid | 2.9996 | 2.8349 |

The predictions shrink the hybrids' gains with more data: the CKDA hybrid is predicted to fall
behind attention at 6BT. The KDA arms and XSA keep a gain of 0.005-0.012. If the references
confirm this, the 2BT screen alone overstates what the hybrid mixers buy at fixed size.

### 7.8 Batch-size axis, angular MuOwn, s1 (done, `hp_s1_bs2` + earlier gbs-64 runs)

1BT; stable trunks over lr, WSD branch (20 % cooldown) at 4e-3 per gbs; val loss. No unhealthy
run.

| gbs (x 4096 tokens) | stable 2e-3 / 4e-3 / 8e-3 | cooled 4e-3 | tok/s/GPU (1 node) |
|---:|---|---:|---:|
| 32 | 3.305 / 3.324 / 3.444 | 3.147 | 150.8k |
| 64 | 3.201 / 3.231 / - | **3.077** | 160.1k |
| 128 | 3.215 / 3.231 / 3.344 | 3.095 | 167.5k |
| 256 | 3.280 / 3.275 / 3.392 | 3.155 | 171.3k |

- b* at 1BT and 85M non-emb: **64-128 sequences (0.26-0.52M tokens)**; flat between them
  (+0.018 at 128). Halving or doubling beyond costs ≈ 0.07.
- The LR optimum barely moves with batch size under angular MuOwn: 4e-3 cooled is near-best at
  every gbs, and the stable optimum moves up to 4e-3 only at gbs 256. A product law in (lr, bs)
  is a fair description; there is no sqrt-scaling of lr with bs here.
- For the b*(D) exponent the same axis is needed at a second token budget (3BT, gbs 64 / 128 /
  256 / 512, WSD only, ≈ 40 GPU-h). ComplexKDA's b* ∝ D^0.46 would put the 6BT optimum at
  ≈ 2.3x the 1BT one, i.e. 150-300 sequences.

### 7.9 s2 screen with angular MuOwn (done, `screen_s2_am`) and b*(D) (3BT, partial)

s2, lr 2.5e-3 (s1's 4e-3 x the ComplexKDA LR law), no warmup, 2BT WSD; val loss:

| arm | s2 val | vs attn (s2) | vs attn (s1) | gate (both sizes) | tok/s/GPU s2 |
|---|---:|---:|---:|---|---:|
| KDA hybrid + XSA | **2.7950** | -0.032 | -0.034 | pass | 73.2k |
| KDA hybrid | 2.7997 | -0.027 | -0.026 | pass | 74.5k |
| attention + XSA | 2.8067 | -0.020 | -0.018 | pass | 92.8k |
| CKDA hybrid + XSA | 2.8100 | -0.017 | -0.020 | pass | 73.2k |
| CKDA hybrid + gated attention | 2.8134 | -0.014 | -0.016 | pass | 73.5k |
| CKDA hybrid | 2.8182 | -0.009 | -0.016 | pass | 74.7k |
| attention (base) | 2.8269 | - | - | - | 98.3k |
| GDN hybrid | 2.8324 | +0.006 | +0.013 | **fail** | 81.4k |
| attention + output gate | 2.8368 | +0.010 | -0.001 | **fail** | 93.1k |

- **Feature decisions (angular MuOwn, both sizes)**: keep XSA (on attention: -0.018 / -0.020;
  on the KDA hybrid: -0.008 / -0.005). Drop GDN and the attention output gate. **Correction
  (2026-10-17, see §7.10)**: the `h_ckda*` arms above are NOT the reference CKDA (spread alpha
  init, linear output gate). KDA vs CKDA is not a choice: the mixer is CKDA with both inits
  standard, now screened as `h_ckda_std*`.
- The law-scaled LR (2.5e-3) gives s2 the same relative picture as s1. The LR transfer works
  for the ranking; the s2 optimum itself is not swept.

b*(D), angular MuOwn, s1, lr 4e-3, WSD 20 %, val loss at 3BT: gbs 64: 2.8949, **gbs 128:
2.8653**, gbs 512: 2.9363 (gbs 256 rerun: its index build hit the 10-min NCCL timeout, see
TODO_SPEED §0b). The optimum moves from 64-128 (1BT) to ≥ 128 (3BT), consistent with b* ∝
D^~0.46 (3^0.46 = 1.66x).

### 7.10 CKDA with standard inits, triadic CKDA, and why GDN differs from the multilingual sweep

**The screen's CKDA was not the reference CKDA.** `gen_screen.py`'s `CKDA` had the extended
ranges (signed_sigmoid2 alpha, beta_max 2; beta init standard / bias-free), but **spread**
alpha init and a **linear** output gate. The reference CKDA
(`config/experiments/korbi/complex_kda/sweep_spec.yaml`, ComplexKDA paper) has both inits
standard (alpha `shipped`, beta bias-free) and the low-rank gate. So the "KDA > CKDA" rows above
compare KDA against a non-standard CKDA. **Decision (user): the mixer is CKDA, always, with
standard inits; KDA is not an option to choose.** New arms (`gen_screen.py` `CKDA_STD`,
`TRI_CKDA`), matched by `count_params.py` (non-emb vs attention):

| arm | s1 d_ffn / non-emb | s2 d_ffn / non-emb |
|---|---|---|
| `h_ckda_std`: 3:1 CKDA (standard inits, low-rank gate) | 1984 / 85.13M (+0.21 %) | 2752 / 257.41M (+0.18 %) |
| `h_ckda_std_xsa`: + XSA in the attention layers | 1984 / 85.13M | 2752 / 257.41M |
| `h_tri_ckda`: 3:1 triadic CKDA, E = 4 (standard inits) | 1920 / 84.36M (-0.70 %) | 2688 / 256.43M (-0.20 %) |
| `h_tri_ckda_xsa`: + XSA | 1920 / 84.36M | 2688 / 256.43M |

Running with angular MuOwn: `screen_s1_std` (4e-3) and `screen_s2_std` (2.5e-3). Triadic + XSA +
CKDA had only been speed-tested (`tri_ckda_e4`, 0.54 of dense at 130M); this is its first loss
test. The s2 6BT references for KDA and the spread CKDA were cancelled. `ref6_s2_std` runs
`h_ckda_std` and `h_ckda_std_xsa` at 6BT instead, next to the running s2 attention /
attention + XSA references.

**All screen arms are parameter-matched** (non-embedding within 1 %, d_ffn per arm).

**Why the GDN hybrid fails here but beat attention in the multilingual sweep**
(`config/experiments/architecture_scaling_variants/multilingual/`):
1. *Optimizer*: that sweep is AdamW. Under AdamW our GDN hybrid also wins (-0.032 s1, -0.050
   s2, i.e. 1.0-1.7 %, in line with it). It falls behind only under angular MuOwn, which halves
   every hybrid's gain.
2. *Parameter matching*: that sweep's sizes were not matched. After pricing out the extra
   parameters, its GDN gains shrink to ≈ 40-55 % of raw (gdn7 -1.02 % -> -0.38 %, gdn3_qwen35
   -1.59 % -> -0.88 %; `dump/scaling_laws/*_nadj.csv`). Here d_ffn is trimmed instead.
3. *Layer setup*: its GDN arms use NoPE attention layers (`gdn3_nope`; qwen35 adds gated
   attention, partial RoPE, zero-centred norm) and GDN heads of 128 with k = query groups,
   v = heads. Ours: RoPE attention, GDN 12 x 64 (s1) / 16 x 64 (s2).
4. *Budget*: 20-300BT there, 2BT here. Hybrid gains may grow with tokens, which the 6BT
   references test (the losspred extrapolation in §7.7 predicts the opposite at fixed size).

So "GDN fails" holds for angular MuOwn at 2BT with our layer setup. It does not contradict the
multilingual result, which is AdamW, unmatched and NoPE.

### 7.11 Warmup, spike checks, where the curves are

**AdamW warmup was far too short.** The HP sweep used 1 % of the run, i.e. 19 / 38 / 76 steps at
gbs 128 / 64 / 32 (1BT). The AdamW screen used max(100, 1 %) = 100 steps. The usual setting is
≈ 1000 steps (multilingual sweep 2000, ComplexKDA ladder 1000-2000). The AdamW rows above are
therefore an almost warmup-free AdamW, and the gap to angular MuOwn (0.17 at 1BT) may partly be
the warmup. The MuOwn "no warmup needed" A/B (38 steps vs 0) only rules out a *short* warmup.
- `hp_s1_wu1k` (running, ≈ 11 GPU-h): the 1BT gbs-64 WSD calibration with a 1000-step warmup.
  AdamW at 1e-3 / 2e-3 / 4e-3, and angular MuOwn at 4e-3 (its warmup A/B). At 3814 steps, 1000
  warmup steps are 26 % of the run, so this also measures what a 1k warmup costs at this length.
- The generators now default to 1000 warmup steps (`gen_hpsweep.py` `aux.warmup_iters`,
  `gen_screen.py`); the warmup-free optimizers set 0 explicitly.

**Spike / instability checks.** `analyze.py` runs losspred's detection pipeline (hysteresis
spike detector + monotonicity regimes) only after max(2 x warmup, 100) steps, so by itself it
does not cover warmup instabilities. Adding losspred's warmup-instability criterion
(`scripts/scan_warmup_instabilities.py`: rolling median of 51 logged points vs its running
minimum, flag > 0.05) on all 129 runs, without the rolling median's 25 edge points:
- the largest rise in the first 3000 steps is 0.004;
- over the whole run it is 0.012 (AdamW 4e-3 gbs 32, at the same data position as other
  gbs-32 runs, i.e. the data order).
- No run is unstable. A common plateau at steps 2250-2750 in every gbs-64 run is data (same
  seed / order).

Megatron logs `lm loss` as the mean over log_interval = 10 iterations, so spikes shorter than
≈ 10 steps are smoothed out. Lower log_interval to 1 if those matter.

**Curves.** No wandb: MN5 compute nodes have no internet, so `wandb_project` is null and the
runs log to TensorBoard at `<run dir>/tb`
(`/gpfs/scratch/ehpc533/users/korbip/hybrid_v019_runs/autoexp/campaign_3b/<suite>/<arm>/tb`).
These can be synced into a wandb project from a machine with internet.

### 7.12 Stable phase only (user decision 2026-10-17)

No cooldowns. Runs are warmup + stable (or stable only), and are compared on the **stable-phase
loss**. Rationale: at equal gbs / lr the best model after a cooldown is the best during the
stable phase. Our screens confirm this. Ranking at the last validation before the cooldown
(step 6000) vs after it:

| screen (9 arms, same lr / gbs) | Spearman | pairs in the same order | max rank change |
|---|---:|---:|---:|
| s1 AdamW | 1.000 | 100 % | 0 |
| s2 AdamW | 0.983 | 97 % | 1 |
| s1 angular MuOwn | 0.983 | 97 % | 1 |
| s2 angular MuOwn | 1.000 | 100 % | 0 |

- Stable-phase gaps are 10-30 % larger than cooled ones (the cooldown compresses, does not
  reorder; the two swaps are pairs within ≈ 0.005).
- *Across* LRs the cooldown does reorder (§7.4: MuOwn 2.5e-4 -> 5e-4, angular MuOwn 2e-3 ->
  4e-3 after cooling). If the production run has no cooldown either, the LR is chosen on the
  stable loss. The WSD calibration runs (`hp_s1w*`) are kept as the record, and no new ones are
  started (`hp_s1_wu1k` was cancelled).
- `gen_screen.py` is now stable-only (constant LR after 1000 warmup steps). Running jobs that
  still have a 20 % cooldown (`screen_s1_std`, `screen_s2_std`, the 6BT references) are read at
  their last pre-cooldown validation.
- The 6BT "reference trainings" thus give stable-phase loss at 4.8BT (80 %) from the running
  ones; future references run stable to the full budget.

### 7.13 Warmup study: the standard transformer (done, `hp_wu_study`)

s1 attention + qk-norm (85M non-emb), AdamW (beta2 0.95, clip 1.0), gbs 128 x 4096, exactly
2000 steps (1.05BT), constant LR after the warmup, every step logged. Spikiness from losspred's
hysteresis detector over the whole run (from step 20); the common +0.015 bump at step 1592 is
the data order.

| lr | warmup | val @2000 | train (last 100) | spikes | largest | max grad norm |
|---|---:|---:|---:|---:|---:|---:|
| 2e-3 | 50 | 3.389 | 3.404 | 0 | - | 5.1 |
| 2e-3 | **200** | **3.339** | 3.356 | 0 | - | 4.1 |
| 2e-3 | 1000 | 3.381 | 3.401 | 0 | - | 5.4 |
| 4e-3 | 50 | 3.465 | 3.483 | **1** | **+0.255** | **11.4** |
| 4e-3 | 200 | 3.403 | 3.418 | 0 | - | 6.6 |
| 4e-3 | 1000 | 3.378 | 3.396 | 0 | - | 6.4 |

- **A short warmup at high LR leaves AdamW fragile well after the warmup.** At 4e-3 / 50 steps a
  hard batch at step ≈ 1650 gives grad norm 11.4 and +0.32 in the 25-step mean loss (recovered
  in ≈ 25 steps). The longer-warmup runs show only the shared +0.03 data bump there. The loss
  improves monotonically with the warmup at 4e-3.
- At 2e-3 nothing spikes, but 200 steps is best. 50 steps costs 0.050. 1000 steps costs 0.042
  at a 2000-step budget, where half the run is warmup (less LR area); this cost shrinks with
  run length.
- The round-1 AdamW numbers (19-100 warmup steps) are therefore ≈ 0.05 too pessimistic at
  their LRs (the 50-vs-200 gap at 2e-3). That is about a third of angular MuOwn's lead.
- The detector flags exactly the one real spike and none of the other runs. With
  log_interval 1 it resolves the 25-step event; at log_interval 10 it would appear as ≈ +0.1 in
  ≈ 3 logged points.
- Running: the same study for angular MuOwn (lr 4e-3, warmup 0 / 50 / 200 / 1000,
  `hp_wu_study_am`, ≈ 6 GPU-h), so that the optimizers are compared at their best warmup each.

## 8. Round 1b: the valid setup, learning-curve fits for every run (2026-10-17)

The round-1 arch screens are not a valid comparison: AdamW had 19-100 warmup steps (§7.11,
§7.13), and the angular MuOwn screens had none. That also ruled out strong competitors (the
Qwen3.5-like GDN hybrid) under conditions where they cannot be judged. Round 1b redoes the
small-scale arch comparison in a valid setup. Every run is learning-curve fitted, with health
indicators, so that later-stage loss is estimated from the early stage rather than training
out, and the fitted parameters are then modelled from the hyperparameters.

### 8.1 Screen 2 (running: suites `screen2b_s1`, `screen2b_s2`, ≈ 190 GPU-h)

- **Setup**: AdamW (beta2 0.95, clip 1.0), **1000-step warmup**, then **stable** (constant LR,
  no cooldown), gbs **128** x 4096, **4BT (7629 steps, ≈ 6.6k stable)**, datamix4, every step logged. lr: s1 2e-3;
  s2 1.25e-3 (s1 x the ComplexKDA law (N2/N1)^-0.417).
- **Arms** (parameter-matched within 0.7 %, d_ffn per arm, `param_match.json`):

  | arm | what | s1 d_ffn | s2 d_ffn |
  |---|---|---:|---:|
  | attn | attention + qk-norm, RoPE (base) | 2048 | 2816 |
  | attn_xsa | + exclusive self attention | 2048 | 2816 |
  | attn_gate | + sigmoid output gate | 1792 | 2496 |
  | h_ckda_std | 3:1 CKDA (standard inits, low-rank gate) | 1984 | 2752 |
  | h_ckda_std_xsa | + XSA | 1984 | 2752 |
  | h_tri_ckda | 3:1 triadic CKDA, E = 4 | 1920 | 2688 |
  | h_tri_ckda_xsa | + XSA | 1920 | 2688 |
  | h_gdn | 3:1 GDN, our setup (12 / 16 heads of 64, RoPE attention) | 1856 | 2560 |
  | h_gdn_nope | multilingual `gdn3_nope`: GDN heads of 128, NoPE attention | **896** | **1280** |
  | h_gdn_qwen35 | Qwen3.5-like: + attention output gate, partial RoPE 25 %, zero-centred norm | **832** | **1152** |

  The two multilingual GDN arms have large GDN layers (heads of 128, k = query groups,
  v = heads: +32M params at s1). Matched, their MLP shrinks to ≈ 40 % of the base's. In the
  multilingual sweep they ran unmatched with the full MLP. If they lose matched, an unmatched
  pair (same d_ffn as attn, compared by the measured-N scaling term) separates "GDN is worse"
  from "this GDN width is not worth its parameters".
- An angular MuOwn screen 2 follows once `hp_wu_study_am` fixes its warmup (same arms, its
  own best warmup; a matched-warmup optimizer comparison).

### 8.2 Learning-curve fits and health for every run (`scripts/korbi/campaign/curves.py`)

Per run, from the Megatron log:
- **Fit**: losspred's model `L = E + A (tau + t0)^-alpha` in LR-area time, anchored at the
  initial loss, on the stable phase after the warmup (from max(warmup, 200)). Gives E, A,
  alpha, t0, lr_power with standard errors and the trend RMSE.
- **Prediction**: the stable trend continued to later budgets (6BT, 20BT).
- **Backtest**: fit the first half, predict the stable end; the extrapolation error.
- **Health**: spikes (hysteresis detector, whole run incl. warmup; count, height,
  excess), growth events (monotonicity detector), flattening (late residual vs the
  backtest fit), noise.

First results:
- Long runs (2BT at gbs 64, 7.6k steps): trend RMSE 0.004; the first half predicts the end
  to +0.4-0.6 %. The sign is consistent: the anchored single power law slightly underpredicts
  late improvement.
- Short runs (2000 steps, 1-2k stable): the fit is good in-sample, but E is not identified
  (it lands near the current loss and the extrapolation saturates). Single-run extrapolation
  needs a few thousand stable steps. For short screens, alpha (and lr_power) are shared across
  the arms of a size in a joint fit, or come from a prior (the ComplexKDA ladder / the base
  arm's longer run).
- All round-1 runs are fitted into `dump/campaign_3b/curves_round1.{csv,md}`.

### 8.3 Parameters from hyperparameters

As losspred's `param_relations.py` does for its wandb grid: regress log(E), alpha, lr_power,
the noise and the spike / instability measures on log(lr), log(gbs), log(N) and the warmup, per
optimizer and arch, over the fitted runs (HP sweeps + screens + the warmup study). Goal: predict
a configuration's curve, and its health, from its hyperparameters, and read off the optimal lr
/ gbs and its scaling with N and D (the b* ∝ D^0.46 check in §7.9) from the fitted surface
rather than from grid minima. This needs the round-1b runs, which cover lr x gbs x warmup at a
valid warmup.

### 8.4 Screen 2 restart (2026-10-17): 4BT, and two config fixes

- **4BT instead of 2BT.** The round-1 fits extrapolate only past ≈ 6.5k stable steps:
  backtest within 1 % for 47/68 runs with > 6.5k stable steps, 8/44 at 2.5-6.5k, 6/18 below.
  At gbs 128, 2BT is only 2.8k stable steps. At 4BT each run has a usable fit, and its second
  half checks the first half's extrapolation. The first launch (2BT) was cancelled after 8 min.
- **`h_gdn_nope` crashed:** `fused_qk_norm_rope needs: RoPE` (the speed base fuses q/k norm +
  RoPE). NoPE arms set `fused_qk_norm_rope: false`.
- ~~The fused q/k-norm + RoPE kernel ignores zero-centred gamma~~ **(wrong, retracted)**: the
  kernel scales by w, but its call site (`attention.py` `_fused_qk_norm_rope`) already passes
  (1 + w) under zero-centred gamma. The qwen35 arm would have been correct with the fusion. The
  guard added for it was reverted. Screen 2's two multilingual GDN arms run unfused
  (`fused_qk_norm_rope: false`), which is numerically the same, only slower.
- **q/k-norm with NoPE in the fused path (2026-10-17).** The unfused path always applied q/k
  RMSNorm with NoPE; only the fused kernel required RoPE. `fused_qk_norm_rope(freqs=None)` now
  norms without rotating (an identity rotation by angle 0 on the smallest 2-channel block), and
  the attention layer no longer lists RoPE as a requirement. Tests:
  `tests/unit_tests/fusions/test_fused_qk_norm_rope.py` `test_nope_norm_only` (fwd + bwd + weight
  grads vs the reference and a plain RMSNorm, fp32 / bf16, with / without gate) and
  `test_zero_centered_gamma_weights` (the (1 + w) call path); 25 / 25 pass. Future NoPE and
  qwen35 arms keep the fusion on.
- **`h_tri_ckda*` at s2: OOM at mbs 8 → mbs 4** (gbs unchanged).

### 8.5 Warmup study for angular MuOwn (done, `hp_wu_study_am`)

Same setup as §7.13 (s1 attention, gbs 128, 2000 stable steps, every step logged), angular
MuOwn at lr 4e-3:

| warmup | val @2000 | spikes | max grad norm |
|---:|---:|---:|---:|
| **0** | **3.222** | 0 | 4.8 |
| 50 | 3.227 | 0 | 22.3 |
| 200 | 3.233 | 0 | 7.9 |
| 1000 | 3.281 | 0 | 11.4 |

- **Angular MuOwn needs no warmup** and does not spike without one. A warmup only costs LR area
  (0.06 for 1000 steps at this length).
- **Optimizers at their best warmup each, same budget**: angular MuOwn 3.222 vs AdamW 3.339
  (200 steps, 2e-3) -> **0.117**. With 1000 warmup steps for both: 3.281 vs 3.381 -> 0.10. So the
  round-1 AdamW warmup handicap explained only part of the gap.
- Angular MuOwn screen 2 (same arms, warmup 0, lr 4e-3 / 2.5e-3) follows once the fused-NoPE
  kernel test has passed on MN5 (`kt_qknorm`).

## 9. The procedure, as learned in round 1 / 1b (for the < 10k GPU-h search)

A concrete version of the general design in `docs/optimal_architecture_search_procedure.md`
(curve-model layer, HP response surface, envelope law, sequential design), with the fixes round
1 found. Each step names its tool and its gate.

### 9.1 Valid setup (all small-scale comparisons)

1. **Schedule: warmup + stable, no cooldown.** Compare on the stable-phase loss. At equal
   lr / gbs the stable ranking equals the cooled one (§7.12: Spearman 0.98-1.00 over 4 screens).
   The LR optimum is chosen on the stable loss too (production has no cooldown).
2. **Warmup per optimizer, measured** (§7.13, §8.5): AdamW 1000 steps (200 is best at 2000-step
   budgets, but 50 spikes at high LR); angular MuOwn none (0 is best and spike-free). Never
   the 1 % default (19-100 steps).
3. **Batch size: gbs 128 x 4096 at ~100M** ("common value"; the measured b* at 1-3BT is 64-128,
   growing ≈ D^0.46, §7.8 / §7.9).
4. **Length: ≥ 6.5k stable steps per run**, so that a run's own curve fit extrapolates (backtest
   within 1 % for 47/68 such runs, 8/44 at 2.5-6.5k, §8.4). At gbs 128 that is 4BT.
5. **Parameter matching**: non-embedding within 1 %, d_ffn solved per arm by
   `count_params.py` (Megatron's own count, `param_match.json`). Report the d_ffn: wide mixers
   (GDN heads of 128) cost most of the MLP when matched.
6. **Reference mixer configs**: CKDA with both inits standard (§7.10; `CKDA_STD`), never the
   spread / linear-gate variant. q/k L2-normalized in every linear mixer (KDA, CKDA, triadic,
   GDN), qk-RMSNorm in attention (also with NoPE, now fused).
7. **One data cache per run** (concurrent index builds corrupt a shared one), NCCL timeout 45
   min (the 156-prefix index build can exceed 10 min on GPFS). Node-local prefetch
   (`+data_staging=mirror_mn5`) if many small runs stall on GPFS.
8. **Log every step** (log_interval 1): the spike detector resolves 25-step events; at 10 they
   are averaged away.
9. **Dry-run every sweep** and check the rendered task arguments: interpolations inside sweep
   entries resolve against the base config (§7.4 note, memory), and wrong cooldown / min_lr
   values slipped through twice.

### 9.2 Per-run analysis (every run, automatically)

`scripts/korbi/campaign/refresh_curves.sh` -> `dump/campaign_3b/curves_all.{csv,md}`:
- **Health**: spikes (hysteresis detector, whole run), growth events, flattening, noise.
  Unhealthy = growth or a spike above 0.05.
- **Curve fit**: E, A, alpha, t0 (+ standard errors) in LR-area time, anchored at the initial
  loss, from max(warmup, 200).
- **Backtest**: fit the first half, predict the end. "Extrapolable" = within 1 %.

Open issue: the anchored single power law under-predicts late improvement by a systematic
+0.4-0.6 % (all arms, both sizes). Ruled out:
- the data order (correcting by the shared train - val gap makes it +1.8-2.2 %);
- dropping the anchor (+1.2-1.5 %).
The two-component forms of losspred's `compare_stable_forms.py` (power law + exp, two power
laws; `curves.py --form`) do not help either: +1.3-1.6 %, their extra component collapsing at
2BT. The bias is very consistent, +0.40 to +0.63 % over the 18 long runs (both sizes, all arms),
so a calibration (prediction x ≈ 0.9945) gives ≈ ±0.1 % for a 2x extrapolation. That is
empirical and only for that distance: the 6BT references (3x) and the screen-2 4BT runs check
whether it holds further out. Arm *rankings* are unaffected (same sign and size of the bias
for every arm).

### 9.3 Decisions (gates)

1. **Feature gate**: an arm beats the base at matched parameters, at equal lr / gbs, at **two
   sizes** (s1 ≈ 85M, s2 ≈ 257M non-emb), by more than ≈ 3x seed noise (0.002 -> ≈ 0.006), on
   the stable loss. Run under the chosen optimizer (§7.5: the AdamW-screen ranking does not
   carry over to angular MuOwn).
2. **Optimizer**: each at its own best warmup and a swept LR. Angular MuOwn leads AdamW by 0.10-0.12
   at equal budget (§8.5); confirmed per arm by the two screen-2 runs.
3. **Hyperparameters**: LR x gbs on stable trunks, the optimum from a fitted surface (losspred
   `param_relations.py`-style regressions of the curve parameters on lr, gbs, N, warmup) rather
   than grid minima. Size transfer by the ComplexKDA law (eta ∝ N^-0.417; verified s1 -> s2 for
   the ranking).
4. **Speed**: tok/s/GPU only (the frameworks count FLOPs differently). The 3B-active layout is
   chosen at its real size (§7.2: TP1 / PP4 x VPP2 with the head on its own segment / EP4
   HybridEP, mbs 4).
5. **Scale-up**: survivors -> 6BT references at s1 / s2 (stable) to check the extrapolations ->
   the third / fourth size and the MoE ladder (§4).

### 9.4 Budget so far

≈ 1.7k of the 2k round-1 GPU-h committed (speed work + round 1 + round 1b). The round-1b screens
(both optimizers, two sizes, 10 arms, 4BT) give the feature decisions in the valid setup. The
remaining ≈ 300 GPU-h go to:
- the s1 / s2 6BT stable references of the survivors (≈ 200);
- the flag-data switch once it has landed (≈ 97 % on 2026-10-17).

### 9.5 6BT references, s1 (running: `ref6v_s1`, ≈ 40-65 GPU-h)

Valid setup: angular MuOwn (no warmup, lr 4e-3), stable, gbs 128, 6BT (11444 steps), datamix4
(flag replaces it once fully landed, as decided), every step logged. Arms: attention, attention
+ XSA, CKDA (standard inits), CKDA + XSA, and the Qwen3.5-like GDN hybrid (the strongest
outside competitor). They test the 2BT/4BT curve extrapolations (§9.2, incl. the ≈ 0.9945
calibration) at 3x, and give the 6BT stable-phase comparison of the main candidates. The s2
references follow once screen 2 has picked the survivors at s2.

### 9.6 Short runs + fitted parameters (user decision 2026-10-17)

Runs are only *started*: a few thousand steps, then compared through the fitted curves instead of
being trained out. A single short run cannot be fitted (E and alpha trade off: at < 3k stable steps
E = 0.001 ± 25 or alpha = 2 ± 433). So the arms of a suite (one size, one optimizer) are fitted
**jointly**: `scripts/korbi/campaign/joint_fit.py`.

Model: a shared exponent alpha, and per arm E_i and A_i, anchored at each arm's initial loss, in
LR-area time.

Validation on the two complete angular MuOwn screens (gbs 64, 7.6k steps): fit up to step N,
predict step 6000 (the last stable point), compare with the measured value.

| fit to step | shared alpha (s1) | ranking Spearman s1 / s2 | mean error | per-arm error spread | raw loss at N: Spearman / spread |
|---:|---|---|---:|---:|---|
| 2000 | 0.11 ± 0.14 (unidentified) | 0.87 / 0.75 | -3.5 % | 0.009-0.011 | 0.93-0.95 / 0.005 |
| 3000 | 0.57 ± 0.06 | 0.98 / 0.88 | +0.9 % | 0.004 | 0.95-0.97 / 0.002-0.004 |
| 4000 | 0.43 ± 0.03 | 0.98 / 0.90 | +0.2 % | 0.003 | 0.95-0.98 / 0.001-0.002 |

- From ≈ 3-4k steps the joint fit identifies alpha (± 0.03-0.06) and predicts the 2x-later loss to
  +0.2-0.9 %. The arm differences come out with ≈ 0.003-0.004 nats error, against arm gaps of
  0.01-0.04.
- **For ranking at a 2x horizon, the raw loss at the cut is as good** (shared alpha gives every
  arm the same curvature). The fit adds:
  - absolute predictions;
  - the parameters (alpha, E, A) for the HP / scaling model;
  - long-range extrapolation where curves differ in shape.
  The last one needs validation further out: the 6BT references (11.4k steps, i.e. 3-4x from a
  3k cut) and three full-length arms per screen-2 suite.
- **Protocol**:
  - every arm runs to ≈ 4000 steps (after AdamW's 1000-step warmup: 3000 stable);
  - per suite, attention, CKDA (std) + XSA and the Qwen3.5-like arm run to the end as validation;
  - the other screen-2 arms are stopped at step 4000 (a watcher on MN5, 2026-10-17);
  - comparisons use the joint fit's predicted loss at the target stage, with the raw loss at the
    cut as a check.
- Next for the fit: a per-arm alpha deviation with a shrinkage prior, if the long-range
  validation shows shape differences (the single-run fits of the long screens put the hybrids'
  alpha above attention's: 0.42-0.46 vs 0.36-0.37).

### 9.7 The fit form, and per-run parameters

Per run, stable phase, no cooldown, **three free parameters**:

    L(t) = E + A (tau(t) + t0)^(-alpha)
    tau(t) = integral_0^t lr(s) / lr_max ds   (LR area: t without warmup, t - W/2 after a linear warmup W)
    t0 = (A / (L0 - E))^(1/alpha)             (derived: anchored at the measured initial loss L0 ≈ ln(vocab))

- Fitted on log L: block means, noise-weighted residuals, soft-L1, from step max(W, 200).
- losspred's cooldown terms (drop, drop_exponent, shape) and lr_power are unidentifiable without
  a cooldown and held fixed.

Each run has its own (E, A, alpha). `joint_fit.py` fits a suite's runs together, with a weak
Gaussian prior (width 1.0) tying the exponents. It regularises the short fits without forcing
one alpha (a shared alpha is the width-0 limit). Validation, fit to step 4000, predict step 6000:

| exponent | arm-to-arm error spread s1 / s2 (nats) | ranking Spearman s1 / s2 |
|---|---|---|
| shared (prior width 0) | 0.0027 / 0.0035 | 0.98 / 0.90 |
| prior width 0.1 | 0.0009 / 0.0016 | 0.98 / 0.97 |
| **per run (prior width 1.0, default)** | **0.0017 / 0.0006** | **0.98 / 1.00** |
| raw loss at the cut | 0.0020 / 0.0013 | 0.98 / 0.95 |

Mean offset +0.2 % (common to all arms). A 2000-step cut is too short for every variant (-3.5 %).

Per-run parameters (angular MuOwn screens, fit to step 4000):

| arm | s1 E / A / alpha | s2 E / A / alpha |
|---|---|---|
| attention | 2.670 / 12.1 / 0.366 | 2.511 / 13.7 / 0.378 |
| attention + XSA | 2.734 / 13.5 / 0.400 | 2.500 / 13.7 / 0.381 |
| attention + gate | 2.748 / 15.2 / 0.414 | 2.591 / 17.4 / 0.425 |
| KDA hybrid | 2.775 / 16.9 / 0.441 | 2.533 / 16.5 / 0.412 |
| KDA hybrid + XSA | 2.776 / 18.1 / 0.452 | 2.557 / 18.5 / 0.433 |
| CKDA (spread init) hybrid | 2.819 / 21.6 / 0.480 | 2.590 / 21.2 / 0.451 |
| CKDA (spread init) + XSA | 2.792 / 19.3 / 0.460 | 2.599 / 22.8 / 0.464 |
| GDN hybrid | 2.718 / 12.8 / 0.383 | 2.556 / 15.6 / 0.404 |

The hybrids have larger alpha and A and a higher E than attention, at both sizes. They improve
faster now, towards a higher floor. Taken literally, the curves cross much later: at a fixed size
the hybrids' advantage shrinks with tokens. How far out these per-run fits extrapolate (alpha vs E
trade-off) is what the 6BT references (3-4x) test.

### 9.8 Per-run parameters with bootstrap intervals; parameters and health from hyperparameters

**Goal**: fit (E, A, alpha) for *every* curve, then predict those parameters, and health,
from the hyperparameters. No single shared fit.

**Per-run fits + bootstrap** (`joint_fit.py --bootstrap 50`). Each run is refitted 50 times on
a moving-block resample (blocks of 5) of its own curve residuals, with the weak exponent prior
centred on the suite's alpha. Output: 5-95 % intervals for E, A, alpha and the predicted loss.
The resampling indices are the same for all arms of a suite, so the replicas are **paired** and
the gap to a reference arm (`--reference attn`) gets its own interval. Angular MuOwn screens,
fit to step 4000, predict step 6000:
- per-run parameters are individually loose: alpha ± ≈ 0.15, E ± 0.2-0.5, A within a factor
  2-3 (E and alpha trade off within one run);
- each run's predicted loss: ± 0.02-0.025, with the actual errors (+0.1-0.25 %) inside;
- the paired gaps are tight, ± ≈ 0.001. s1: KDA + XSA -0.037 [-0.038, -0.035], CKDA (spread)
  -0.017, attention + XSA -0.020, gate -0.005, GDN +0.007 [+0.006, +0.008]. s2: GDN +0.0001
  [-0.0008, +0.0011], gate +0.003.
- The bootstrap is conditional on the curve form, so it is a lower bound: measured gaps fall up
  to ≈ 0.0015 outside it (e.g. KDA + XSA s1 measured -0.0397). Report paired interval ⊕ the
  empirical arm-to-arm error (0.0006-0.0017 nats).

**Parameters from hyperparameters** (`hp_model.py`; first pass on the 63 s1 HP-sweep runs:
1BT, lr x gbs x warmup, 4 optimizers). Per optimizer, OLS on log lr, log gbs, log(1 + warmup)
(N is constant: all s1):
- alpha and log A are predictable: R² 0.55 (AdamW), 0.66-0.72 (angular MuOwn), 0.78-0.81
  (MuOwn). For both MuOwn variants, alpha falls with lr (-0.20 to -0.22 per e-fold) and rises
  with gbs (+0.14 to +0.24).
- log E is not (R² 0.06-0.27), the per-run E-alpha trade-off again on these short runs. Weighting
  by the bootstrap intervals (next) and longer / denser runs are needed before E can be modelled.

**Health from hyperparameters**: a linear threshold (logistic regression on the standardised
hyperparameters and an optimizer one-hot; unhealthy iff w·x > θ) is implemented, but only 1 of 63
runs is unhealthy (AdamW 4e-3, warmup 50), so its boundary means nothing yet. The **health grid**
(`hp_health`, 24 runs, ≈ 40 GPU-h, running) is designed to cross it, on the s1 attention base,
2000 steps, every step logged:
- AdamW lr 4e-3 / 8e-3 / 1.6e-2 x warmup 0 / 50 / 200 x gbs 64 / 256;
- angular MuOwn lr 8e-3 / 1.6e-2 / 3.2e-2 x gbs 64 / 256 (no warmup; its runs so far are healthy
  up to 8e-3).

### 9.9 Compare at equal seen tokens D, not steps (user note 2026-10-17)

Runs are compared at equal **seen tokens D = steps x gbs x seq_len**. For an architecture
decision both N (parameter matching) and D are matched. Within a screen (one gbs, one sequence
length) step and D coincide, so the screen results above are D-matched. Across batch sizes (HP
sweeps, health grid) they are not.
- `joint_fit.py` / `hp_model.py` work in tokens: `--cut` / `--target` are token counts, the curve
  time tau is the LR area in tokens, and A is in token units. E and alpha do not depend on the
  time unit (re-running the §9.7 validation in tokens reproduces every E, alpha, prediction
  and error). A did (A_steps x (gbs·seq)^alpha), and is now comparable across batch sizes.
- `curves.py` (single-run fits, health) still fits in steps. Its E / alpha are unaffected, its
  A is per-run step-based, and its predictions are already given at token budgets.

### 9.10 Cut points from the existing long curves (`cut_study.py`)

For each cut D_cut: fit every arm up to D_cut (per-run E, A, alpha; `joint_fit.py`), predict the
stable loss at D_target = 1.57BT (the last stable point of the round-1 2BT screens, gbs 64), and
compare with the measured loss. **Gap error** = RMS over arms of the error in the predicted gap
to attention, in nats: what an architecture decision depends on.

| D_cut (fraction of target) | angular MuOwn s1 / s2: fit gap error | raw loss at the cut | fit ranking | AdamW s2 (100-step warmup): fit / raw gap error, ranking |
|---|---|---|---|---|
| 0.39BT (25 %) | 0.0060 / 0.0091 | 0.0151 / 0.0089 | 0.95 / 0.82 | 0.014 / 0.042, 0.83 |
| 0.52BT (33 %) | 0.0168 / 0.0179 | 0.0128 / 0.0049 | 0.80 / 0.75 | 0.034 / 0.027, 0.65 |
| 0.66BT (42 %) | 0.0094 / 0.0032 | 0.0104 / 0.0034 | 0.98 / 1.00 | 0.017 / 0.020, 0.68 |
| **0.79BT (50 %)** | **0.0027 / 0.0009** | 0.0098 / 0.0032 | **1.00 / 1.00** | 0.011 / 0.015, 0.82 |
| 1.05BT (67 %) | 0.0032 / 0.0007 | 0.0055 / 0.0028 | 1.00 / 1.00 | 0.007 / 0.008, 0.85 |
| 1.18BT (75 %) | 0.0026 / 0.0005 | 0.0024 / 0.0009 | 1.00 / 1.00 | 0.005 / 0.005, 0.90 |

- **Angular MuOwn: cut at ≈ 50 % of the target budget.** From there the per-run fits predict every
  arm's gap to attention to 0.001-0.003 nats with perfect rankings, 1.5-4x better than the raw
  loss at the cut. At a 2x horizon, the fits pay off from half the budget on.
- **AdamW extrapolates worse**: gap error 0.011 at 50 %, 0.005 at 75 %, rankings 0.82-0.90. Its
  arms differ more in curve shape. This screen still had the 100-step warmup. The AdamW screen-2
  validation arms (1000-step warmup, 4BT) check whether that holds at a valid warmup.
- **The data order matters for the cut**: the worst cut (0.52BT = step 2000 at gbs 64,
  -3.5 %, gap error 0.017) is just before the stretch of harder batches that every run sees at
  steps 2250-2750 (§8.4 note: the shared train - val gap). Cuts inside or right before such a
  stretch mislead the fit. Alternatively, the shared data-difficulty term can be modelled.
- **Longer horizons** (3-4x, the 6BT references, `ref6v_s1`) are next. These numbers are all for
  2x.
- Working rule until then: run to ≥ 50 % (angular MuOwn) / ≥ 75 % (AdamW) of the budget at which
  arms are compared. The screen-2 runs stop at 4000 steps of a 7629-step budget (52 %),
  validated by the full-length arms.

### 9.11 Screen-2 validation at 3.9BT: AdamW holds, angular MuOwn does not (2026-10-17)

The screen-2 validation arms (attn, h_ckda_std_xsa, h_gdn_qwen35) ran the full 4BT (7629 steps,
gbs 128); all other arms were stopped at 4000 steps (2.1BT). `cut_study.py`, target 3.9BT,
reference attn (gap error = RMS over arms of predicted - measured gap to attn, nats):

| suite | cut 1.57BT (40 %) | 2.10BT (54 %) | 2.62BT (67 %) | 3.15BT (81 %) | raw loss at 2.10BT |
|---|---:|---:|---:|---:|---:|
| screen2b_s1 (AdamW) | 0.0020 | 0.0018 | 0.0018 | 0.0017 | 0.0109 |
| screen2am_s1 (angular MuOwn) | 0.0123 | 0.0065 | 0.0040 | 0.0026 | 0.0077 |
| screen2am_s2 (angular MuOwn) | 0.0070 | 0.0121 | 0.0089 | 0.0059 | 0.0058 |

- AdamW: the per-run fits hold from 40 % of the target on (gap error 0.002, ranking exact).
  This is better than the round-1 cut study (§9.10, ≥ 75 %), because the 1000-step warmup
  removes the early-transient bias.
- Angular MuOwn: neither the fits nor the raw loss reach 0.005 nats before ~70-80 %. All runs are
  healthy (no spikes, no growth). The gaps simply keep moving. Measured gap to attn (s1, 300-step
  window means):

  | step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 7.6k |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | h_ckda_std_xsa | -0.020 | -0.029 | -0.027 | -0.025 | -0.024 | -0.022 | -0.022 | -0.021 |
  | h_gdn_qwen35 | -0.030 | -0.022 | -0.015 | -0.009 | -0.006 | -0.002 | -0.000 | +0.001 |

  The CKDA hybrid's lead is stable. The Qwen3.5-like GDN hybrid's lead shrinks linearly in D and
  is gone by 4BT, at s2 as well (-0.003 at 7.6k). That curve-shape difference is not captured by
  the per-run fit (it predicts -0.007). Extrapolating the gap directly
  (g_inf + c D^-beta, `cut_study.py` "gap trend") does not fix it (0.004-0.011).
- AdamW shows the same direction more weakly: qwen35's gap goes from -0.165 at 1k to -0.043 at 7.6k
  steps, and the CKDA hybrid's from -0.111 to -0.058. Hybrid gains shrink with D under both
  optimizers, and faster under angular MuOwn.

Consequences for rounds 3 / 3B:
- AdamW short runs (4000 steps) are decision-grade for gaps ≥ ~0.004 nats (2 x the gap error).
- Angular MuOwn short runs only separate gaps ≥ ~0.01-0.015 nats. Closer calls are decided on
  AdamW, or by extending the two contenders to the full 4BT (~6 GPU-h each at s1). Round 3 already
  runs attn_ref and h3 to 4BT under both optimizers.
- Every hybrid claim reports the gap trend (gap at 2k / 3k / 4k steps), not only the fitted
  endpoint. A gap that shrinks steadily under angular MuOwn (as with qwen35) counts as "no
  benefit at scale" until a longer run shows otherwise.

### 9.12 The angular MuOwn fit problem was the fit start, not the fit form (2026-10-18)

The fits start at max(warmup, 200) steps: step 1000 for AdamW (after its warmup), but step 200
for angular MuOwn (no warmup), which is inside the early transient. There, curves of different
variants cross wildly (routed residual -0.14 vs h3 at step 300, AttnRes +0.34). Other fit
settings do not help: a shared exponent per suite (alpha prior 0) is worse, a tight prior (0.1)
is equal, and the gap-trend extrapolation reaches 0.005-0.012. Starting the fit at step 1500
(`--fit-start 1500`, joint_fit.py / cut_study.py) does help. Gap error vs the measured 3.9BT
gaps:

| suite | fit start | cut 1.57BT (40 %) | 2.10BT (54 %) | 2.62BT (67 %) |
|---|---:|---:|---:|---:|
| screen2am_s1 | 200 | 0.0123 | 0.0065 | 0.0040 |
| screen2am_s1 | 1500 | 0.0047 | **0.0008** | **0.0005** |
| screen2am_s2 | 200 | 0.0070 | 0.0121 | 0.0089 |
| screen2am_s2 | 1500 | 0.0025 | **0.0048** | **0.0021** |
| screen2b_s1 (AdamW) | max(1000 warmup, 200) | 0.0020 | 0.0018 | 0.0018 |
| screen2b_s1 (AdamW) | 1500 | 0.0109 | 0.0058 | 0.0027 |

Round-3 check (screen3_am 4BT pair, cut 2.1BT, target 3.9BT): h3 vs attn_ref predicted
-0.0159 with start 1500 (-0.0057 with start 200), measured -0.0189.

Rule: fit from max(warmup, 1000) steps for AdamW and from step 1500 (0.8BT at gbs 128) for
the no-warmup optimizers. §9.11's "angular MuOwn short runs only separate ≥ 0.01-0.015" is
superseded: with the later start, angular MuOwn fits are decision-grade like AdamW's
(≈0.001-0.005 at 54-67 %).

Round 3, angular MuOwn, refitted (start 1500, paired vs attn_ref_4k, 5-95 %): engram -0.070,
mhc4 -0.053, tri -0.021, h3 -0.018, nolxsa -0.018, h5 -0.014, routed_key -0.012,
routed -0.004, attnres +0.002, qwen35 +0.009. These agree with the raw 4k gaps in §10.5.

## 10. Level-1 round 3: the hybrid recipe, residuals and memories (s1, ≤ 1k GPU-h, user 2026-10-17)

**The hybrid recipe** (user):
- every hybrid has **NoPE** attention layers;
- its attention layers get **qk-norm + XSA + the output gate** (what helped plain attention);
- **XSA also in the linear layers** (`linear_xsa`: CKDA via the ComplexKDA fork's `xsa`, triadic
  and GDN natively);
- CKDA always with standard inits;
- the Qwen3.5-like arm stays fully Qwen3.5-like.

Then the residual variants (mHC, block attention residuals, our routed residual) and Engram,
which had only been speed-tested, are compared on the recipe.

Setup: s1 (hidden 768 x 12 blocks), datamix4, gbs 128 x 4096, stable, D-matched. Short runs end
at 4000 steps (2.1BT) through the config; `attn_ref` and `h3` run 4BT (validation).
Checkpoints: rolling non-persistent + one final. Both optimizers at their own best warmup:
- AdamW, 1000-step warmup, lr 2e-3;
- angular MuOwn, no warmup, lr 4e-3.

| arm | definition | d_ffn (non-emb vs attn_ref) |
|---|---|---|
| attn_ref (+ seed 2) | attention + qk-norm + XSA (RoPE): the reference transformer | 2048 |
| **h3** (+ seed 2) | 3:1 CKDA (std) + linear XSA; attention NoPE + qk-norm + XSA + gate | 1920 (+0.21 %) |
| h5 | 5:1, same recipe: 2 attention layers of 12, 6x less KV cache (12 blocks do not divide into 7:1) | 1920 (-0.23 %) |
| h3_tri | triadic CKDA (E = 4), same recipe | 1856 (-0.70 %) |
| h3_nolxsa | h3 without linear XSA (ablation) | 1920 (+0.21 %) |
| qwen35 | Qwen3.5-like GDN hybrid | 832 (+0.39 %) |
| h3_mhc4 | h3 + mHC, 4 residual streams | 1856 (+0.21 %) |
| h3_routed | h3 + routed residual (2 of 4 streams) | 1920 (+0.38 %) |
| h3_attnres | h3 + block attention residuals (4 blocks) | 1920 (+0.25 %) |
| h3_engram | h3 + Engram n-gram memory before blocks 2 and 8 | 1920: **not matched**, + ≈ 271M table params (sparse lookups, like embeddings; matching them would need a negative MLP width) |

Analysis: `joint_fit.py` per suite (per-run E, A, alpha; paired bootstrap gaps vs `attn_ref` and
vs `h3`), `curves.py` health, cut validation on the two full-length arms.

Cost: ≈ 12 runs x 2 optimizers, ≈ 150-200 GPU-h, after a 30-step smoke of every arm (the
residual / memory variants had never run inside a hybrid pattern).

### 10.1 Layout switch and routed-residual extension (2026-10-17)

**Layout (user)**: head dim 128 everywhere, Qwen3.5-proportioned, GQA. At hidden 768:
- attention: 6 q heads x 128, 2 KV groups;
- CKDA / triadic: 6 heads x 128 (CKDA derives heads = hidden / head dim, so its width stays 768);
- Qwen3.5-like GDN: 6 key / 12 value heads x 128. Public Qwen3.5-4B: hidden 2560, FFN 3.6x;
  attention 16 q x 256, 4 KV (GQA 4:1); GDN 16 key x 128 = 0.8x hidden, 32 value x 128 = 1.6x.
  The earlier `qwen35` (12 key heads = 2x hidden) was mixer-heavy, and matching cut its MLP
  to 1.1x hidden.
- vs the ComplexKDA ladder: head dim 64 there, MHA, dense FFN 4x hidden. The CKDA arms share
  the standard init, the low-rank gate, beta_max 2, 3:1 and gated NoPE attention. Its E / alpha
  are a rough reference only.

Parameter match (`param_match.json` keys `r3b:*`, non-emb vs attn_ref 84.96M):

| arm | d_ffn | non-emb vs attn_ref |
|---|---:|---:|
| attn_ref | 2048 | - |
| h3, h5, h3_tri, h3_nolxsa, h3_routed, h3_attnres, h3_ngpt, h3_routed_lerp | 1856 | -0.05 to +0.75 % |
| h3_mhc4, h3_routed_key, h3_routed_key_lerp | 1792 | -0.24 to +0.16 % |
| qwen35 | 1216 | +0.32 % |
| h3_engram | 1856 (= h3; tables not matched) | + table params |

**Routed-residual extension** (Megatron 931938bdf):
- `h3_routed_key`: the routed residual (2 of 4 streams) with the content-key router, which reads
  a 64-channel tag of every stream (`mhc_routed_key_dim 64`; gain 0 at init = the positional
  router).
- `h3_ngpt`: h3 in nGPT form (normalized weights / hidden states, LERP residuals;
  fused_qk_norm_rope off: nGPT's s_qk). The baseline for the LERP write, which requires nGPT.
- `h3_routed_lerp`: h3_ngpt + routed residual with the nGPT LERP write (each selected stream
  steps toward the normalized sublayer output, alpha scaled by the router's write weight).
- `h3_routed_key_lerp`: both.
Equal lr / bs across arms: the nGPT arms use the suite's lr, though nGPT's own AdamW optimum
was higher (4e-3).

The autoexp mirror files (base_defaults.yaml, config_schema.py, cli_metadata.py) were
regenerated for the new Megatron arguments.

Smokes: the old 12 x 64 layout ran all 10 base arms for 30 steps without an error (incl.
mHC / routed / AttnRes / Engram inside the hybrid pattern, and linear XSA in CKDA). The new
layout (`screen3_smoke3`) and the 4 extension arms (`screen3_smoke4`, new Triton kernels) are
running.

### 10.2 Angular MuOwn matrix-update scale (round-3 extension, done 2026-10-18)

Megatron's (angular) MuOwn has one `lr` for every parameter. The orthogonalized matrix update is
scaled by `angular_muown_extra_scale_factor` (default 0.2, Muon's 0.2·sqrt(max(m, n)): matches
AdamW's update RMS); embeddings / norms / gains get AdamW at the same lr. The lr sweep (optimum
4e-3) therefore moved both together. A larger step for the matrices alone (what others report
MuOwn tolerates) is a larger scale factor at fixed lr: `screen3_am_sf04` / `screen3_am_sf08`,
factor 0.4 / 0.8 at lr 4e-3 on attn_ref and h3 (4BT), next to the round-3 default 0.2.
(muown has `muown_extra_scale_factor`, default 1.0, the same knob.)

Result (paired with screen3_am's 4BT attn_ref / h3, same train_iters and seed: residual
correlation +1.00; 300-step window means of the training loss; all runs healthy):

| arm | factor | step 1k | 2k | 4k | 6k | 7.6k | vs 0.2 at 7.6k |
|---|---:|---:|---:|---:|---:|---:|---:|
| h3 | 0.2 | 3.4330 | 3.1993 | 3.0450 | 2.9794 | 2.9508 | |
| h3 | 0.4 | 3.5177 | 3.2801 | 3.1065 | 3.0247 | 2.9877 | +0.037 |
| h3 | 0.8 | 3.6915 | 3.4287 | 3.2356 | 3.1376 | 3.0885 | +0.138 |
| attn_ref | 0.2 | 3.4305 | 3.2186 | 3.0656 | 2.9987 | 2.9694 | |
| attn_ref | 0.8 | 3.7384 | 3.4894 | 3.2817 | 3.1761 | 3.1233 | +0.154 |

At lr 4e-3 a larger matrix step is worse, monotonically. The 0.4 gap narrows (+0.085 at 1k ->
+0.037 at 7.6k), but over 4BT it does not cross. Kept: factor 0.2. (sf04 attn_ref died at
startup with the CUDA-graph capture error and was not rerun: h3 decides it.) Open: 0.1, or
factor x lr trade-offs (0.4 at lr 2e-3), if the optimizer is revisited in round 4.

### 10.3 Round 3B: MoE design on the attention base (s1, ~300 GPU-h, user 2026-10-17)

Run in parallel with round 3, so that round 4 starts from a known-good MoE recipe rather than
sweeping it on the hybrids. Base: round 3's attn_ref (hidden 768 x 12 blocks, 6 q heads x 128,
GQA 2, qk-norm + XSA, RoPE) with an MoE FFN in every block (`*E` x 12). The active FFN width
is matched to the dense twin's d_ffn 2048 by construction (top-k x expert width (+ shared)
= 2048; the router adds 768 x E per layer). EP 4 on one node, the speed suite's MoE stack
(alltoall, grouped GEMM, permute fusion, fp32 router; CUDA graphs on attention only, because the
per-layer router statistics are not graph-safe), `moe_per_layer_logging` +
`moe_expert_viability_metrics`, every step logged. Stable only, 4000 steps at gbs 128 (2.1BT,
equal D for every arm, including the gbs arms), compared via the per-run fits (§9.6) and
`moe_health.py` (TensorBoard: load entropy / log E, max violation, dead and persistently
near-dead experts, router logit spread, router entropy).

Generated by `gen_screen.py` -> `screen3b_s1_mn5.yaml`; 16 designs, each with AdamW (2e-3,
1000 warmup) and angular MuOwn (4e-3, no warmup) = 32 runs:

| axis | arms | decision |
|---|---|---|
| dense twin | dense (d_ffn 2048) | MoE gain at equal active N and D |
| sparsity (x 512, top-4) | e16k4 / e32k4 / e64k4 (base) / e128k4 = 4 / 8 / 16 / 32x | total / active ratio for round 4 (gain vs memory / EP cost) |
| granularity (16x) | e32k2w1024 / e64k4 / e128k8w256 | expert width |
| shared expert | e64k3s512 (top-3 + 512 shared) | shared expert yes / no |
| router balancing | aux 1e-3 (base) / aux1e-2 / sigmoid (aux 1e-3) / bias (sigmoid, aux-loss-free, rate 1e-3) / bias_aux1e-4 | balancing method (loss and health); sigmoid separates the score function from the bias, which needs sigmoid |
| lr | lr_half / base / lr_x2 | does the MoE optimum move against dense (§8)? |
| gbs (equal tokens) | gbs64 (8000 steps) / base / gbs256 (2000 steps) | MoE batch-size sensitivity |

Each axis is decided per optimizer as well: if the optimizers disagree on an axis, round 4 keeps
both options for that axis. Smoke: 30 steps of the five riskiest arms (e128k4 memory,
e128k8w256, shared expert under MuOwn, bias under MuOwn, gbs 256).

### 10.4 Round 3 results, interim (2026-10-18; paired references running)

**Pairing problem.** Megatron's data order depends on train_iters and the seed. The round-3
short arms end through their config (aux.tokens, so train_iters 4000), and therefore saw
different batches from the 4BT validation arms attn_ref / h3 (train_iters 7629). Per-step
loss residuals (loss minus a 101-step running median) correlate at -0.02 between the groups and
+1.00 within a group. The seed-2 arms are paired only with each other. Unpaired gaps carry
~0.01-0.02 nats of data-order offset at 4k steps (seed2 pair vs main pair: -0.012 vs
-0.020). Screen 2 was not affected: all its arms had the full train_iters and a watcher stopped
the short ones.

Fix: attn_ref_4k and h3_4k, the short arms' exact schedule (4000 steps, seed 1234), for both
optimizers (~12 GPU-h, running). `analyze.run_logs` now reads each log's arm from the log itself:
re-rendering a suite for a one-arm rerun renumbers task_*.sh, and the old mapping mislabelled
screen3_am's attn_ref. Future rounds: every compared arm shares train_iters, gbs and seed with
its reference.

**Paired among the short arms** (reference qwen35, which is short; per-run fits to 2.1BT ->
4BT, bootstrap 5-95 %; raw gap at steps 2k / 4k; throughput from the 30-step smokes, tok/s/GPU):

| arm | AdamW: fit 4BT | AdamW: raw 2k / 4k | angular MuOwn: fit 4BT | angular MuOwn: raw 2k / 4k | tok/s/GPU |
|---|---:|---:|---:|---:|---:|
| h3_engram (tables not matched) | -0.055 [-0.058, -0.053] | -0.027 / -0.045 | -0.058 [-0.063, -0.053] | -0.069 / -0.075 | 118k |
| h3_mhc4 | -0.051 [-0.053, -0.049] | -0.010 / -0.035 | -0.051 [-0.053, -0.048] | -0.065 / -0.064 | 102k |
| h3_attnres | -0.046 [-0.048, -0.045] | -0.041 / -0.044 | +0.036 [+0.028, +0.045] | +0.028 / +0.003 | 115-118k |
| h3_routed | -0.040 [-0.042, -0.038] | -0.033 / -0.037 | -0.002 [-0.004, -0.000] | -0.025 / -0.017 | 114k |
| h3_routed_key | -0.037 [-0.038, -0.036] | -0.042 / -0.041 | +0.001 [-0.002, +0.004] | -0.027 / -0.020 | 105k |
| h3_nolxsa | -0.011 [-0.014, -0.008] | +0.001 / -0.008 | -0.011 [-0.015, -0.007] | -0.026 / -0.027 | 133k |
| h3_tri | -0.008 [-0.012, -0.004] | +0.019 / +0.004 | -0.019 [-0.021, -0.016] | -0.025 / -0.028 | 119k |
| qwen35 | 0 | 0 | 0 | 0 | 128k |
| h5 | +0.019 [+0.016, +0.023] | +0.029 / +0.021 | -0.003 [-0.007, +0.002] | -0.019 / -0.022 | 131k |
| h3_routed_key_lerp | +0.168 | | +0.260 | | 92k |
| h3_ngpt | +0.188 | | +0.235 | | 126k |
| h3_routed_lerp | +0.208 | | +0.252 | | 101k |

(h3 itself: 133k tok/s/GPU; attn_ref 170k.) Reading, before the paired h3 / attn_ref arrive:
- mHC-4 and Engram are the largest gains, under both optimizers and with healthy curves. Under
  AdamW the gap is still growing at 4k steps; under angular MuOwn it is flat (mHC) or growing
  (Engram). mHC costs ~23 % throughput at s1 and Engram ~11 %. Engram's tables are not
  parameter-matched (sparse lookups, like embeddings).
- Under AdamW, the routed residual (both routers) gains 0.037-0.040 nats and is about as good
  as mHC-4 at 4k steps. Under angular MuOwn the gain fades (-0.025 -> -0.017 raw, ~0 fitted).
  The content-key router is no better than the positional one.
- Block AttnRes splits by optimizer: -0.046 under AdamW, worse than qwen35 under angular MuOwn
  (shrinking from +0.046 to +0.003). Throughput -10 % vs h3 (corrected 2026-10-18: the earlier
  "6x slower" was the 30-step smoke, dominated by the one-time torch.compile of the aggregation).
- The nGPT family (h3_ngpt, both LERP routed variants) is 0.17-0.26 nats worse at the shared lr.
  nGPT needs its own lr / no-warmup setting (hp_s1 nGPT arms), so this tests it out of its
  regime, not its potential. The LERP write is only defined on nGPT, so it cannot be judged
  separately here.
- h3 vs h5 vs h3_tri vs h3_nolxsa: their ranking flips between optimizers (AdamW: nolxsa
  < tri < qwen35 < h5; angular MuOwn: tri ≈ nolxsa < h5 ≈ qwen35). The gaps are 0.01-0.03, at
  the MuOwn resolution limit (§9.11). Decided once h3_4k gives the paired h3.

### 10.5 Round 3 final (paired) and round 3B results (2026-10-18)

**Round 3, paired against attn_ref_4k** (same schedule, data order and seed as every short
arm; gaps in nats; AdamW: per-run fit 2.1BT -> 4BT [5-95 %] / raw gap at step 4000; angular
MuOwn: raw gap at 2k / 4k steps, because under MuOwn the fits are not decision-grade (§9.11) and
mis-predicted h3 here: fit -0.003, while the 4BT pair holds -0.019 to 7.6k):

| arm | AdamW: fit 4BT | AdamW: raw 4k | angular MuOwn: raw 2k / 4k | tok/s/GPU |
|---|---:|---:|---:|---:|
| attn_ref (RoPE, qk-norm, XSA) | 0 | 0 | 0 / 0 | 170k |
| h3 (CKDA 3:1, NoPE + gate + XSA attention, linear XSA) | -0.010 [-0.014, -0.006] | -0.006 | -0.022 / -0.022 | 133k |
| h5 (5:1) | -0.003 [-0.006, +0.001] | -0.001 | -0.016 / -0.017 | 131k |
| h3_tri (triadic CKDA) | -0.031 [-0.034, -0.027] | -0.017 | -0.021 / -0.023 | 119k |
| h3_nolxsa (no XSA in CKDA) | -0.033 [-0.036, -0.030] | -0.030 | -0.022 / -0.022 | 133k |
| qwen35 (GDN, Qwen3.5-like) | -0.022 [-0.023, -0.021] | -0.022 | +0.004 / +0.005 | 128k |
| h3_mhc4 | -0.073 [-0.075, -0.071] | -0.057 | -0.062 / -0.059 | 102k |
| h3_engram (tables not matched) | -0.078 [-0.080, -0.075] | -0.067 | -0.065 / -0.070 | 118k |
| h3_routed | -0.062 [-0.064, -0.060] | -0.059 | -0.022 / -0.012 | 114k |
| h3_routed_key | -0.059 [-0.061, -0.058] | -0.062 | -0.023 / -0.015 | 105k |
| h3_attnres | -0.068 [-0.070, -0.067] | -0.065 | +0.032 / +0.008 | 115-118k |
| nGPT family | +0.15 to +0.19 | | +0.24 to +0.27 | 92-126k |

Gap noise across data orders (h3 vs attn_ref at step 4000 in three differently ordered pairs:
-0.006 / -0.008 / -0.014 under AdamW): about ±0.004, on top of the fit error.

Decisions (s1, dense; to be confirmed at s2 / on the MoE base in round 4):
- **Hybrid, CKDA 3:1: kept, but the gain is smaller in this layout.** With head dim 128 / GQA
  attention, h3 beats attention by 0.006-0.020 under AdamW (gap growing: -0.014 at 4k -> -0.020
  at 7.6k on the 4BT pair) and by a stable ~0.02 under angular MuOwn. Screen 2 (head dim 64)
  had -0.057 / -0.021. The hybrid's main argument is now inference efficiency, with a small
  loss gain.
- **XSA in the CKDA layers: dropped.** It costs 0.023 under AdamW (h3_nolxsa better) and is
  neutral under angular MuOwn.
- **Triadic CKDA: optional.** It gains 0.021 over h3 under AdamW (fit), but under angular MuOwn
  it equals h3, at -10 % throughput. Not worth it for the MuOwn recipe.
- **5:1 vs 3:1: 5:1 is acceptable.** It costs 0.005-0.007 under both optimizers, for half the
  attention layers (KV cache / long-context cost). Choose by the inference budget.
- **Qwen3.5-like GDN: not better than CKDA.** It wins by 0.012 under AdamW but loses by 0.027
  under angular MuOwn, and its lead shrank with D in screen 2.
- **mHC-4: the clearest residual win** (-0.06 under both optimizers, stable or growing). It
  costs 23 % throughput at s1 (smaller relative cost at 3B-active, where mixing is a smaller
  share). Round-4 candidate.
- **Engram: largest gain, but not parameter-matched** (sparse tables add parameters, like
  embeddings). Keep it as a candidate only with its parameter / memory cost reported. A
  matched comparison against a larger vocabulary or more experts is the fair test.
- **Routed residual: AdamW only** (-0.06; under angular MuOwn it fades to -0.012 by 4k). The
  content-key router is no better than the positional one.
- **Block AttnRes: on hold** (not dropped: the 6x slowdown was a smoke artefact; real runs are
  -10 % vs h3, the aggregation is already Inductor-compiled). It splits by optimizer; round 3C
  tests whether warmup / the mixer lr fixes it under angular MuOwn.
- **nGPT + LERP: not judged.** The shared lr is outside nGPT's regime (§8). A fair test needs
  nGPT's own lr sweep. Low priority given the residual results above.

**Round 3B (MoE on attn_ref; gaps vs the AdamW dense twin, fit 2.1BT -> 4BT, paired; health
from TensorBoard, tail 20 %: load entropy normalised to 1 = uniform, max_vio = worst layer's
max/mean - 1):**

| arm | AdamW: vs dense | angular MuOwn: vs AdamW dense (vs MuOwn dense) | max_vio AdamW / MuOwn | tok/s/GPU |
|---|---:|---:|---:|---:|
| dense | 0 | -0.070 (0) | | 165k |
| e16k4 (4x) | -0.080 | -0.155 (-0.085) | 1.5 / 0.9 | 100k |
| e32k4 (8x) | -0.115 | -0.186 (-0.117) | 3.5 / 1.3 | 100k |
| e64k4 (16x, base) | -0.140 | -0.214 (-0.144) | 6.9 / 2.1 | 97k |
| e128k4 (32x) | -0.157 | -0.226 (-0.156) | 16.3 / 3.9 | 93k |
| e32k2w1024 (16x, coarse) | -0.121 | -0.201 (-0.131) | 7.6 / 1.8 | 112k |
| e128k8w256 (16x, fine) | -0.152 | -0.214 (-0.144) | 8.6 / 2.0 | 76k |
| e64k3s512 (shared expert) | -0.150 | -0.223 (-0.154) | 2.8 / 1.5 | 102k |
| aux1e-2 | -0.130 | -0.209 (-0.139) | 3.4 / 0.5 | 100k |
| sigmoid (aux 1e-3) | -0.141 | -0.225 (-0.156) | 6.7 / 1.0 | 97k |
| bias (aux-free) | -0.138 | -0.226 (-0.156) | 2.5 / 3.6 | 101k |
| bias_aux1e-4 | -0.144 | -0.229 (-0.159) | 0.6 / 1.9 | 99k |
| lr x 0.5 | -0.163 | -0.207 (-0.138) | 11.8 / 1.3 | 96k |
| lr x 2 | -0.042 | -0.122 (-0.052) | 4.5 / 3.1 | 97k |
| gbs 64 (equal D) | -0.049 | -0.170 (-0.100) | 6.8 / 3.2 | 94k |
| gbs 256 (equal D) | -0.165 | -0.218 (-0.148) | 7.9 / 2.2 | 100k |

(gbs arms are unpaired: their data order differs. All 32 runs healthy: no spikes, no loss
growth; load entropy ≥ 0.95 everywhere; dead experts ≤ 0.65 expert-layer slots, only in the
bias arms.)

Decisions:
- **MoE gain at equal active N and D is large: -0.14 nats at 16x** (64 x 512, top-4), under
  both optimizers. It grows with sparsity at a diminishing rate (4x / 8x / 16x / 32x: -0.08 /
  -0.12 / -0.14 / -0.16; ~0.02 per doubling at the top). 32x costs 2x the total memory for
  0.016, so **16-32x** goes into round 4, with 32x where memory allows. For the 3B-active
  target this matches the planned 30-100B totals (10-32x).
- **Granularity: finer is better.** 128 x 256 top-8 equals or beats 64 x 512 top-4 (-0.152 vs
  -0.140 AdamW, equal under MuOwn), but it is 21 % slower at s1. 32 x 1024 top-2 is 0.01-0.02
  worse. Keep top-4 x 512-class experts; recheck the fine version's throughput at 3B scale
  (the speed suites).
- **Shared expert: yes** (k3 + 512 shared beats k4 by 0.010 under both optimizers, at no speed
  cost). Router balance is also better.
- **Router: sigmoid scores, expert bias + small aux loss (1e-4).** It is the best on loss under
  both optimizers (-0.144 / -0.159 vs softmax + aux 1e-3 -0.140 / -0.144) and on balance (AdamW
  max_vio 0.6 vs 6.9). Aux 1e-2 balances well but costs 0.01. Under angular MuOwn, sigmoid
  alone already gives +0.012 over softmax.
- **lr / gbs: the MoE optimum moves under AdamW only.** AdamW: lr x 0.5 is better by 0.023
  (1e-3; the dense optimum was 2e-3), gbs 256 better than 128 at equal D. The MoE wants a
  lower noise scale. Angular MuOwn: base lr 4e-3 stays best (x0.5: +0.006, x2: +0.09), gbs
  128-256 flat, 64 worse. Angular MuOwn is again better than AdamW by ~0.07-0.08 on MoE,
  without retuning.
- **Optimizer for round 4: angular MuOwn** (best loss, best balance at the default lr, no
  warmup, lr transferred from dense). AdamW stays only as a cross-check at lr 1e-3.

### 10.6 Round 3C proposal: residual variants under angular MuOwn, routed-8, nGPT lr, N trend (~200 GPU-h, for review)

Questions after round 3:
1. Is the AdamW / angular MuOwn split of the residual variants real, or an untuned optimizer?
   Under angular MuOwn the residual-mixer parameters (mHC phi / b / alpha, routed router,
   AttnRes zero-init query + key norm) are on Adam at the shared lr 4e-3 with no warmup. That
   is 2x the dense AdamW optimum and 4x the MoE AdamW optimum. The AdamW runs had a 1000-step
   warmup. AttnRes under MuOwn starts +0.34 behind h3 at step 300 and recovers to +0.02 by
   4k; under AdamW it is -0.06. This points to optimization, not architecture.
2. "Fading" routed residual: the early lead (-0.14 vs h3 at step 300 under both optimizers)
   decays. Under AdamW it settles at -0.06 (-0.052 at 4k, slowly shrinking); under MuOwn it is
   level at step 2000 and +0.011 behind at 4k, still drifting. N-trend data: none yet (s1 only,
   D ≤ 2.1BT).

Arms (s1, 4000 steps at train_iters 4000 / seed 1234 / gbs 128, paired with the round-3 suites,
so no new references; angular MuOwn unless noted):
- Warmup for the residual variants: h3, h3_mhc4, h3_routed, h3_attnres with a 1000-step
  warmup (4 runs). No code change.
- Mixer lr: Adam lr x 0.25 (= 1e-3) for the residual-mixer parameters only, on h3_mhc4,
  h3_routed, h3_attnres (3 runs). Needs a small Megatron option (`residual_mixer_lr_mult`,
  on the params tagged use_adam_with_emerging_optimizers in mhc / routed_residual / attn_res,
  like engram_lr_mult). Local, unpushed.
- Routed 8 streams, 2 active: h3_routed8 (mhc_expansion_rate 8, topk 2) under both
  optimizers (2 runs), plus h3_mhc8 under MuOwn as its dense-mixing counterpart (1 run).
- nGPT lr sweep (AdamW, its own regime: no warmup), h3_ngpt at lr 4e-3 / 8e-3 / 1.6e-2. nGPT's
  curve is more linear and crosses the others twice (better early, worse mid, better after a
  cooldown), so stable short runs cannot judge it. These run to 4BT with a 20 % WSD cooldown,
  next to h3 with the same WSD under AdamW 2e-3 and angular MuOwn 4e-3 (5 runs). This compares
  final cooled losses.
- N trend at s2 (hidden 1024 x 20 blocks), new suite with the exit_interval protocol
  (train_iters = 4BT for every arm, exit_interval 4000: same data order for short and full
  arms, winners extend by resuming): attn_ref, h5 (no linear XSA, the new default), h5_mhc4,
  h5_routed (4 runs, angular MuOwn), with the winner of the warmup / mixer-lr arms applied.

- Mixed / pure-linear stacks (user 2026-10-18), angular MuOwn, no linear XSA (the new default),
  12 blocks = 2 periods of 6:
  - h5_nolxsa: attn 1 + CKDA 5, the reference (round-3 h5 had linear XSA, which is neutral
    under MuOwn but not exactly the same arm);
  - h5_tri1: attn 1 + triadic CKDA 1 + CKDA 4;
  - lin_ckda: CKDA only (no attention);
  - lin_tri1_e4 / lin_tri1_e8: triadic CKDA 1 (E = 4 / 8) + CKDA 5, no attention.
  d_ffn matched per arm (count_params). Expectation: swapping all linear layers to triadic
  (h3_tri) gained 0.011-0.021 under AdamW but only ~0.003 under MuOwn, at -10 % throughput,
  so one triadic layer in six inside a hybrid should gain less than the ±0.004 data-order noise.
  The pure-linear arms answer the more useful question: how much attention is worth at this
  scale, and whether a triadic layer (more state per head) recovers part of it. Caveat: at 4k
  context and 2BT the training loss underrates attention (recall / long-context ability is
  not measured). A pure-linear "win" here needs a recall eval before it counts.
  Needs a Megatron change (local, unpushed): the `L` symbol builds the one global
  experimental_attention_variant, so a second linear symbol (`T` = triadic with
  linear_triadic_base / _e) is required to mix triadic and plain CKDA layers in one stack.
  5 runs, ~25-30 GPU-h.

Analysis: fits from step 1500 (§9.12), paired gaps vs h3_4k / attn_ref_4k; at s2 vs the suite's
own attn_ref.

Cost (s1 short ≈ 4.5 GPU-h, mHC/routed ≈ 6, AttnRes ≈ 5, s1 4BT ≈ 9, s2 short
≈ 12): warmup 4 + mixer-lr 3 (2 AttnRes) ≈ 80; routed-8 / mHC-8 ≈ 20; nGPT + references ≈ 45;
s2 ≈ 50; mixed / pure-linear ≈ 30. Total **≈ 230 GPU-h** (2,101 used of 3,000).

### 10.7 Round 3C results (2026-10-18)

Paired with round 3 (same train_iters 4000, seed, gbs: residual correlation 0.999); raw gaps,
200-step window means; angular MuOwn unless noted.

**Residual variants under angular MuOwn: the split was the mixer learning rate.**

| arm | gap at 2k / 4k steps | reference |
|---|---:|---|
| h3_mhc4 (round 3: no warmup, mixer lr 4e-3) | -0.039 / -0.036 | h3 |
| h3_mhc4, mixer lr x 0.25 | **-0.048 / -0.048** | h3 |
| h3_mhc4, 1000-step warmup | -0.054 / -0.049 | h3 with the same warmup |
| h3_routed (round 3) | +0.001 / +0.011 | h3 |
| h3_routed, mixer lr x 0.25 | **-0.027 / -0.022** | h3 |
| h3_routed, 1000-step warmup | -0.035 / -0.030 | h3 with the same warmup |
| h3_attnres (round 3) | +0.053 / +0.030 | h3 |
| h3_attnres, mixer lr x 0.25 | **-0.006 / -0.011** | h3 |
| h3_attnres, 1000-step warmup | +0.003 / -0.013 | h3 with the same warmup |
| h3_routed8 (8 streams, 2 active; mixer lr x 1) | +0.021 / +0.028 | h3 |
| h3_mhc8 (mixer lr x 1) | diverged (grad norm NaN from the start) | |
| AdamW: h3_routed8 / h3_routed (4) / h3_mhc4 | -0.043 / -0.058 / -0.037 at 2k; -0.034 / -0.052 / -0.051 at 4k | AdamW h3 |

(The warmup itself costs h3 +0.068 / +0.028 at 2k / 4k, shrinking.)
- Under angular MuOwn the residual mixers (Adam at the shared lr 4e-3, no warmup) were trained
  at ~4x Adam's optimum. Either a 1000-step warmup or the mixer lr x 0.25 (residual_mixer_lr_mult)
  recovers the gains, and every residual variant now beats h3 under both optimizers. The
  round-3 "AdamW only" verdicts for the routed residual and AttnRes were an optimizer artefact.
  This answers the question of why labs train Muon + mHC / AttnRes.
- Recipe for angular MuOwn: **residual_mixer_lr_mult 0.25** (no warmup). Ranking: mHC-4
  (-0.048) > routed-4 (-0.022) ≈ AttnRes (-0.011, still improving) at 4k steps.
- 8 streams: routed-8 and mHC-8 ran at the mixer lr x 1 (routed-8 worse than h3, mHC-8
  diverged). The x 0.25 rerun is the missing arm. Under AdamW, routed-8 (-0.034) is worse than
  routed-4 (-0.052) at s1.

**Mixed / pure-linear stacks** (gap to attn_ref at 2k / 4k steps; no XSA in the CKDA layers):

| arm | gap |
|---|---:|
| h3 (round 3, with linear XSA) | -0.022 / -0.022 |
| h5 (round 3, with linear XSA) / h5_nolxsa | -0.016 / -0.017; -0.012 / -0.013 |
| h5_tri1: attn 1 + triadic CKDA 1 + CKDA 4 | -0.015 / -0.014 |
| lin_ckda: CKDA 6, no attention | +0.047 / +0.033 |
| lin_tri1_e4 / lin_tri1_e8: triadic 1 (E = 4 / 8) + CKDA 5 | +0.032 / +0.024; +0.025 / +0.017 |

- Attention is worth ~0.045 nats over pure CKDA at s1 (and more on recall, not measured). One
  triadic layer in six recovers 0.009 (E = 4) to 0.017 (E = 8), about a third, but the pure-linear
  stack stays behind attention.
- Inside the hybrid, a triadic layer adds nothing measurable (-0.001). The gap between
  h5_nolxsa and the round-3 h5 is within the ±0.004 data-order noise.

**s2 (hidden 1024 x 20 blocks; exit_interval 4000 of 4BT; angular MuOwn; mixer lr x 1, i.e.
before the fix above)**, gap to attn_ref at 1k / 2k / 3k / 4k steps:

| arm | gap |
|---|---:|
| h3 (CKDA 3:1, no linear XSA) | -0.002 / -0.010 / -0.012 / -0.010 |
| h3_mhc4 | -0.081 / -0.080 / -0.071 / -0.066 |
| h3_routed (2 of 4) | -0.032 / -0.026 / -0.016 / -0.009 (fading: the mixer-lr pattern) |

The hybrid's own gain shrinks from s1 to s2 (-0.022 -> -0.010). mHC-4 holds at -0.066. The s2
routed arm needs the x 0.25 rerun (round 4). exit_interval worked with the monitor: all four
stopped at 4000 and were marked finished.

**64x MoE** (round 3B suite, angular MuOwn, raw gap to the dense twin at 4k steps, mbs 4):
e16k4 -0.077, e32k4 -0.104, e64k4 -0.125, e128k4 -0.135, **e256k4 -0.143**, e128k8w256 -0.128,
**e512k8w256 -0.148**. So 32x -> 64x gains another 0.008. Fine granularity at 64x gains
0.005 more, but trains at 84 TFLOP/s (43k tok/s) against 123 (63k): expert width 256 and 8-way
dispatch. For the 80B-A3B class at this granularity: 256 x 512 top-4.

**nGPT (AdamW, no warmup) to 4BT with a 20 % cooldown** (training loss, last 100 steps):

| arm | stable end (6k) | cooled (7629) | validation |
|---|---:|---:|---:|
| h3, AdamW 2e-3 | 3.042 | 2.881 | 2.875 |
| h3, angular MuOwn 4e-3 | 2.978 | 2.832 | 2.826 |
| h3 nGPT, lr 4e-3 | 3.373 | 3.007 | 2.994 |
| h3 nGPT, lr 8e-3 / 1.6e-2 | diverged | 7.9 / 11.6 | |

nGPT has the large cooldown drop expected for it (-0.37 vs AdamW's -0.16), but it ends 0.13
behind AdamW and 0.18 behind angular MuOwn, and diverges at 2x its lr. **Dropped.**

## 11. Round 4: the MoE hybrid with a residual stream + LatentMoE (2026-10-18)

Generated by `gen_screen.py` -> `screen4_s1_mn5.yaml` / `screen4_s2_mn5.yaml`.

**Base.** The round-3B MoE recipe at 64x: 256 routed experts, top-3 + 1 shared expert of the
expert width (the dense twin's active FFN = 4 x width), sigmoid router + expert bias (rate 1e-3)
+ aux 1e-4, mbs 4. CKDA 5:1 (reference CKDA, no linear XSA); attention NoPE + qk-norm + XSA +
output gate (the all-attention reference: RoPE + qk-norm + XSA). Angular MuOwn without warmup
(s1 lr 4e-3, s2 2.5e-3 by the (N2/N1)^-0.417 law), matrix scale 0.2, every residual mixer at
residual_mixer_lr_mult 0.25 (§10.7).

**Protocol.** train_iters = 4BT for every arm, exit_interval 4000 (2.1BT): equal data order for
short and full runs, and the finalists extend to 4BT by resuming. Fits from step 1500 (§9.12),
paired gaps vs each suite's attn_moe and h5_moe.

**Sparsity: 48x** (144 routed experts, top-3 + 1 shared; the 80-90B-A3B class: total at 3B
active ≈ 2.1B + (E/k) x 1.81B; 64x OOMed at s2 on one node). mbs 2 for every arm; per arm the
cheapest of + selective recompute (moe_act, layernorm) / + full recompute that fits (smoke).

**Sizes.** s1 = hidden 768 x 12 blocks (6 q heads x 128); s2 = hidden 1024 x **18** blocks (8 q
heads x 128). 5:1 needs blocks % 6 == 0, so s2 is not round 3's 20-block s2 and has its own
references.

| arm | s1 | s2 | decides |
|---|:---:|:---:|---|
| attn_moe | x | x | reference |
| h5_moe (CKDA 5:1) | x | x | does the hybrid gain survive MoE / size? |
| + routed 2 of 4 | x | x | routed baseline |
| + routed 2 of 8, reversible | x | x | more streams at the same active cost |
| + mHC-4 | x | x | the loss-best residual (-22 % speed at A3B) |
| + mHC-2 | x | x | the routing control: as many streams per sublayer as routed 2-of-n; if it matches routed, the routing adds nothing |
| routed 3 of 8 | x | | each stream about as active as in 2 of 4 (2 of 8: 2-of-8 lost to 2-of-4 at s1 under AdamW, -0.034 vs -0.052) |
| routed 2 of 8 at mixer lr x 0.125 / x 0.5 | x | | the adapted mixer lr |
| routed 2 of 8, content-key router (64) | x | | |
| LatentMoE lat2: latent 384, 512 x 512, top-6 + shared | x | | equal active compute, total and dispatch volume: finer routing in latent space |
| LatentMoE lat2_half: latent 384, 256 x 512, top-3 + shared | x | | half the routed cost at (hopefully) similar loss |

LatentMoE (`moe_latent_size`) adds the two latent projections (2 x hidden x latent per layer:
~+8 % active parameters at s1, reported, not matched away). At hidden 768 the latent GEMMs are
tiny, so its s1 throughput is not representative: loss is judged at s1, speed at A3B
(`speed_a3b30_mn5.yaml`: r3_ckda5_lat2 / r3_ckda5_lat2_half, latent 1024).

**Speed (A3B-30B / 80B):** routed-8 with sublayer recompute + fused router + reversible (pending:
the interleaved-pipeline hang, §7.2) against mHC-4's -22 %; LatentMoE equal / half cost.

**Preference (user, 2026-10-18): the routed residual** (with the adapted mixer lr) for the final
model. It is new, decodes faster (a decode step touches 2k + 1 of the n streams), and has neither
mHC's memory overhead (the n-wide mixing) nor AttnRes's (keeping many block states). Round 4
picks its stream count / mixer lr / router. mHC-4 is the loss reference it is measured against
(compute-matched at s1, §10.7: routed-4 net -0.005, mHC-4 -0.013, AttnRes +0.004 under
angular MuOwn). Routed now also runs under per-layer ("local") CUDA graphs: replay matches eager
for plain, content-key and reversible routing (test_depth_residual_cuda_graphs.py; reversible:
the router gradients within rounding, since the graphed run reads the untouched input instead
of the undone one). The s1 throughput of the routed arms so far was without graphs, against h3
with them.

**Decision rule.** Routed-8 if it is within ~0.01 of mHC-4's loss at a clearly lower speed
cost, else mHC-4. LatentMoE if lat2 beats h5_moe (or lat2_half matches it).

**Cost.** s1 ~10 GPU-h per arm (10 arms), s2 ~20 (5 arms), extension of 2 s2 finalists to 4BT
(~35), speed (~40): ~275 GPU-h (budget: ~2,450 of 3,000 used).

### 11.1 Round 4B: speed and scale feasibility of the best bet, bf16 vs fp8 (2026-10-18)

The best bet so far: CKDA 5:1 (reference CKDA, no linear XSA) + attention NoPE / qk-norm / XSA
/ gate, routed residual 2 of 8 (reversible, sublayer recompute, fused router), angular MuOwn,
MoE with a shared expert and the sigmoid + expert-bias router. Before a production run, its
layout, memory and speed go through exp_irene's MN5 setup (TP1, EP4 HybridEP, full recompute,
gbs 1024) at the target sizes. fp8 uses the 32B-dense production recipe
(`config/backend/megatron/oellm_32b_dense/precision/fp8.yaml`: hybrid, tensorwise, first / last
layer bf16, no fp8 param gather). The 32B dense run's other speed settings (deferred embedding
wgrad, overlapped grad reduce / param gather, native fused CE) are already in the speed base
(fused CE via Liger, gradient-accumulation fusion, overlap).

`speed_r4b_mn5.yaml` (25 steps, mbs 2, no VPP: routed + VPP hangs at A3B, §7.2):

| family | experts | sparsity | total / active | layout |
|---|---|---:|---|---|
| m48a3 | 336 x 768, top-7 + shared | 48x | ~79B / 3B | PP8 x EP4, 16 nodes |
| m32a3 | 224 x 768, top-7 + shared | 32x | ~55B / 3B | PP8 x EP4, 16 nodes |
| m48a2 | 192 x 512, top-4 + shared | 48x | ~31B / ~2.9B (1.8B non-emb) | PP4 x EP4, 8 nodes |

each in bf16 and fp8. Estimate <= 98 GPU-h (budget: 2,499 used + round 4 ~300 + this ~100 ->
~2,900 of 3,000). The production training config (data: the flag mix, 16 dataloader workers +
prefetch 8 against the GPFS stall, checkpointing, wandb) follows from the feasible family.

**Results (25 steps, mbs 2, no VPP, full recompute; tok/s/GPU, memory):**

| family | bf16 | fp8 | fp8 vs bf16 |
|---|---:|---:|---:|
| m48a3 (~79B-A3B, 16 nodes) | 3,431 (68 TFLOP/s, 89 %) | 3,190 (63, 90 %) | -7 % |
| m32a3 (~55B-A3B, 16 nodes) | 3,731 (74, 75 %) | 3,559 (71, 76 %) | -5 % |
| m48a2 (~31B-A2.9B, 8 nodes) | 5,359 (72, 82 %) | 4,940 (67, 84 %) | -8 % |

- All three fit on MN5's 64 GB GPUs. 48x at 3B active is tight (89 %): PP8 x EP4 on 16 nodes.
- **fp8 is slower here, and uses slightly more memory.** At hidden 2048 with 512-768-wide
  experts the GEMMs are small, so the tensorwise quantize / dequantize work outweighs the faster
  matmuls, and the fp8 weight copies cost memory. The 32B-dense gain came from much larger GEMMs.
  Keep bf16 for this model class (an MXFP8 / blockwise recipe on Blackwell would be a different
  question).
- Against the round-3/3B recipe without a residual stream (A3B-80B: 3,880 with VPP), m48a3 with
  routed-8 and without VPP is at -12 %. VPP for routed is blocked by memory (the NCCL OOM above;
  the mbs-1 test is pending).
- 10T tokens at these rates: m48a3 ~810k GPU-h (66 days on 512 GPUs); m32a3 ~745k; m48a2 ~520k
  GPU-h (42 days on 512 GPUs).

**Why fp8 is slower: microbenchmark on one H100** (`scripts/korbi/hybrid_speed/fp8_bench.py`,
TE 2.19, 8192 tokens, fwd + bwd; x = speed vs bf16, > 1 is faster):

| shape | bf16 | fp8 tensorwise | tensorwise, weights cached | delayed, cached |
|---|---:|---:|---:|---:|
| attention q + kv proj 2048 -> 2560 | 0.49 ms | 0.96x | 1.02x | 1.10x |
| CKDA-style proj 2048 -> 6144 | 0.96 ms | 0.96x | 1.02x | 1.13x |
| shared expert fc1 2048 -> 1536 | 0.31 ms | 0.59x | 0.63x | 0.68x |
| shared expert fc2 768 -> 2048 | 0.22 ms | 0.36x | 0.44x | 0.46x |
| MoE fc1, 84 experts x 2048 -> 1536, 176 tok/expert | 3.84 ms | 0.35x | 0.45x | 0.58x |
| MoE fc2, 84 experts x 768 -> 2048, 176 tok/expert | 4.52 ms | 0.33x | 0.41x | 0.58x |
| MoE fc1, 688 tok/expert | 5.01 ms | 0.37x | 0.45x | 0.65x |
| 32B-dense-like 5120 -> 13824 | 5.26 ms | 1.19x | 1.29x | 1.36x |

1. **The fp8 GEMMs are faster, but not by much at these sizes:** about 1.8x for the dense
   projections, 1.4x for the grouped expert GEMMs.
2. **Quantization overhead outweighs the gain.** Tensorwise (current) scaling adds an amax pass
   over every input and a cast-transpose to every GEMM. Dense 2048 -> 2560: 0.22 ms of GEMM vs
   0.27 ms of amax / cast. In the grouped expert GEMM the cast-transposes are per-expert
   `cast_transpose_general` kernels (~0.85 ms each), about 5 ms of quantization against 1.6 ms
   of GEMM.
3. **CPU overhead.** Small fp8 calls are host-bound: the shared expert's fc2 needs 0.23 ms of
   kernels but takes 0.61 ms of wall time, and the MoE fc1 ~4 ms over its kernels (84 per-expert
   quantizers per call).
4. **Coverage.** The CKDA mixers use flash-linear-attention's plain nn.Linear, so 40 of 48
   mixer layers stay bf16 under fp8. fp8 reaches only the 8 attention layers, the experts and the
   shared expert, and it slows the experts down. The router (fp32) and the routed-residual
   kernels stay bf16.
5. **Memory.** The fp8 weight copies (and their transposes) add ~1-2 % of memory.
6. The 32B-dense GEMMs are large enough for fp8 to win (1.2-1.4x), hence that run's gain.

**Verdict: bf16 for this model class on H100.** Making fp8 pay would need a fused quantization
path (quantize once in the token dispatch, e.g. HybridEP / DeepEP fp8 dispatch, instead of per
expert GEMM), cached weight casts, router-level padding (`moe_router_padding_for_quantization`,
off in our runs, so the experts also padded and unpadded per call) and fp8 projections for the
CKDA mixers. Even with zero overhead the ceiling is modest: GEMMs are 60-85 % of these layers'
bf16 time and fp8 speeds them up 1.4-1.8x. MXFP8 with fused block quantization (Blackwell) is
the setting where fp8 would be worth re-testing.

### 11.2 A3B slice: CUDA graphs, fused kernels, fp8 host overhead (2026-10-18)

One node, one 5:1 period at m48a2's layer shapes (192 x 512, top-4 + shared, EP4, mbs 2,
routed 2 of 8, no recompute; `speed_a3bslice_mn5.yaml`, 20 steps, ms per step):

| arm | ms / step | mem |
|---|---:|---:|
| bf16 | 1492 | 82 % |
| bf16 + per-layer graphs (attn scope) | 1516 | 90 % |
| bf16 + TE projections in CKDA (`linear_te_projections`) | 1522 | 82 % |
| bf16 + TE projections + graphs | 1490 | 89 % |
| fp8 tensorwise (+ router padding) | 1842 | 84 % |
| fp8 + graphs | 1745 | 89 % |
| fp8 + TE projections | 1910 | 85 % |
| fp8 delayed + graphs | 1622 | 90 % |
| fp8 delayed + TE projections + graphs | 1680 | 92 % |
| full-iteration graph | capture fails (host copy during capture) | |

- **The fp8 CPU overhead is Transformer Engine's per-call Python and quantizer work.** Every fp8
  GEMM needs amax / scale / cast(-transpose) launches and per-expert quantizer objects (84 per
  grouped GEMM), so the step issued 3.9x the launches of bf16 (94k vs 24k) and the GPU was busy
  70 % of the time (bf16: 88 %). CUDA graphs remove the launch cost (with graphs the GPU is ~93 % busy; the extra
  static-input copies eat the gain in bf16) but not the quantization kernels themselves, which outweigh the GEMM savings (265 ms vs 193 ms).
- **TE projections for the CKDA mixers** (new, `linear_te_projections`; same parameters, tested
  exact): they fuse the weight-gradient accumulation (the separate `AccumulateGrad` adds were
  ~5 % of the bf16 step) and make the projections fp8-eligible. Measured: neutral in bf16
  (within the 2 % noise of 20 steps), worse in fp8 (more small fp8 GEMMs). Left off.
- **CUDA graphs for the routed residual through the MoE layers:** the routed / mHC stream read and
  write now run eagerly (as fused kernels) around the graphed MoE router and postprocess
  (`moe_router` / `moe_preprocess` scopes, `cuda_graph_impl=local`), so the router coefficients
  and the routed gradient stash never cross a graph boundary. Replay matches eager exactly
  (mHC, routed 2 of 8, reversible; test_depth_residual_cuda_graphs.py). Needs
  `moe_permute_fusion` (the unfused permute's backward breaks on replay even without mHC; we
  always fuse). Per-layer graphs still exclude full recompute (Megatron rule), so they apply to
  the no-recompute / selective layouts only.
- fp8 stays off for this model class on H100 (§11.1); fusing quantization into the dispatch is
  the only lever left, and HybridEP has no fp8 dispatch.

### 11.3 Document masking and 8k context (user 2026-10-18)

Pre-training moves to 8k with document masking (`dataloader_inter_document_masking`, as
oellm_32b_dense). The CKDA mixer now takes the packed (THD) batch: `packed_seq_params` ->
fla's `cu_seqlens`, so the recurrence and the short convolutions restart at every document
(packed == separate documents in outputs and gradients, including trailing padding:
test_complex_kda_packed.py). The routed sublayer recompute handles packed batches (the MoE
un-flatten inside the checkpoint). XSA and the output gate are per token. Model-level test
(CKDA + XSA attention + MoE + routed residual, with and without full recompute):
test_hybrid_packed_docs.py, cluster only (local TE attention is broken) -- passes on MN5
(job 47497549, 30 of 32 tests; the 2 CKDA layer tests then failed only on the NGC image's
forced TF32 and now compare at a relative 1e-2). `pretrain_hybrid.py` had no
packed-doc-attention "scatter" path (pretrain_gpt.py's OELLM patch: cu_seqlens broadcast once
per iteration to every pipeline stage), so the first 8k doc-masked speed arms died at startup;
ported. Not wired: packed + context parallelism in CKDA, and packed + MoE partial CUDA graphs.

### 11.4 MoE granularity (user 2026-10-18)

Granularity as 2 h / w_expert: Qwen3.5 / Qwen3-Next 8, DeepSeek-V3 / Kimi K2 7, ours 5.3 for the
production 160 x 768 shape; the round-4 s1/s2 arms 3 (144 x 512 at h 768).
- **Loss, s1 (`screen4g_s1_mn5.yaml`, running):** h5_moe split finer at fixed sparsity (E / k =
  48), fixed active and total expert parameters and the same shared expert: g2 = 288 x 256 top-6,
  g4 = 576 x 128 top-12 (2 h / w = 6 and 12). Pairs with screen4's h5_moe.
- **Speed, production shape (`speed_prodw_mn5.yaml`, running):** ~39B / 2.74B active, 48 blocks,
  8k with document masking, routed residual, full recompute, PP4 / EP4 HybridEP on 4 nodes:
  160 x 768 top-5 + 1 vs 240 x 512 top-8 + 1 (equal total and active), mbs 1 and 2.
- Decision rule (from docs/inference_model_size.md): take 512 if s2 gains >= 0.004 nats and the
  A3B speed is >= 0.9x of 768. The s2 pair is not launched yet (budget, see below).
- **Speed, first result (2026-10-18, 8 nodes, PP4 / EP4, DP8, gbs 128, mbs 1, 8k doc-masked):**
  240 x 512 top-8: 2,526 tok/s/GPU, 42.8 TFLOP/s/GPU, 84 % memory. 160 x 768: hung at the first
  pipeline p2p (NCCL watchdog after 600 s, no iteration completed; not yet diagnosed). 4 nodes
  OOM (expert DP 1: the full fp32 optimizer state of 2.3B expert parameters per GPU, ~50 GB
  before activations; plus a 4 GiB liger CE chunk at 8k -> chunk 2048).
- The 2,526 tok/s is half of m48a2 at 4k without masking (5,359, §11.1). Suspects, to separate
  with one control arm each: the pipeline bubble (gbs 128 = 16 micro-batches per pipeline vs
  gbs 1024), and document masking's host syncs (fla's varlen kernels read cu_seqlens on the host
  in every CKDA call, forward, recompute and backward; plus the per-iteration broadcast).

### 11.5 Flag data ready; status against the goal (2026-10-18 evening)

**Flag mix: complete** (452 / 452 shards, 100 % of the mix weight, 73 TB;
`scripts/korbi/flag_datamix_mn5.py` -> `$HYBRID_ENV/data/flag_datamix_mn5.txt`).
`data_flag_mn5.yaml` is now the 8k pre-training data config: document masking, 16 dataloader
workers + prefetch 8 (the GPFS fix). Smoke (`flagsmoke_s1_mn5.yaml`, generated by gen_screen.py):
the round-4 best bet (CKDA 5:1 + routed 2 of 8 + MoE, s1) for 200 steps at 8k, mbs 1.

| goal item | where | state |
|---|---|---|
| speed optimizations, mbs | §7.2, §11.1-§11.2 | done for the A3B recipe: bf16 (fp8 slower on H100 at these GEMM sizes), mbs 2 at 4k / mbs 1 at 8k, full recompute + routed sublayer recompute, fused router; CUDA graphs for routed + MoE (local, no full recompute) |
| A3B 30B-100B speed, two shapes | §7.2, §11.1, §11.4 | m48a3 (~79B) 3,431, m32a3 (~55B) 3,731, m48a2 (~31B) 5,359 tok/s/GPU at 4k; production 39B shape at 8k doc-masked: 2,526 (512-wide; slow, two controls pending), 768-wide hung |
| Megatron vs torchtitan | §7.1 | decided: Megatron |
| flag data | this section | complete, smoke running |
| parameter-matched feature gate | §2.2, §9.3, rounds 1-4 | in use for every round |
| linear variants, XSA vs attention | §7.3, §10, rounds 3-4 | CKDA 5:1 hybrid + XSA attention won; linear XSA dropped |
| lr / bs sweeps, no-warmup optimizers | §7.4, §7.8, §8.5, §10.2 | angular MuOwn without warmup; lr / bs at s1, b*(D) partial at s2 |
| health metrics + loss prediction | §2.3, §8.2, §9.2-§9.12 | every run fitted (loss_prediction forms, fit from step 1500 for MuOwn) |
| 6BT references | §3.6, §9.5 | s1 done; s2 finalists extend from round 4 by resume (needs budget) |
| MoE granularity | §11.4 | s1 loss arms running; production speed pair partly measured |
| 8k + document masking | §11.3 | implemented and tested (CKDA varlen, routed recompute, scatter in pretrain_hybrid, fused qk-norm with NoPE) |

**Budget: at the 3,000 GPU-h cap.** Next, with the user's OK: the two speed controls for the
8k production shape (no masking; gbs 1024) and the 768-wide hang, the s2 granularity pair, the
s2 finalists to 6BT on the flag mix.

### 11.6 Round 4 interim, round 4R / 4R-x (2026-10-18 night)

Raw paired loss vs h5_moe (same data order; s1 steps 1700-1930, s2 1450-1690): routed 3-of-8
-0.032, mHC-2 -0.026 / -0.025 (s1 / s2), routed 2-of-4 -0.026 / -0.019, keyed routed 2-of-8 -0.025
(s1), plain routed 2-of-8 -0.001 / -0.016, attn_moe -0.008 / -0.012, LatentMoE equal-cost -0.013,
half-cost +0.024. **Dense mHC diverged under MoE** (mHC-4 NaN at step 24 / 32, mHC-2 at 1935 at s1;
the runs went on with NaN gradients: rerun mode off, no monitor rule) -> cancelled; the monitor
now cancels on a non-finite grad norm. Round 4 ran without arch metrics; they are now on every 100
steps, with a new per-layer residual-stream size metric (`<scheme>_<kind>/stream_rms`: main,
others, max, max/min).

Interpretation: the gain follows the number of ACTIVE streams (2 or 3), not the stream count; plain
routing picks from stream 0's content alone and cannot address the others by content (keyed
routing: -0.001 -> -0.025). Cost: at s1 the residual arms ran 33.6k vs 50.3k tok/s/GPU (-33 %),
worth ~0.09 nats at equal GPU time (h5_moe gains 0.093 from 1.5x tokens at 2-3k steps) -- they
must be compared at equal compute, with the production overhead (~12 % at A3B). Part of the s1
gap is that the residual arms ran without the CUDA graphs h5_moe uses.

- **4R (s1, running):** mHC-4, mHC-2, routed 2-of-8, keyed 2-of-8, keyed 3-of-8, all with the
  stream diagnostics every 25 steps.
- **4R-x s2 (running):** mHC-2, keyed 2-of-8, keyed 3-of-8 (does the gain grow with size; pairs with
  screen4_s2 h5_moe).
- **4R-x s1 (running):** keyed 3-of-8 + mixing of the picked streams, + pick balancing; the seed
  noise of a gap (h5_moe and keyed 3-of-8 at seed 1235, paired with each other); keyed 3-of-8 speed
  with CUDA graphs on attn (300 steps).
- **4R-y (s1, running; user 2026-10-18: the routed residual must stay a CHEAP axis, so 2 and 1 of
  8, not 3):** keyed 2-of-8 + pick balancing; + mixing of the picked streams; free routing 2-of-8
  (no forced main stream); free keyed 1-of-8 + balancing (3 rows of stream state per token and
  sublayer). New in Megatron: `mhc_routed_free` -- all k picked by score; the write carries a
  straight-through gate p / sg(p) (softmax over all n logits; value 1) so the selection learns at
  k = 1, where the read weight is constant; balancing covers all n streams.

### 11.7 Run-to-run noise, graph-safe routing statistics, graphed restart (2026-10-18 night)

**Run-to-run noise is large in these MoE runs.** Same config, seed and data order, no CUDA graphs
(round 4 vs its 4R repeat; 10-step mean losses -- round 4 logged every step, 4R every 10, which made
a 2.3-nat "difference" at step 10 a logging artefact): routed 2-of-8 +0.016 / -0.021 / -0.026 /
-0.021 at steps 300 / 500 / 600 / 800; keyed 2-of-8 -0.035 / -0.007 / -0.002 at 300 / 500 / 600;
mHC-2 +0.005 / +0.002. Graphed vs eager (keyed 3-of-8): -0.025 at 300 -- the same size, so not a
graph effect: nondeterministic kernels (MoE permute / scatter atomics, all-to-all, fla backward)
amplified by the chaotic early phase (lr 4e-3, no warmup). Single-run s1 gaps of 0.01-0.03 are
inside this band; round 4's "plain routed 2-of-8 gains ~0" is likely a bad draw (its repeat runs
0.021-0.026 lower). Decisions need repeats (or a deterministic mode, cost to be measured).

**Routing statistics under CUDA graphs** (Megatron e16420cf1): arch metrics graph mode (record on
every step, capture included, in-place into static buffers; reports are interval means; not in the
warmup passes of graph creation) and the MoE router-health / expert-viability trackers recording
inside the router graph (no capture early-return, no host syncs or host-to-device copies; viability
per stage under partial graphs). Tested: graphed runs accumulate exactly the eager counts and
statistics; the tests fail with graph mode off, with warmup recording, or with the old early-return.

**Balanced vs collapsed (s1, 10-step mean loss at step 900, vs h5_moe 3.311; pick entropy
attn / MoE sublayers):** plain routed 2-of-8 3.292 (0.14 / 0.15: collapsed onto 1-2 streams), keyed
2-of-8 3.287 (0.33 / 0.26), keyed 2-of-8 + balancing 3.291 (0.79 / 0.77), + mixing 3.301, keyed
3-of-8 3.292 (0.79 / 0.82), + balancing 3.288 (0.84 / 0.91); keyed 3-of-8 + mixing 3.393 at 700
(h5_moe 3.425). Balancing de-collapses the picks but the loss does not move yet: all within ~0.005
(run-to-run noise ~0.02). "Routed 2-of-8" is 1 + 1 of 8: the main stream always, one of the other
7 by the router (input: the main stream; keyed: + per-stream key tags). **Free keyed 1-of-8 fails:
3.706 (+0.40)** -- every sublayer then reads from one stream only, losing the main residual path;
stopped.

**Graphed restart:** rounds 4R / 4R-x / 4R-y continue with CUDA graphs on the attention / mixer
layers and the MoE router + preprocess (`CG4` in gen_screen.py), each arm resumed from its step-1000
checkpoint. Round 4 (s1 done, s2 near) and 4G finish as they started. Measured graph gain at s1: the
residual arms 30.4k -> 42.6k tok/s/GPU with attention graphs only (the MoE-router graphs come on top);
at A3B ~0 (GPU already 88-93 % busy).

### 11.8 Round 4 learning-curve fits and health (2026-10-19; `refresh_curves.sh`, fit from 1500)

`dump/campaign_3b/curves_all.md` (refreshed every 2 h for rounds 4 / 5). Per run: L = E + A (tau
+ t0)^-alpha in LR-area time, fit on the stable training loss from step 1500 (angular MuOwn,
§9.12); health from losspred's spike / monotonicity detectors.

**Smooth and healthy:** every finished round-4 run (s1 12, s2 6, 4G 2) has 0 spikes, no growth
regime, fit RMSE 0.0056-0.0058 (log-loss; per-step logging) and a per-run backtest (fit the first
half, predict the end) of +0.4 to +0.7 % -- the same sign for all: the curves bend down slightly
faster than one power law (flattening -0.002 to -0.004). The 4R / 4R-x / 4R-y runs, restarted with
graphs, fit as smoothly (RMSE ~0.002 on 10-step means) once their logs are merged (analyze.py
MergedLog; before, the fits saw only the resumed log). Exceptions: free routed 2-of-8 diverged
without a NaN (grad norm 1e2 -> 1e8 from step ~200, loss 5.5 -> 8.7; cancelled; the monitor now
cancels at grad norm >= 1e4); the s2 4R-x arms show one early spike (0.035-0.039) before step 1100;
dense mHC's NaN (round 4 s1) shows no precursor in the loss.

**Extrapolations (10BT, s1 / s2):** s1: h5_moe 2.654, attn_moe 2.642, routed 2-of-4 2.639, keyed
2-of-8 2.636, routed 3-of-8 2.628, LatentMoE (equal cost) 2.618, plain routed 2-of-8 2.669, lat2
half 2.682; s2: h5_moe 2.530, attn_moe 2.528, mHC-2 2.509, routed 2-of-4 2.520, routed 2-of-8 2.539.
**But the seed pair says these are not resolved:** h5_moe seed 1235 (other init AND data order)
extrapolates to 2.617 at 10BT vs 2.654 -- a 0.037 spread (0.013 at 4BT), larger than every gap
above. Single-run extrapolations to 10BT are not discriminative; round 5 decides on joint fits
(shared exponent prior, joint_fit.py --alpha-prior) and on paired seeds of the deciding arms.

### 11.9 The dense-mHC NaN: Sinkhorn's clamped normalizations (2026-10-19)

Every NaN in rounds 4 / 4R / 4R-x went through a Sinkhorn projection -- dense mHC-4 (step ~30,
reproducible), mHC-2 (step 1935) and the routed residual WITH mixing (step 2290) -- while no
routed run without mixing ever did. The exp / normalize form clamped the column and row sums at
1e-8: once the mixing logits spread beyond ~20 (the dynamic part grows), a whole column underflows
after the max subtraction, its sum hits the clamp, the forward returns a different matrix (0.9 off
at scale 20) and the backward of x / sum scales by ~1 / eps^2: NaN gradients with a finite loss --
exactly the observed signature (the loss looked healthy, the grad norm was NaN). Not reproducible
on random tokens at hidden 256 or at full s1 shapes (200 steps, AdamW): the spread needs real
training. Fix (Megatron 0f73ea01c): Sinkhorn in log space (logsumexp), the same fixed point,
finite gradients at any spread; it covers dense mHC, its fused coefficient path and the routed
mixing. Validation: mHC-4 (s1, 4R) rerun fresh, routed 3-of-8 + mixing resumed from 2250.

### 11.10 Round 4 final (2026-10-19): stable loss at step 4000

**Measured loss at step 4000 is reproducible; the 10BT extrapolation is not.** The two seed pairs
(other init AND data order) agree at step 4000 to 0.0003 (h5_moe 2.9002 / 2.8999) and 0.0001
(keyed routed 3-of-8 2.8814 / 2.8813), while their 10BT extrapolations differ by 0.037 / 0.010. So
the gaps at 4000 below are resolved; the extrapolations are only a guide (joint fits next).

| s1 arm (vs h5_moe 2.900) | loss @4000 | gap |
|---|---:|---:|
| mHC-4 (log-space Sinkhorn, 4R) | 2.851 | -0.049 |
| routed 3-of-8, keyed + mixing | 2.870 | -0.030 |
| mHC-2 (4R) | 2.875 | -0.025 |
| LatentMoE equal cost | 2.880 | -0.021 |
| routed 3-of-8 keyed | 2.881 | -0.019 |
| routed 3-of-8 keyed + balancing | 2.885 | -0.015 |
| routed 2-of-8 keyed | 2.888 | -0.013 |
| routed 2-of-8 keyed + balancing | 2.888 | -0.012 |
| granularity x2 / x4 | 2.891 / 2.895 | -0.009 / -0.005 |
| routed 2-of-8 plain | 2.896 | -0.004 |
| routed 2-of-8 keyed + balancing + mixing | 2.898 | -0.002 |
| all-attention MoE | 2.898 | -0.002 |

s2 (vs h5_moe 2.787): routed 3-of-8 keyed 2.757 (-0.030), mHC-2 2.758 / 2.762 (-0.029 / -0.025),
routed 2-of-8 keyed 2.780 (-0.007).

Reading: mHC-4 is the best residual by a wide margin once Sinkhorn is fixed (§11.9). By the
decision rule the 1 + 1-of-8 routed residual does not earn its place (-0.004 / -0.013): the
routed variants that come close need 3 active streams + keys + mixing, losing the "cheap axis"
premise. LatentMoE helps (-0.021), granularity alone little. All-attention = hybrid at 4k (round 5
tests 8k doc-masked). Throughput of mHC-4 vs the routed residual at the production layout: EP agent.

### 12.1 Round 5 status (2026-10-19 morning)

s1: 13 arms (06:34; 3 done, 10 running). s2: 11 arms -- the first launch died at the second
checkpoint save (host OOM: 431 of 503 GB; 16 dataloader workers x prefetch 8 per rank ~400 GB
baseline + the host-side optimizer-state expansion of the save), resumed from step 250 with 8 workers
/ prefetch 2. s3: 4 arms on 4 nodes / EP16 (2 nodes / EP8 OOMed at step 1); qwen35_moe smoke 9.3k
tok/s/GPU, 75 % memory. **Production note:** 16 workers x prefetch 8 leaves little host-memory
headroom (s1 nodes peak at ~423 GB of 503) -- check before the main runs, especially at saves.

### 11.0 Original round-4 outline


Round 4 combines the round-3B MoE recipe (§10.3) with round 3's winners (hybrid recipe, ratio, residual,
optimizer):
- MoE ladder at 2-3 small active sizes (from the speed-tuned 120M-active / 1.2B-total MoE),
  each against its dense twin at equal active parameters and equal D (total parameters and
  memory reported alongside).
- Shape: expert count / width (granularity), top-k, shared expert, total / active ratio
  (8-16x), router (aux-loss vs aux-loss-free balancing). MoE health: expert load, router
  entropy, dropped tokens.
- A short MoE lr / gbs sweep (its optimum differs from dense) and the warmup check under the
  chosen optimizer.
- Throughput at scale from the A3B-30B layout (§7.2), re-checked only for round 3's winners.

## 12. Round 5: the architecture decision before the two main runs (2026-10-19)

Round 5 prepares the two main runs: **A** a Qwen3.5-35B-A3B replica, **B** our hybrid at 2-3B
active (total TBD, ~39B in docs/inference_model_size.md). It is the last small-scale round: every
remaining architecture choice is made here, on the production data setting.

**Protocol (`screen5_{s1,s2,s3}_mn5.yaml`, `gen_screen.py arms5 / write5`):** the flag mix at 8k
with document masking; CUDA graphs on the attention / mixer layers and the MoE router + preprocess
with the routing statistics (graph-safe, §11.7); angular MuOwn without warmup (s1 4e-3, s2 2.5e-3,
s3 1.8e-3), stable only; gbs 64 x 8192 (round 4's token batch), mbs 1; a 4BT schedule (7629
steps) with exit at 4000 (equal data order for every arm; finalists extend by resuming); rolling
checkpoints every 250 steps; the NaN guard; NCCL 4 channels + expandable segments. Every arm has
equal active and total MoE parameters (144 x w, top-3 + shared w; w = d_ffn / 4).

**Sizes** (non-embedding, estimated; + 0.4-0.7B embedding at the 262k vocabulary):

| size | hidden x blocks | q heads x 128 | active | total | nodes |
|---|---|---|---:|---:|---:|
| s1 | 768 x 12 | 6 | ~0.09B | ~2.1B | 1 |
| s2 | 1024 x 18 | 8 | ~0.23B | ~5.7B | 1 |
| s3 | 1280 x 24 | 10 | ~0.5B | ~12B | 2 (EP8, all-to-all) |

**Arms:**

| arm | s1 | s2 | s3 | decides |
|---|:-:|:-:|:-:|---|
| attn_moe: all-attention MoE (RoPE, qk-norm, XSA) | x | x | | is the hybrid worth it at 8k |
| qwen35_moe: GDN 3:1, gated attention, partial RoPE, zero-centred norm, same MoE | x | x | x | main run A's proxy |
| h5_moe: CKDA 5:1 + NoPE / qk-norm / XSA / gated attention | x | x | x | main run B's base |
| + mHC-4 (DeepSeek's stream count) | x | x | x | residual |
| + mHC-2 | x | x | | residual (cheap dense) |
| + routed 1 + 1 of 8, keyed, pick balancing | x | x | x | residual (new, novel) |
| + block AttnRes (4 blocks; ungraphed) | x | x | | residual |
| MoE granularity x2 (288 x w/2, top-6) | x | x | | MoE shape |
| seed pair: h5_moe / + routed, seed 1235 | x | | | noise of a gap |

**Decision rule (user 2026-10-19):** by the fitted curves and their extrapolation to the
overtrained regime (10T tokens: the larger-data estimate, not the loss at 4000 steps), with the
throughput at the production layout (PP: the routed residual sends all 8 streams across every
pipeline boundary, -21 % at full depth per the EP agent; mHC-n sends n). mHC slightly better but
compute-heavy -> dropped; mHC about equal or worse than the routed residual -> the routed residual
(novelty), in its best setting. Gaps are judged against the run-to-run noise (seed pair, §11.7).

**Preconditions:** the angular-MuOwn checkpoint-load fix (Megatron a4545822e: s2 runs could not
resume, OOM in load_state_dict), the dense-mHC NaN diagnosis (mHC-4 NaN at step ~30 under MoE),
the 40-step smoke of every arm. **Auto-start:** when round 4's s2 / 4G arms finish (MN5 login
node, survives this session).

**Budget:** s1 ~10 arms x ~13 GPU-h, s2 8 x ~27, s3 4 x ~50 -> ~550 GPU-h; then lr sweeps (3 lr
for the 2 finalists at s1 / s2, ~160) and the 6BT extensions (~100): ~800 GPU-h, ending near 4.4k
of the 7k hard limit.

### 12.2 Engram allocation sweep (user 2026-10-19)

**Why.** Round 3's Engram (−0.055 to −0.078 at s1, the largest single gain) was a dense model
with ≈ 271M of tables added on top, not parameter-matched: a capacity gain, not a proportion.
Cheng et al. 2026 (arXiv 2601.07372) fix total and active parameters and split the INACTIVE
parameters between routed experts and Engram tables: the loss is U-shaped with its best at
ρ ≈ 75–80 % experts (20–25 % Engram) at both scales they test (5.7B / 568M active and
9.9B / 993M), a shallow U (−0.014 at the optimum vs pure MoE). Engram-27B: 5.7B tables of
26.7B total (≈ 25 % of the inactive). Memorization ("forking", arXiv 2610.00394): n-gram tables
open a train / validation gap at epoch boundaries under data REPLAY (rare contexts drive it,
bigger tables worse); a single pass is unaffected -- watch repeated (up-weighted) sources.

**Arms** (`gen_screen.py` `arms5e`, suites `screen5e_s{1,2}`; baseline = round 5's `h5_moe`,
same data order / cache / seed): h5_moe gives up routed experts (144 → E) for Engram tables of
the freed parameters (per expert 3·h·w, one MoE per block), in two Engram layers (before the
2nd and the middle block), 2- and 3-grams, 8 hash heads × 64 per order, per-layer hashes, the
reference code's signed-sqrt gate and pre-normed conv, DeepSeek's tokenizer compression
(`engram_token_map.py`: OELLM 256k → 175,231 ids, 66.8 %), tables at 5 × lr, sharded over the
stage (`engram_table_parallel: dp`).

| arm | experts | Engram tables | share of inactive |
|---|---:|---:|---:|
| s1 h5_moe (round 5) | 144 | 0 | 0 % |
| s1 h5_moe_eng11 | 128 | 226M | 11 % |
| s1 h5_moe_eng20 | 116 | 396M | 20 % |
| s1 h5_moe_eng34 | 96 | 679M | 34 % |
| s1 h5_moe_eng20_lean | 116 | 396M, lean table optimizer | 20 % |
| s1 h5_moe_eng20_lrx2p5 | 116 | 396M, tables at 2.5 × lr | 20 % |
| s2 h5_moe_eng20 | 116 | 1.09B | 20 % |

The Engram projections add ≈ 1.6M active parameters per Engram layer (≈ 2 % of s1's active).
Cost ≈ 5 × 13 + 27 ≈ 95 GPU-h.

**Megatron work for it** (oellm/v0.19-hybrid 3cc512175):
- Document masking: the n-gram hashes and the dilated causal conv restart at every document
  (positions from cu_seqlens on the device, no host sync); packed = separate, tested.
- The reference code's gate (`engram_gate_signed_sqrt`), conv pre-norm (`engram_conv_prenorm`),
  per-layer hash multipliers (`engram_hash_per_layer`).
- **Memory**: the tables live sharded over the stage's tensor × data × context ranks, so their
  GPU cost per parameter is (fp32 table 4 + fp32 gradient buffer 4 + Adam 8) = 16 B / (stage
  ranks). The lean table optimizer (`engram_table_rowwise_v`: one second moment per row;
  `engram_table_momentum_dtype: bf16`) cuts it to ≈ 10 B; the update now runs over row chunks
  (torch's foreach Adam allocated several table-sized fp32 temporaries). Production estimate
  (≈ 7B of tables, 2 Engram layers, 128 ranks per pipeline stage): ≈ 27M parameters per GPU
  on the two stages that hold them, ≈ 0.45 GB standard / ≈ 0.3 GB lean; on an 8-node test
  layout (8 ranks per stage) ≈ 7 GB / 4.4 GB on those stages -- place the Engram layers off
  the tightest stage (stage 0 holds the embedding).

### 12.3 Linear Depth Memory arm (user 2026-10-19)

González-Martínez, Nazari, Rusch, Liu, "Depth Memory" (arXiv 2610.18809, ELLIS Institute /
MPI-IS Tübingen; code github.com/davidgonmar/depth-memory, MIT). **LDM** keeps the plain
residual stream and adds, per token, a matrix memory M (G groups × K × h/G) of the sublayer
outputs over depth, fit by recursive least squares with an inverse key covariance P (G × K × K)
and learned per-group forgetting: every sublayer reads `x + γ·M(g·q)` (query from the incoming
stream) and writes its output `δ` under a key of the updated stream; the embedding is written
first and read once more before the LM head. ≈ 1 % more parameters. Paper: lowest perplexity
vs PreLN / mHC-4 / Block AttnRes at 400M @ 50B and 1.5B @ 100B (1.5B val ppl 9.15 vs mHC 9.25,
AttnRes 9.26, PreLN 9.48); its throughput (CuTe kernels, state unwinding + windowed writes)
beats mHC in training.

**Here** (Megatron e6c54f610, `megatron/core/transformer/depth_memory.py`): the reference
formulation as a HybridStack depth scheme (the Block-AttnRes route: the state is stashed on the
layers; read / write around every TransformerLayer sublayer and every Mamba / Engram layer
through the ResidualSchemeAdapter). Per-token, so document masking needs nothing. Tested: the
RLS writes reproduce the closed-form weighted ridge solution of the paper's Eq. 6 exactly;
read / write / gradients; packed = separate in the hybrid (cluster). Limits of this first
version: autograd keeps every write's fp32 state (K × h per token per sublayer: ≈ 5 GB per 8k
micro-batch at s1, too much for s2's 50 GB), no CUDA graphs, PP 1, no full recompute -- the
paper's unwinding backward (only u, κ, k, q̃, λ, P_{l-1} saved per write) is the next step if
it wins.

**Arm** `screen5_s1/h5_moe_ldm`: h5_moe + LDM with the paper's setting (G 16, K 8, address
width 32, λ₀ 0.99, α 1; all LDM parameters on Adam at 0.25 × 4e-3 = the paper's 1e-3), no
CUDA graphs (as h5_moe_attnres); paired with round 5's h5_moe / h5_moe_mhc4 / h5_moe_attnres.

### 12.4 LDM and AttnRes with kernels and CUDA graphs (user 2026-10-19)

Both depth schemes kept their state on the layers (a Python object a CUDA graph cannot see) and so
ran without graphs. Now both have a **stream form**: the state travels as `hidden_states`
`[s, b, slots, h]` through the mHC plumbing (TransformerLayer read / write, the Mamba / Engram
adapter, the MoE partial-graph path), so per-layer graphs and the MoE router graphs capture them.

- **Block AttnRes** (`attn_res_impl: streams`, Megatron a6c61a449 / bd1d165d3): slot 0 the
  embedding, slot 1 + j block j; the block being filled sits in the next free slot (sealing moves
  nothing), every sublayer reads a static slice (its source count) through the compiled
  aggregation and its write replaces one slot. = the state form (outputs and gradients, 2 / 3
  blocks), graph replay = eager bit for bit, and -- new -- full activation recompute works.
- **LDM** (`depth_memory_impl: torch | cute`, Megatron d6e2a9544 / 8ad746dc4): the state (M, P)
  lives in a static workspace allocated once per token count (a graph bakes its addresses); every
  write site runs the paper's fused step (write its output, then read with the next site's query)
  at a fixed step index, and owns that next query network (so a layer's graph touches only its
  own parameters); the residual travels as `[x, r_next]`. The backward unwinds M from per-step
  tapes (no per-layer state kept: ≈ 70 MB per sublayer at s1 instead of ≈ 200). `torch`: our
  PyTorch unwinding (the oracle; = the reference implementation in outputs and every gradient).
  `cute`: the paper's CuTe DSL kernels (forward, window-end and interior backward, E = 4 windows;
  vendored MIT into cute_kernels `cute_ldm` ac737f9 with a static workspace in place of their
  per-forward SegmentState; untuned shapes run the default schedule), = the torch oracle on H100
  (reads and all input gradients, 1-11 steps, D 48 / 64 / 128, fp32 / bf16 covariance), graph
  replay = eager bit for bit. Limits: pipeline parallelism 1, no full activation recompute yet (a
  recomputed step would rewrite the workspace; a read-only recompute from the tapes is the fix).

Speed in round 5's setting: suite `speed5res_s1` (100 steps per arm).

### 12.5 Round 5 results (2026-10-20)

All gaps are paired loss differences vs our plain hybrid (`h5_moe`: CKDA 5:1, NoPE / qk-norm /
XSA / gated attention, 144 x w top-3 + shared MoE) at matched steps with the same data order,
step 4000 = 2.1B tokens (flag mix, 8k, document masking). The 10B column is the
gap trend `a + b ln t`, fitted from step 1000 (`scripts/korbi/campaign/gap_trend.py`). In the
backtest (fit 1000-2000, predict 4000) flat gaps predicted to +-0.005, drifting gaps overshot by
up to 0.015 (shown in brackets). Single seed: the s1 seed pair differs by ~0.002. Active / total
sizes estimated from the configs: s1 ~85M / ~2.1B, s2 ~230M / ~5.7B, s3 ~490M / ~12B (non-emb).

**Round 5 grid (1 node per arm; s1 / s2 / s3):**

| variant | s1 @4000 (10B) | s2 @4000 (10B) | s3 @4000 (10B) | with size | with tokens |
|---|---:|---:|---:|---|---|
| mHC-4 (mixer lr x0.25) | -0.051 (-0.048) | -0.040 (-0.039) | -0.037 (-0.038) | shrinks ~N^-0.18 | flat |
| mHC-2 | -0.028 (-0.025) | -0.020 (-0.017) | | shrinks faster | flat |
| mHC-8 | -0.053 (-0.054) | | | (= mHC-4) | flat |
| mHC-4 x LatentMoE | -0.054 ((-0.067)) | | | | widening |
| LatentMoE (288 x w, top-6, latent h/2) | -0.008 ((-0.022)) | -0.007 ((-0.023)) | | stable | widening |
| Block AttnRes (4 blocks) | -0.002 ((-0.029)) | +0.023 ((-0.008)) | | worse | closing |
| MoE granularity x2 | +0.000 | -0.001 | | none | none |
| routed residual 1+1 of 8 (dropped) | -0.007 (+0.024) | | +0.010 (s3 @3130) | | rising |
| Qwen3.5-like (GDN 3:1, gated attn) | +0.023 (+0.032) | +0.012 (+0.025) | +0.004 (+0.028) | shrinks at 2B tokens | rising |
| all-attention MoE | +0.007 ((+0.033)) | -0.004 ((+0.023)) | | | rising |

- **Our hybrid** wins against Qwen3.5 and all-attention at every size. At fixed 2B tokens the lead
  narrows with size, but it grows with tokens: about +0.025 to +0.03 at 10B tokens at all sizes.
- **mHC-4:** flat with tokens, shrinking with size (~N^-0.18; about -0.027 extrapolated to ~3B
  active). mHC-8 adds nothing over mHC-4; mHC-2 is half as good and shrinks faster.
- **LatentMoE:** the one MoE change that holds across size and widens with tokens.
  Granularity x2 does nothing. AttnRes gets worse at s2: dropped.

**Engram allocation (§12.2; Engram tables replace routed experts at equal total and active
parameters):**

| share of the inactive parameters | s1 @4000 (10B) | s2 @4000 (10B) |
|---|---:|---:|
| 11 % (128 experts) | -0.012 ((-0.005)) | |
| 20 % (116) | -0.015 ((-0.007)) | -0.018 ((-0.012)) |
| 20 %, lean table optimizer | -0.016 ((-0.009)) | |
| 20 %, tables at 2.5 x lr | -0.013 ((-0.007)) | |
| 34 % (96) | -0.018 ((-0.013)) | |

- **The Engram gain shrinks with tokens**, unlike mHC (a memorization-like early benefit), and is
  slightly larger at s2.
- **"More table" is better up to 34 %** at s1. That disagrees with the paper's 20-25 % optimum,
  where the U was flat (-0.014 at the optimum). Undecided.
- **The lean table optimizer** (row-wise v, bf16 momentum: 16 -> 10 B per table parameter) costs no
  quality. Production cost: 2 Engram layers -5.4 % throughput (EP agent).

**Residual variants, 2-node suite (`screen5ldm2_s1`, data parallel 8).** The 2-node runs do not
pair with the 1-node ones (+-0.04 per window at step 500 for the same arm), so plain and mHC-4 were
rerun in this setup:

| arm (same 2-node setup) | final loss | gap @4000 | trend | 10B | s1 tok/s/GPU (concurrent, 1 node) |
|---|---:|---:|---|---:|---:|
| plain | 2.3689 | | | | 40,476 |
| **LDM (CuTe) x2.0, per sublayer** | **2.3007** | **-0.068** | flat | **-0.070** | 32,404 (-20 %) |
| LDM x1.0 | 2.3077 | -0.061 | flat | -0.062 | |
| LDM x0.5 | 2.3095 | -0.060 | flat | -0.060 | |
| mHC-4, mixer lr x0.25 | 2.3155 | -0.054 | shrinking | -0.043 | 32,800 (-19 %) |
| mHC-4, mixer lr x1.0 | 2.3258 | -0.043 | shrinking | -0.029 | |
| LDM x2.0, per transformer layer | 2.3295 | -0.040 | slightly shrinking | -0.036 | 36,667 (-9 %) |
| mHC-4 x0.25, per transformer layer | 2.3324 | -0.037 | slightly shrinking | -0.035 | 36,171 (-11 %) |

(The LDM multiplier scales LDM's own parameters' Adam lr: x2.0 = 8e-3.)

- **LDM needs a high lr of its own.** Round 5's LDM ran at x0.25 (1 node) and trailed mHC-4 by about
  0.008. At x2.0 it is the best residual: ahead of mHC-4 (x0.25) by 0.005 at step 1500 and 0.014 at
  step 4000. LDM's gap is flat while mHC-4's shrinks, so the lead widens to ~0.027 at 10B. The optimum
  may lie above x2.0 (the grid's edge); x4.0 is queued.
- **mHC-4 stays at mixer lr x0.25.**
- **Per transformer layer** (`mhc_per_layer`: one read + one write per mixer + MoE block) keeps 59 %
  (LDM) / 69 % (mHC-4) of the per-sublayer gain, for half the s1 overhead. Gains are sublinear in
  the number of mixing sites. LDM, with its input-dependent reads, loses more.
- **At equal compute at s1** (about 0.19 loss per ln of tokens from the fits): LDM per sublayer
  ~-0.026, LDM per layer ~-0.022, mHC-4 per layer ~-0.017, mHC-4 per sublayer ~-0.012 (shrinking).

**LDM at mHC-4's speed (§12.4).** The CuTe stream form (static workspace, fused address networks,
cached launch arguments) runs LDM at mHC-4's tok/s at s1 (32.4k vs 32.8k concurrent). Its GPU
compute is 5 % below mHC-4's (139.6 vs 146.4 ms per micro-batch). The earlier "2x slower" was the
reference implementation (no graphs, fp32 per-layer state) plus the data-loader cap below.

**Production layout (EP agent, `../oellm-autoexp-hybrid-ep/docs/ep_fp8_speed.md` §6d):** 8 nodes,
16k, PP4 x VPP2, recompute offload to the host, fused mHC read:

| arm | tok/s/GPU |
|---|---:|
| plain, mbs 2 | 7,539 |
| mHC-4 | 6,026 (-20 %) |
| mHC-4 per layer | 6,217 (-18 %) |

Per layer halves the mHC kernels but the step barely moves: the pipeline p2p of the 4-wide state
(512 MB per message) is the bound. LDM at scale needs its state (x, r, M, P) to cross recompute
chunks and stages. That is 35 KB per token at K = 8 vs mHC-4's 16 KB, which hurts in exactly this
p2p-bound regime. The EP agent has segmented LDM working (state across chunks / stages, the
next-window gradient folded into dM at boundaries; Megatron 66ce624ac on ep-fp8-speed-r5) and
will measure K 8 / 4.

**Measurement lessons (round 5).**
- **FLAG8K's 4 data workers x prefetch 2** (the fix for the s2 save-time host OOM) capped every s1
  arm at ~23.8k tok/s/GPU: plain, mHC-4 and LDM matched per log interval. Runs launched after
  2026-10-19 noon were slowed; their losses are unaffected. s1 now uses 8 x 4, s2 6 x 3; size
  workers per layout and check tok/s against a known-good run.
- **Wall-clock A/Bs:** concurrent arms, ~50 steps past warm-up. tok/s drifts 10-30 % with the
  launch time on MN5.
- **GPU compute per micro-batch** from torch-profile traces (normalized by the cross-entropy kernel
  count, evaluation out of the window) is the robust speed measure.

**Decisions so far**
- Residual: **LDM per sublayer at lr x2.0 or higher**, if it holds at s2 and if the at-scale state
  is affordable at the production layout. Else mHC-4 per sublayer.
- Keep **LatentMoE** as a candidate for the main run (stacks on mHC-4 at s1). **Engram** is a small
  effect that fades with tokens; still under study.
- Hybrid: **CKDA 5:1 over Qwen3.5-like GDN 3:1 and all-attention** at every size.

**Round 5d, s1 (2-node setup, final; gaps vs the same plain run as above):**

| arm | gap @4000 | trend | 10B |
|---|---:|---|---:|
| LDM x2.0 (above) | -0.068 | flat | -0.070 |
| LDM x4.0 | -0.062 | shrinking (-0.073 at step 1000) | -0.052 |
| LDM x2.0, K = 4 | -0.051 | flat | -0.052 |
| LDM x2.0, mixer sublayers only | -0.048 | slightly shrinking | -0.046 |
| LDM x2.0, MoE sublayers only | -0.014 | flat | -0.011 |

- **lr x2.0 is LDM's optimum.** x4.0 leads early and then loses its edge.
- **K = 4 keeps 75 % of K = 8's gain**, below mHC-4 per sublayer (-0.054), at ~25 KB vs 35 KB per
  token. So K = 8 it is, if the state is affordable.
- **The gain sits at the mixer sublayers:** mixer-only gets 70 % of the full gain, MoE-only 20 %.
  Together they make ~90 % (the sites are roughly additive). Per transformer layer (one site per
  block, -0.040) lands a bit below mixer-only.

**Running:** s2 plain vs mHC-4 x0.25 vs LDM x2.0 (round 5d), the size check for LDM.
- Budget: ~5.6k of 7k GPU-h used before round 5d (~+270).

### 12.6 KronLSTM as the linear mixer (user 2026-10-20)

KronLSTM (github.com/NX-AI/KronLSTM, arXiv 2610.26834) replaces the delta rule's vector residual
with the scalar association v^T S k: one rank-1 write along k v^T per token. A write interferes
only with pairs that overlap in both key and value. It has separate forget / erase / write gates.
The chunkwise form is DeltaNet's WY/UT scheme on the Kronecker features k (x) v (Gram
A_kk (.) A_vv), so it costs about KDA plus a value Gram.

**Integration**
- Ops and layer are vendored from 3d800c9 into our FLA fork (`fla/ops/kronlstm`,
  `fla.layers.KronLSTM`).
- The Megatron mixer is `experimental_attention_variant: kronlstm` (`fla_hybrid.KronLSTM`).
- **Document resets:** packed THD batches pass `cu_seqlens`, which restarts the recurrence, the
  sign parity and the short conv per document. Test: packed = separate through the hybrid.
- **Extended ranges, as CKDA over KDA:**
  - *beta in [0, 2]* is KronLSTM's own `allow_neg_eigval` (`linear_fla_allow_neg_eigval`). It sets
    s = 2 in beta* = s beta / ||v||^2, the negative-eigenvalue half of the contraction range.
  - *Forget gate in [-1, 1]* (`linear_fla_gate: signed_sigmoid2`) uses Complex KDA's gate and
    standard init, run through a sign gauge. With P_t the running sign product per key dim
    (reset per document), the unsigned kernels on (P q, P k) are exact: every use of q and k is
    bilinear against the state, and ||P k|| = ||k|| keeps the unit-norm key. Tested against the
    naive signed recurrence, dense and packed.
- **Context parallelism.** FLA's KCP compresses a rank's effect on the incoming state into a K x K
  transition, which needs a transition acting on the key axis only. KronLSTM's score
  (gamma k)^T S_0 v contracts both axes, so the map S_0 -> S_C is KV x KV. The op therefore
  relays the state instead:
  - Grams, inverse and decay-scaled keys run on all ranks at once.
  - The forward scans run in rank order, each from the previous rank's final state.
  - The read-out runs in parallel again.
  - The backward does the same in reverse: dh0 goes to the previous rank as its dht.
  - A rank whose first document starts on it does not wait. Only the scans serialize (a fraction
    of the layer).
  - The sign parity takes FLA's CP prefix.
  - Tests: CP2 / CP4 vs one process (cut, aligned and spanning documents), and Megatron CP=2 vs
    CP=1.
