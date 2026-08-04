# canGPT (titan-sci) versus Muown — Analysis & Megatron-LM Comparison Plan

Author: analysis for K. Pöppel · Date: 2026-08-03

Scope: (1) dissect the `canGPT_master` model in `../titan-sci` and contrast it
with a *normal* Megatron transformer as configured in
`config/experiments/korbi/megatron_niccolo_130M_smoke_jupiter.yaml`; (2) contrast
canGPT's normalization philosophy with the reference paper
**"Muown: Row-Norm Control for Muon Optimization"** (arXiv 2605.10797, Lion,
Hübler, Li, Orvieto, He); (3) lay out a concrete plan to reproduce and compare
all three in a new Megatron-LM branch.

canGPT is a direct descendant of **nGPT** (Loshchilov, Hsieh, Sun, Ginsburg,
*nGPT: Normalized Transformer with Representation Learning on the Hypersphere*,
arXiv 2410.01131) — the origin of the hypersphere geometry, the eigen-LR LERP
residual, and the `s_qk/s_uv/s_z` scalers. So the comparison is really a
**lineage versus an alternative philosophy**: nGPT→canGPT (normalize the *model*)
versus Muown (normalize inside the *optimizer*).

Source references below are `titan-sci` paths at the current worktree; line
numbers are indicative.

---

## 0. To summarize — the three philosophies

| | **Baseline (Megatron niccolo 130M)** | **canGPT (titan-sci, nGPT-derived)** | **Muown (paper)** |
|---|---|---|---|
| Where norm lives | Pre-LN **RMSNorm** on activations; weights unconstrained | Representations **on the unit hypersphere**; weights **L2-row-normalized**; norm baked into forward *and* optimizer | Standard architecture, **no activation/weight normalization** in forward |
| Residual | Additive `x + sublayer(x)` | **LERP on the sphere** `x' = norm(x + α(â − x))` with analytic norm factor | Additive (unchanged) |
| Scale recovery | Implicit (RMSNorm γ, weight magnitudes) | **Explicit learnable per-channel scalers** (sq/sk/sv/suv/sz/α) with μP eigen-LR reparam | Row-magnitude is an **explicit optimizer variable** |
| Optimizer | AdamW (distributed) | AdamW + post-step **weight normalizer**, or **spherical/Muon** variants (BSA/SBM/BM) | **Muon on the direction** + ℓ∞ update on row-magnitude |
| Goal | conventional | keep everything unit-norm → stable, well-conditioned geometry | **kill spectral-norm drift** without changing the architecture |

The essential contrast the user is after: **canGPT enforces normalization
architecturally** (forward-pass constraints + projection-based weight
normalization; representations *live* on the sphere and scale is re-injected by
learnable scalers). **Muown leaves the architecture standard** and instead
controls weight *row-norms* inside the optimizer by decomposing each weight
matrix into `row-magnitude × row-direction`, optimizing the magnitude explicitly
under ℓ∞ geometry and applying Muon (orthogonalized update) to the direction.
Both target the same disease — uncontrolled growth of weight/representation norms
during training — from opposite ends (model versus optimizer).

---

## 1. Baseline recap — the "normal" Megatron transformer

`megatron_niccolo_130M_smoke_jupiter.yaml` (+ `backend/megatron/base.yaml`):

- **Geometry**: standard pre-norm decoder. `hidden=576`, `layers=18`,
  `heads=9`, `kv_channels=64`, `ffn=2304`, `vocab=50304`, `seq=4096`.
- **Normalization**: `normalization: RMSNorm` (pre-LN before attn and MLP),
  `qk_layernorm: true` (per-head RMSNorm on Q/K), SwiGLU MLP, `add_qkv_bias`.
- **Residuals**: plain additive residuals; no constraint on hidden-state norm.
- **Positional**: RoPE, `rotary_base=100000`, `rotary_percent=1`.
- **Weights**: unconstrained; normal init; `weight_decay=0.1`, `clip_grad=1`.
- **Optimizer**: distributed AdamW, `adam_beta2=0.99`, WSD LR schedule,
  `lr=0.002`, bf16 params with fp32 grad accumulation.

This is the control arm. Everything canGPT/Muown do is a *modification* of this.

---

## 2. canGPT (`canGPT_master`) — full dissection

`canGPT_master` is an **nGPT-style normalized transformer** (Loshchilov et al.,
*nGPT: Normalized Transformer with Representation Learning on the Hypersphere*)
with a very large superset of optional features. The four axes the user asked
about — **scaling factors, normalization, normalization during optimization,
LERP residuals** — plus **how it is optimized** are below.

### 2.1 Normalization (in the forward pass)

The design invariant: **every token representation is a unit vector on
`S^{d-1}`**, and every weight matrix has **unit-norm rows** (or columns for
output projections). Concretely:

- **QK normalization** (`Attention.forward`, model.py:1101-1106): after RoPE,
  `xq, xk = F.normalize(·, dim=-1)` then multiplied by learnable per-dim scalers
  `sq, sk`. `qk_norm=True` by default. Softmax scale is `√head_dim`
  (model.py:917) — because Q,K are unit vectors, the dot product is a cosine, so
  the temperature is set explicitly rather than the usual `1/√d`.
- **Value / activation normalization** (optional): `v_norm` L2-normalizes V
  (model.py:1131); `post_act_norm_attn/ffn` L2-normalize the attention output
  before `wo` and the FFN intermediate before `w2` (model.py:1305, 1399).
- **Post-layer normalization** (`post_layer_norm_attn/ffn`, default True): the
  sublayer output `â` is L2-normalized *before* the LERP residual (see §2.3).
- **Weights** are L2-normalized (rows for input projections `dim=1`, columns for
  output projections when `out_norm_dim_0`) at init and re-projected every
  optimizer step (see §2.4).
- **No RMSNorm/LayerNorm modules** in the trunk — normalization is pure L2 on the
  sphere, not learned-affine. (Optional `pre_head_rms_norm` only at the head.)

### 2.2 Scaling factors (learnable, μP "eigen-LR" reparameterized)

Because activations and weights are forced to unit norm, *all magnitude
information must be re-introduced by explicit learnable scalars*. canGPT does this
pervasively, and uses a **chain-rule / "eigen learning-rate" reparameterization**
so Adam sees a normalized parameter while the effective value starts where you
want it:

- Pattern: a trainable `s` and a **non-trainable** `s_scaler`; the effective
  value used in forward is `s · s_scaler`. Init sets `s = base_scale` and
  `s_scaler = init_value / base_scale`, so `s·s_scaler = init_value` at step 0,
  but Adam's per-step update to `s` is scaled by `s_scaler` — decoupling the
  *initial value* from the *effective learning rate* (μP-style). See
  `_resolve_scale_factor` (model.py:572), `Attention.init_weights`
  (model.py:989-996), `_init_residual_update_params` (model.py:1587).
- **Scalers present**:
  - `sq`, `sk` — per-(head·dim) Q/K scale (nGPT `s_qk`), model.py:889-892.
  - `sv` — per-dim V scale (`use_sv`), model.py:912.
  - `suv` — FFN gate/up intermediate scale (`use_suv`), replaces the fixed
    `1/√(hidden/dim)` up-scale, model.py:1360.
  - `attn_alpha`, `ffn_alpha`, `ple_alpha` — per-dim LERP step sizes (§2.3).
  - `residual_scale` (`block_residual_scale`) — optional per-block σ_l on `h`.
  - **Head scalers**: `pre_head_scale_vector` (learnable `(d,)` on `h`),
    `pre_head_scaler` (fixed `√d/2`), `pre_head_rms_norm`; and post-head
    `sz` (per-vocab or scalar logit scale), `head_bias`, input-dependent
    `temp = softplus(h_n·w_temp+b_temp)` (model.py:2515-2608). `sz` init modes
    target `σ_logits ≈ sz_init/√d` (model.py:587-601).
- **Fixed (non-learnable) FFN scales**: `scale_up = 1/√(hidden/dim)`,
  `scale_down = 1/√(dim/hidden)`, SwiGLU activation scale `√hidden`
  (model.py:1351-1352, 1396) — μP-correct magnitude bookkeeping.

### 2.3 LERP residuals — the analytic normalization factor (the crux)

nGPT replaces additive residuals with a **spherical linear interpolation** toward
the (normalized) sublayer output. `TransformerBlock._residual_update`
(model.py:1597-1672):

```
α  = activation(attn_alpha · attn_alpha_scaler)          # per-dim step, fp32
â  = normalize(sublayer_out)          if post_layer_norm  # unit vector
x' = normalize( x + α · (â − x) )     = normalize( (1−α)x + α·â )
```

The key design choice — documented in `docs/Normalization_Factor.md` — is **how**
`normalize(...)` is done. Two methods:

- **Method A — post-hoc renormalization**: `x' = h / ‖h‖`, `h=(1−α)x+α·â`.
  Its Jacobian `∂x'/∂x = (1−α)/‖h‖ · (I − ĥĥᵀ)` is a **rank-deficient tangent
  projector** (det = 0): backprop removes the radial gradient component.
- **Method B — analytic normalization factor** (the "canary" trick): since
  `x` and `â` are already unit vectors, `‖h‖² = (1−α)² + 2α(1−α)cosθ + α²` with
  `cosθ = xᵀâ`, so `x' = f·h`, `f = 1/√((1−α)²+2α(1−α)cosθ+α²)`. This keeps a
  **full-rank Jacobian** (det > 0), is C∞ in α, and behaves as a "gentle nudge"
  `f ≈ 1 + α(cosθ−1)` for small α — preserving radial gradient information.

The implementation exposes both plus hybrids by way of `lerp_normalization`
(model.py:1621-1670):
- `approx` (default): uses the *simplified* factor
  `f = rsqrt(1 − 2α(1−α))` (assumes `cosθ≈0`, that is orthogonal update — the nGPT
  approximation), forward has a slowly **growing residual norm**.
- `direct`: Method A (`F.normalize` after LERP) — exact sphere, projection
  backward.
- `both`: approx factor *then* exact normalize.
- `direct_ste` / `both_ste` / `approx_fwd`: straight-through-estimator variants
  that decouple the forward norm from the backward Jacobian (used to ablate
  whether the benefit comes from the growing-norm forward or the scalar-vs-
  projection backward). This is exactly the Method A/B question turned into a
  controlled ablation.

`α` itself is bounded by way of `alpha_activation` (model.py:1542): `abs` (unbounded,
default), `half_sigmoid`/`half_tanh_sq` (cap at 0.5 — the principled LERP
ceiling), `tanh_smooth_bound` (norm_gpt 0.95 ceiling). Init solves the
pre-activation so `α_eff(0) = alpha_init_value` (default 0.05) for any activation
(model.py:1563-1585). There is an optional **3rd LERP sublayer** (Per-Layer
Embeddings / PLE, model.py:1756-1782).

### 2.4 Normalization *during optimization* — how it is optimized

Two complementary mechanisms keep weights on the manifold; the model selects them
by way of `optimizer.name` and the `normalizer` config block
(`weight_norm_optimizer.py`, `components/gpt_normalizer.py`,
`optimizer/*`):

1. **Post-step weight normalizer (`GPTNormalizer`)** — the default nGPT path.
   Registered as a `register_step_post_hook` on the optimizer
   (weight_norm_optimizer.py:664-669). After every Adam(W) step it calls
   `normalize_weights()` which, per weight matrix, applies:
   - **L2 normalization** — rows to unit norm (`dim=1`), or columns (`dim=0`) for
     output projections when `out_norm_dim_0`; optional whole-tensor Frobenius.
   - **bounded** mode — only shrink (`w/max(‖w‖, max_norm)`), never grow.
   - optional **spectral normalization** (power iteration → `w/σ₁`) and weight
     **rounding/INT-quantization** stages.
   - `normalize_every_n_steps` throttle.
   Body weights therefore get `weight_decay=0` (the projection controls
   magnitude); only unconstrained head params get wd=0.1 by way of a dedicated param
   group (weight_norm_optimizer.py:403-481, 568-643). Per-channel **scalers** can
   be split into their own AdamW group with dedicated wd/β (the "scaler group").

2. **Constrained / spherical / Muon optimizers** — instead of "Adam then project",
   these fold the constraint into the step (`optimizer/`):
   - **`constrained_adam`** — Adam with row-norm constraint + optional momentum
     projection / parallel transport (a Riemannian-lite Adam).
   - **`bounded_spherical_adam` (BSA)** — tangent-space projection of the gradient
     (`g ← g − (g·ŵ)ŵ`) *before* Adam, boundary-predictive activation, then a
     post-step row projection to `‖w_i‖ = 1` (normalized) or `≤ max_norm`
     (bounded). Modes: `normalized`, `bounded`, `partial_orthogonal[_bounded]`
     (spectral→Newton–Schulz→row-normalize).
   - **`shared_bound_muon` (SBM)** — BSA + **Newton–Schulz orthogonalization of
     the gradient** before Adam (`muon_on_gradient`, `muon_ns_steps`,
     `muon_ns_mode ∈ {muon, convergent, cubic, polar_express}`).
   - **`bounded_muon` (BM)** — true Muon: momentum buffer → all-gather →
     global NS orthogonalization → Moonlight RMS scaling `0.2·√max(R,C)` → SGD
     update on the local shard, then BSA-style row projection. Optional geodesic
     (Riemannian exp-map) and distributed-ownership NS.
   - Plus Cayley / orthogonal-reprojection / Riemannian variants.

The through-line: canGPT's weights and activations are **kept on a product of
spheres/Stiefel manifolds** by a combination of *architecture* (forward L2 +
LERP) and *optimizer projection* (post-step normalize or Riemannian step).

### 2.5 Other canGPT features (context, not core to this comparison)

SSMax scalable-softmax, sliding-window / global-local layers, p-RoPE, K=V
sharing, attention gating, ReLU² FFN, scheduled post-layer dropout, chunked/Liger
fused head loss, FP8. These are orthogonal knobs and should be **off** for the
apples-to-apples comparison.

---

### 2.6 Origin: nGPT (arXiv 2410.01131)

canGPT is the engineering superset of **nGPT** (Loshchilov, Hsieh, Sun,
Ginsburg, NVIDIA). Mapping the paper onto the code above:

- **Hypersphere invariant**: *all* vectors — embeddings, MLP/attention matrix
  rows, and hidden states — are constrained to unit norm; a token representation
  "travels on the surface of a hypersphere," and each layer applies a
  **normalized displacement** toward the (normalized) sublayer output. This is
  exactly canGPT's L2-everywhere + LERP residual (§2.1, §2.3).
- **Eigen learning rate / LERP update**: nGPT's residual is
  `h ← Norm(h + α_A · (Norm(h_attn) − h))` for attention and the analogous
  `α_M` for the MLP, where `α_A, α_M` are **learnable per-dimension "eigen
  learning rates."** canGPT's `attn_alpha`/`ffn_alpha` with the eigen-LR
  reparam (`α = raw · scaler`) are the literal implementation; the analytic
  normalization factor (`docs/Normalization_Factor.md`, §2.3) is canGPT's
  refinement over nGPT's `Norm(...)`.
- **Scaling factors**: nGPT introduces `s_qk` (restore Q/K scale after
  normalization), `s_uv` (SwiGLU intermediate scale), and `s_z` (logit scale),
  each stored with a **base value / true value decomposition** so the effective
  scale and its learning rate are decoupled — canGPT's `sq/sk`, `suv`, `sz`
  with `_resolve_scale_factor` eigen-LR (§2.2) are one-to-one with these.
- **Weight normalization during optimization**: nGPT normalizes all matrices
  along their embedding dimension **after every optimizer step** (there is no
  weight decay and no separate LayerNorm/RMSNorm) — canGPT's `GPTNormalizer`
  post-step hook (§2.4) is precisely this, extended with bounded/spectral/Muon
  variants.
- **Result**: nGPT reports **4–20× fewer training steps** to reach the same
  accuracy (larger factor at longer sequence length) and improved conditioning.
  This is the upside Arm B is meant to reproduce/measure in Megatron.

So Arm B below = "nGPT in Megatron," with canGPT's analytic-LERP and optimizer
variants as ablation sub-arms.

## 3. Muown (arXiv 2605.10797) — contrast

**Core idea**: keep a *standard* transformer, but fix Muon's tendency to let the
**spectral norm of weight matrices drift** upward during training. The authors
decompose each weight matrix's spectral norm into two factors —
**row-magnitude** and **row-coherence (direction)** — and observe the drift comes
entirely from the *row-magnitude* factor while the direction is stable.

**Method**:
- Treat the **row-magnitude vector as an explicit optimizer variable**, updated
  under the **ℓ∞ geometry** induced by the decomposition (a dual-norm update, not
  a projection).
- Apply **Muon unchanged** to the remaining **direction** component (Newton–Schulz
  orthogonalized momentum update).
- No hypersphere normalization, **no LERP residual**, no forward-pass activation
  normalization, no explicit learnable per-channel scalers.

**Claims**: optimal non-convex rates (deterministic & stochastic) with better
noise constants; across GPT 124M–2.7B — better perplexity, wider LR plateau,
lower weight-decay sensitivity, and **eliminated spectral-norm drift** at minimal
overhead.

### 3.1 canGPT versus Muown — where the "normalization" acts

| Axis | **canGPT** | **Muown** |
|---|---|---|
| Locus | Architecture + optimizer projection | Optimizer only |
| Weight norm target | **row L2 = 1** (per-row unit) — projection to sphere | **row-magnitude** as a tracked variable under ℓ∞; spectral-norm drift removed, magnitudes *free but controlled* |
| Direction handling | rows normalized; optional NS (partial-orthogonal / Muon) | **Muon (NS) on the direction** is the core |
| Residual | LERP on sphere (analytic factor) | standard additive |
| Scale recovery | explicit learnable scalers (sq/sk/sv/suv/sz/α, eigen-LR) | row-magnitude optimizer variable — no architectural scalers |
| Activation norm | unit-hypersphere (L2 everywhere) | none |
| Overhead | high (L2 + LERP + many scalers + optional NS) | low (one extra vector update per matrix) |
| Failure mode addressed | representation collapse / uncontrolled magnitude | spectral-norm drift under Muon |

The two are **not mutually exclusive** and this is the interesting comparison:
canGPT's `bounded_muon` already puts Muon on unit-norm rows (magnitude pinned to
1); Muown instead lets Muon run on the *direction* while a **separate ℓ∞ update**
governs magnitude. Muown is effectively "the magnitude degree of freedom canGPT
throws away, kept and optimized properly."

---

## 4. Plan — new Megatron-LM branch to compare the three

Goal: reproduce, in the Megatron-LM training stack used by oellm-autoexp, three
comparable arms at the niccolo-130M scale (and a 1.3B follow-up), sharing data,
tokenizer, batch, and schedule, differing only in the normalization/optimization
scheme.

### 4.1 Branch & layout

- Submodule: `submodules/Megatron-LM` (fork
  `OpenEuroLLM/NVIDIA-Megatron-LM`).
- New branch: **`feature/norm-geometry-comparison`** (or
  `korbi/cangpt-vs-muown`).
- Keep changes additive and flag-gated (`TransformerConfig` fields, default off →
  bit-identical to baseline when disabled), mirroring how titan-sci made every
  feature a config knob.

### 4.2 Three arms

- **Arm A — Baseline**: exactly `megatron_niccolo_130M_smoke_jupiter.yaml`.
  Standard RMSNorm pre-LN, additive residuals, AdamW. Control.
- **Arm B — canGPT/nGPT-in-Megatron**: implement the nGPT geometry:
  1. L2-normalize Q/K after RoPE + learnable `sq/sk` (Megatron already has
     `qk_layernorm`; add the nGPT variant / scalers).
  2. Replace pre-LN + additive residual in `TransformerLayer` with the
     **LERP-on-sphere residual** (`_residual_update`), incl. the
     `approx/direct/both/*_ste` modes and per-dim `attn_alpha/ffn_alpha` with
     eigen-LR reparam.
  3. Learnable `suv` FFN scaler and fixed μP up/down scales.
  4. Head: `pre_head_scale_vector` + post-head `sz` logit scaler.
  5. Post-step **weight L2-normalizer** hook (rows `dim=1`, output cols `dim=0`),
     `weight_decay=0` on body, wd=0.1 on unconstrained head. Optionally the
     spherical/Muon optimizer variants as a sub-arm (B-muon).
- **Arm C — Muown**: keep Arm A's *architecture* untouched; implement the
  **Muown optimizer**: per-2D-weight row-magnitude/direction split, Muon (NS
  momentum) on the direction, ℓ∞ update on the row-magnitude vector. Applies to
  attention/MLP projection matrices; embeddings/head/1D params fall back to Adam.

A useful 4th arm, **C′ — Muon-plain** (Muon without Muown's row-norm control),
isolates *what Muown's magnitude control adds over vanilla Muon* and connects to
canGPT's `bounded_muon` (Muon on pinned-unit rows).

### 4.3 Concrete Megatron touch-points

- `megatron/core/transformer/transformer_config.py` — add gated fields:
  `use_ngpt_geometry`, `lerp_normalization`, `alpha_init_value`,
  `alpha_activation`, `use_sq_sk`, `use_suv`, `post_head_sz*`,
  `weight_normalizer_*`, `optimizer=muown|muon|...`.
- `megatron/core/transformer/transformer_layer.py` — swap the residual/bias-
  dropout-add path for `_residual_update` when nGPT geometry is on; L2 the
  sublayer output.
- `megatron/core/transformer/attention.py` +
  `dot_product_attention.py` — QK L2 + `sq/sk` scalers; softmax scale = `√d`.
- `megatron/core/transformer/mlp.py` — `suv`/fixed μP scales; optional post-act
  L2.
- Head (`GPTModel` output layer) — `pre_head_scale_vector`, `sz`.
- **Optimizer**: new `megatron/core/optimizer/` entries — a post-step
  weight-normalize hook (Arm B) and a **Muown/Muon** distributed optimizer (Arm
  C). Muon/Muown need the full 2D matrix for Newton–Schulz; under Megatron's
  distributed optimizer this means a gather/scatter per matrix (mirror
  titan-sci `bounded_muon`'s all-gather-NS-scatter, or restrict Arm C to TP=1/DP
  for the 130M smoke first).
- Init: row-normalized weight init (`normalize_weight`, orthogonal/partial-
  orthogonal options) for Arm B.

Port math directly from titan-sci (`_residual_update`, `_resolve_scale_factor`,
`GPTNormalizer.normalize_weights`, `bounded_muon`) — it is reference-quality and
already handles the fp32/DTensor/eps subtleties.

### 4.4 Held-fixed experiment matrix (130M smoke first)

Fixed across all arms: data (`data_nemonicco_jupiter`), tokenizer
(gpt-neox-20b), `seq=4096`, `gbs=128`, `train_iters` for a fixed token budget
(≈6B for smoke), WSD schedule, `clip_grad=1`, 2×GH200 nodes, bf16 + fp32 grad
accum.

| Arm | Norm geometry | Residual | Optimizer | Notes |
|---|---|---|---|---|
| A | RMSNorm pre-LN | additive | AdamW | control |
| B | hypersphere L2 + scalers | LERP `approx` | AdamW + weight-normalizer | nGPT default |
| B-abl | " | LERP `direct`/`both`/`*_ste` | " | Method A vs B ablation (§2.3) |
| B-muon | " | LERP | bounded_spherical_adam / bounded_muon | spherical optimizer |
| C | standard | additive | **Muown** | paper repro |
| C′ | standard | additive | Muon-plain | isolates Muown's magnitude control |

LR is *not* shared blindly — nGPT and Muon have different optimal LRs; run a
small LR sweep per arm (for example ×{0.5,1,2} around each family's known-good) and
compare each arm at *its own* best LR, plus report the width of the LR plateau
(Muown's headline claim).

### 4.5 Metrics & instrumentation

- Train/val loss & perplexity versus tokens (primary); tok/s and MFU (overhead).
- **LR-plateau width** per arm (Muown claim) and **weight-decay sensitivity**.
- **Spectral-norm drift**: log per-matrix `σ₁` and row-norm distributions over
  training (the quantity Muown targets and canGPT pins to 1). This is the single
  most informative diagnostic to compare the three philosophies directly.
- For Arm B: log `α_eff` per layer, scaler magnitudes (`sq/sk/suv/sz`), and
  residual-norm growth under `approx`.
- Gradient/update norms; cosθ of the LERP update (Arm B).

### 4.6 Phasing

1. **P0 — scaffolding**: branch, gated config fields, logging of σ₁/row-norms on
   the *baseline* (no behavior change). Verify bit-identical to Arm A.
2. **P1 — Arm B (nGPT)**: QK-norm+scalers → LERP residual → weight normalizer →
   head scalers. Smoke-train 130M; confirm stability and that
   `direct` vs `approx` reproduces the Method A/B gap from
   `docs/Normalization_Factor.md`.
3. **P2 — Arm C (Muown) + C′ (Muon)**: implement the optimizer at TP=1 first,
   then distributed. Smoke-train 130M.
4. **P3 — comparison run**: full matrix at 130M on the fixed token budget with
   per-arm LR sweep; produce the loss/perplexity + σ₁-drift + LR-plateau report.
5. **P4 — scale check**: best 1–2 arms at ~1.3B to confirm rankings hold.

### 4.7 Risks / open questions

- **Muon/Muown × Megatron distributed optimizer**: Newton–Schulz needs the whole
  2D matrix; the distributed optimizer shards it. Start TP=1/PP=1 with DP-only
  gather; port titan-sci's distributed-NS ownership scheme later. (Cross-ref the
  `megatron_resilience` / distributed-optimizer notes in memory.)
- **Fair LR**: the three families have different LR scales — comparing at a single
  LR would be misleading; hence per-arm sweeps.
- **Torch.compile / CUDA-graph** interaction with L2/LERP fp32 sections and with
  NS iterations (titan-sci hit Inductor issues with complex RoPE and SymInt head
  dims — keep dims static, avoid complex RoPE).
- **Scope creep**: keep canGPT's optional features (SSMax, sliding window, PLE,
  gating, FP8) *off*; they confound the comparison.
- **Weight-decay semantics differ**: Arm B uses wd=0 on normalized body; Muown
  claims low wd-sensitivity — report wd as a swept axis, not a fixed constant.

---

## 5. Appendix — key titan-sci references

- Model: `titan_sci/models/canGPT_master/model/model.py`
  - LERP residual & modes: `_residual_update` (1597-1672); block forward
    (1707-1786).
  - Analytic factor doc: `docs/Normalization_Factor.md` (Method A vs B).
  - Scalers / eigen-LR: `_resolve_scale_factor` (572), `Attention` sq/sk
    (889-996), FFN suv (1351-1439), head (1954-2024, 2515-2608).
  - QK/V/act normalization: 1101-1131, 1304-1305, 1398-1399.
- Args / knobs: `titan_sci/models/canGPT_master/model/args.py`.
- Optimization: `titan_sci/components/weight_norm_optimizer.py`,
  `titan_sci/components/gpt_normalizer.py`, `titan_sci/optimizer/{constrained_adam,
  shared_bound_adam, shared_bound_muon, bounded_muon}.py`.
- Origin paper: Loshchilov, Hsieh, Sun, Ginsburg, *nGPT: Normalized Transformer
  with Representation Learning on the Hypersphere*, arXiv 2410.01131 (4–20×
  faster training; source of hypersphere geometry, eigen-LR LERP, s_qk/s_uv/s_z).
- Contrast paper: Lion, Hübler, Li, Orvieto, He, *Muown: Row-Norm Control for
  Muon Optimization*, arXiv 2605.10797.
</content>
</invoke>

---

## 6. Measured throughput (JUPITER, 130M, 4x GH200) — 22 tuning rounds

Final, all arms at their best measured `gbs/mbs`:

| arm | start | best | >200 |
|---|---|---|---|
| probe 1.3B baseline | – | 440.2 | yes |
| baseline | 157 | 283.3 | yes |
| muown | 126 | 270.9 | yes |
| ngpt_approx | 107 | 223.9 | yes |
| ngpt_analytic | 88 | 192.0 | no |
| ngpt_wnorm | 95 | 191.5 | no |
| attnres8 | 60 | 152.1 | no |
| attnres4_lerp | 64 | 144.3 | no |
| mhc4 | 42 | 96.2 | no |
| mhc2_lerp | 34 | 67.3 | no |

### What produced the gains
1. **Liger fused linear+CE** (merged from the hybrid branch): removes the
   `(b,s,V)` logits, ~0.41 GiB per micro-batch sample at V=50304/seq=4096.
2. **torch.compile** (+19–26% alone): was silently disabled by an inherited
   `TORCHDYNAMO_DISABLE=1`; needs `TRITON_LIBCUDA_PATH` set to the *directory*
   holding `libcuda.so.1`.
3. **No dataloader attention mask**: FlashAttention takes `is_causal` and
   discards it, so the `[1,s,s]` fp32 tensor is built and broadcast for nothing.
4. **Dropping `overlap_grad_reduce/param_gather`**: the hybrid logs measure
   176.9 vs 317.2 against the chunk-8192 variant at identical batch.
5. **Removing retained fp32 copies** in nGPT/mHC/AttnRes (~10 GiB): `mul`,
   `normalize` and `pow` save their operands, so `.float()` on an `[s,b,h]`
   tensor keeps a full fp32 copy alive through backward.
6. **Compile-fusing** the Sinkhorn loop, the AttnRes aggregation and the mHC
   coefficient path — these modules are launch-bound, not compute-bound.

### Why the remaining six fall short (diagnosed, not speculative)
* **nGPT analytic/wnorm (~192).** `alpha` is per-channel `[h]` and `cos_theta`
  per-token `[s,b,1]`, so `denom` broadcasts to a full `[s,b,h]`, plus another
  for its `rsqrt`. That is ~2 more full activations than `approx`, which is why
  `approx` sustains mbs=32 (223.9) and these cannot. Fix: fuse the correction
  into one kernel, or use a scalar alpha.
* **AttnRes / mHC.** Cannot use CUDA graphs at all: `DepthResidualState` holds
  Python lists of freshly allocated tensors and is reached by way of
  `layer._depth_state`. Every arm above 200 has graphs; none below do. Round 22
  also showed fusion and memory trading against each other here (attnres8 got
  faster, then OOMed), which is the signature of the data structure rather than
  the kernels being the constraint. Fix: preallocated `[N+1,s,b,h]` buffer,
  static shapes, state passed as tensor arguments.
* **mHC additionally.** Residual traffic scales with `n`: at n=4 every layer
  moves 4x the residual, and 96.2/283.3 ≈ 0.34 is about what 4x traffic on the
  dominant term predicts. The paper reaches 6.7% overhead only with fused
  TileLang kernels plus selective recompute. A torchtitan implementation is
  reported near 400 TFLOP/s, so this is an implementation gap, not a property
  of the method — but closing it means writing the fused mixing kernel.
