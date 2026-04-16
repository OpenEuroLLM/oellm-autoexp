# MoE Router Metrics Reference

This document describes every metric currently logged to W&B and TensorBoard during MoE training.
It is intended as a quick lookup for anyone inspecting training runs.

## How Metrics Are Collected

Metrics come from two independent in-memory trackers that accumulate values across micro-batches
and are flushed at each logging interval.

- **Aux losses tracker** (`_MOE_LAYER_WISE_LOGGING_TRACKER`): stores one scalar per layer. Router
  code calls `save_to_aux_losses_tracker(name, value, layer_number, ...)` each forward pass;
  values are summed across micro-batches, reduced across ranks, and logged at the interval.
- **Expert utilization tracker** (`_MOE_EXPERT_UTILIZATION_TRACKER`): stores per-expert token
  counts per layer. The router accumulates raw counts every forward pass; statistics are derived
  from the accumulated counts at logging time.

Both trackers are cleared after each log interval so values represent that interval only, not
cumulative training history.

**Per-layer logging:** when `moe_per_layer_logging: True` is set in the experiment config, every
metric also appears as `moe/{name}_layer_{i}` (0-indexed layer). The aggregate top-level value is
always the mean across all MoE layers.

---

## Section 1 — Load Balancing Losses

These losses are active only when the corresponding routing type or coefficient is configured.
They are **training objectives, not pure diagnostics**: their gradients flow back into the router
and affect token routing. The logged value is the **unscaled loss** (divided by its coefficient),
so values are directly comparable across runs with different coefficients.

| W&B key | Active when | Paper |
|---|---|---|
| `load_balancing_loss` | `moe_router_load_balancing_type: aux_loss` | [Switch Transformer, Fedus et al. 2021](https://arxiv.org/abs/2101.03961) |
| `seq_load_balancing_loss` | `moe_router_load_balancing_type: seq_aux_loss` | per-sequence variant of the above |
| `global_load_balancing_loss` | `moe_router_load_balancing_type: global_aux_loss` | running-average over global batch, [Pan et al. 2025](https://arxiv.org/abs/2501.11873) |
| `z_loss` | `moe_z_loss_coeff` set to a non-zero value | [ST-MoE, Zoph et al. 2022](https://arxiv.org/pdf/2202.08906) |

### `load_balancing_loss` / `seq_load_balancing_loss` / `global_load_balancing_loss`

All three variants implement the Switch Transformer auxiliary loss formula:

```
L_aux = num_experts * sum_i ( f_i * P_i )
```

where `f_i` is the fraction of tokens dispatched to expert `i` and `P_i` is the mean router
probability assigned to expert `i`. The loss is minimised when load is perfectly uniform.

- **Healthy range:** small positive value, stable or slowly decreasing.
- **Warning:** a sudden spike indicates the router is attempting to correct a large imbalance.
  A value permanently at zero (with `coeff > 0`) may indicate the config is not wiring the loss
  correctly.

### `z_loss`

Penalises large router logits by adding `mean( log(sum_e exp(logit_e))^2 )` to the loss. This
discourages the partition function from growing, which prevents routing from becoming numerically
deterministic. It does not directly penalise imbalance.

- **Healthy range:** small positive value, roughly constant across training.
- **Warning:** a growing `z_loss` means logit magnitudes are increasing despite the penalty —
  this can precede router saturation. A flat, near-zero value is expected when the penalty is
  working.

---

## Section 2 — Router Diagnostics

These two metrics are **always computed** regardless of which aux loss is configured. They are
pure diagnostics with no gradient — computed inside `torch.no_grad()` on the raw logits
immediately after the z-loss attachment and before the top-k selection. They capture pre-decision
router state that post-routing metrics (token counts) cannot reveal.

Source: `TopKRouter.routing()` in
`submodules/Megatron-LM/megatron/core/transformer/moe/router.py`.

---

### `router_logit_logsumexp`

**Formula:**
```
mean over tokens of:  log( sum_e exp(logit_e) )
```
This is the mean log-partition-function of the routing distribution — the quantity that z-loss
penalises the square of.

**Interpretation:** measures the overall *scale* of the router logits. When logits are all zero
the value is `log(num_experts)`. As the router learns to prefer certain experts its logits grow
in magnitude, increasing this value. Unchecked growth indicates saturation: the softmax is
approaching a one-hot distribution even before top-k selection.

| Condition | Expected value (64 experts) |
|---|---|
| Random init, uniform routing | ≈ 4.16 (`log(64)`) |
| Healthy learned routing | 4.5 – 6.0 |
| Router saturation / collapse onset | > 7.0, still growing |

**Why it matters:** this is the earliest observable signal of collapse. It starts rising well
before token-count metrics like CV% or dead-expert count change, because those only reflect the
hard routing decisions, not the confidence behind them.

---

### `router_mean_max_prob`

**Formula:**
```
mean over tokens of:  max_e softmax(logits)_e
```
The full softmax is taken over **all** `num_experts` experts, not just the selected top-k. This
gives the probability the router assigns to its single most-preferred expert for each token,
before any truncation or normalisation to the top-k.

**Interpretation:** measures routing *confidence* independently of load distribution. A router
can be perfectly balanced (every expert gets the same number of tokens) yet have high
`router_mean_max_prob` if it is confident about different experts for different tokens. This is
the key distinction from entropy-on-counts: load balance and routing sharpness are orthogonal,
and this metric captures sharpness.

| Condition | Expected value (top-8 / 64 experts) |
|---|---|
| Random init, uniform routing | ≈ 0.016 (= 1/64) |
| Healthy selective routing | 0.05 – 0.20 |
| Sharp but balanced routing | 0.20 – 0.50 |
| Routing near-collapse | > 0.70, approaching 1.0 |

**Why it matters:** it catches the "confident-but-balanced" regime that entropy-on-counts misses.
If `router_mean_max_prob` rises while `moe/expert_cv_pct` stays flat, the router is becoming
sharper without yet concentrating load — an early warning that collapse is likely unless
regularised.

---

## Section 3 — Expert Utilization

These metrics are derived from accumulated token counts per expert per layer. They are always
computed when the utilization tracker is populated (which happens every forward pass in
`TopKRouter.routing()`).

Source: `track_moe_metrics()` in
`submodules/Megatron-LM/megatron/core/transformer/moe/moe_utils.py`.

All aggregate values below are **means across all MoE layers** unless stated otherwise.

---

### `moe/expert_cv_pct`

**Formula:** `( std(tokens_per_expert) / mean(tokens_per_expert) ) * 100`

The coefficient of variation of the token-count distribution across experts, expressed as a
percentage. This is the primary load-balance health indicator.

| Value | Interpretation |
|---|---|
| 0% | Perfect balance — every expert receives exactly the same number of tokens |
| < 30% | Healthy; minor fluctuations are normal |
| 30 – 60% | Moderate imbalance; worth monitoring |
| > 60% | Severe imbalance; likely harming training efficiency and model quality |

---

### `moe/expert_active_frac`

**Formula:** `count(experts with tokens > 0) / num_experts`

Fraction of experts that received at least one token during the log window. An expert that
receives zero tokens in an entire log interval is effectively unused.

| Value | Interpretation |
|---|---|
| 1.0 | All experts active |
| 0.8 – 1.0 | A few experts occasionally idle; monitor for trend |
| < 0.8 | Structural dead experts; routing is collapsing |

---

### `moe/expert_max_frac`

**Formula:** `max(tokens_per_expert) / sum(tokens_per_expert)`

The fraction of all dispatched token-expert assignments captured by the single busiest expert.

| Value | Interpretation |
|---|---|
| 1/num_experts ≈ 0.016 (64 experts) | Perfect balance |
| < 0.05 | Healthy |
| 0.05 – 0.10 | Mild hot-spot; watch trend |
| > 0.10 | One expert handling >10% of all load; severe imbalance |

---

### `moe/expert_mean_entropy`

**Formula:** `mean over layers of: -sum_e p_e * log(p_e)` where `p_e = tokens_e / total_tokens`

Shannon entropy (in nats) of the token distribution across experts, averaged across all MoE
layers. Measures how evenly spread routing is, on a logarithmic scale.

| Value | Interpretation |
|---|---|
| `log(num_experts)` ≈ 4.16 nats (64 experts) | Perfectly uniform routing |
| > 3.5 | Healthy |
| 2.0 – 3.5 | Noticeably skewed; investigate per-layer entropy |
| < 2.0 | Severe collapse; effective expert count below ~7 |

---

### `moe/expert_min_entropy`

**Formula:** same as `expert_mean_entropy` but takes the **minimum** across layers instead of the
mean.

This is the canary metric for per-layer collapse. In practice, collapse usually begins in one or
two layers while the others remain healthy; the mean entropy hides this. Monitoring the minimum
gives an early warning.

- **Rule of thumb:** if `expert_min_entropy` is more than 0.5 nats below `expert_mean_entropy`,
  at least one layer is collapsing while others are masking it in the mean.

---

### `moe/expert_e_eff_at_min`

**Formula:** `exp(expert_min_entropy)`

The effective number of experts in the worst (most collapsed) layer. Derived by treating the
token distribution as a probability distribution and computing its perplexity.

This converts the abstract entropy value into an interpretable count: if `expert_e_eff_at_min`
is 8 with 64 total experts, the worst layer is effectively routing all tokens to only 8 experts.

| Value (64 experts) | Interpretation |
|---|---|
| 64 | All experts equally used in the worst layer |
| 32 – 64 | Mild concentration |
| 8 – 32 | Significant collapse in at least one layer |
| < 8 | Severe collapse; most experts are dead in at least one layer |

---

### `moe/expert_dead_count`

**Formula:** `count(experts with zero tokens)` summed across all MoE layers.

Total number of expert-layer slots that received zero tokens in the entire log window. A value of
0 means every expert in every layer was used at least once. With 9 MoE layers and 64 experts,
the maximum possible value is 576.

- **Healthy:** 0 at the start of training; may increase slightly over time in long runs.
- **Warning:** any non-zero value that grows monotonically across training steps indicates
  structural dead experts that are unlikely to recover without intervention.

---

### Per-layer metrics (requires `moe_per_layer_logging: True`)

When per-layer logging is enabled, each aggregate metric above is also logged individually for
every MoE layer as `moe/{metric}_layer_{i}` (0-indexed). Additionally:

**`moe/tokens_per_expert_layer_{i}`** — a full W&B histogram of per-expert token counts for
layer `i`. This is the most detailed view available: the shape of the histogram tells you whether
imbalance is concentrated (a few dominant experts) or diffuse (many slightly-overloaded experts).
Compare across layers to identify which layers are driving the aggregate signals.

---

## Section 4 — Failure Mode Cheat Sheet

| Failure mode | First signal | Confirms with | Notes |
|---|---|---|---|
| **Router saturation** — logit scale grows unboundedly, routing becomes near-deterministic | `router_logit_logsumexp` rising beyond ~6 | `z_loss` also rising despite coeff | Can happen before any load change; usually caused by missing or weak z-loss |
| **Expert collapse** — a few experts capture almost all tokens | `router_mean_max_prob` rising above ~0.3 | `moe/expert_cv_pct` > 50%, `moe/expert_max_frac` > 0.10 | Often a downstream effect of saturation; aux loss coeff too low |
| **Dead experts** — some experts receive zero tokens and stop learning | `moe/expert_dead_count` > 0 and growing | `moe/expert_active_frac` < 1.0, per-layer histograms | Once dead, experts rarely recover; reduce aux loss coeff or reduce top-k |
| **Layer-specific collapse** — one layer collapses while others look healthy | `moe/expert_min_entropy` diverges from `moe/expert_mean_entropy` | Per-layer `expert_entropy_layer_{i}` and `tokens_per_expert_layer_{i}` histograms | Easy to miss if only watching aggregates; always monitor min entropy separately |
| **Confident-but-balanced routing** — router is highly certain yet load appears balanced | `router_mean_max_prob` high while `moe/expert_cv_pct` is still moderate | `router_logit_logsumexp` also high | Precursor to collapse; balance will break once one expert pulls slightly ahead |
| **Healthy steady-state** | `router_logit_logsumexp` stable in 4.5–6.0, `router_mean_max_prob` stable in 0.05–0.20 | CV% < 30%, dead count = 0, min entropy > 3.5 | All metrics should stabilise after warmup; continued drift in any metric warrants investigation |

---

## Quick Reference Table

| W&B key | Always logged | Healthy direction | Primary failure signal |
|---|---|---|---|
| `load_balancing_loss` | Only with aux_loss | Low and stable | Spike = imbalance event |
| `seq_load_balancing_loss` | Only with seq_aux_loss | Low and stable | Spike = imbalance event |
| `global_load_balancing_loss` | Only with global_aux_loss | Low and stable | Spike = imbalance event |
| `z_loss` | Only with z_loss_coeff | Low and stable | Growth = logit saturation |
| `router_logit_logsumexp` | Yes | Stable after warmup | Growth = saturation onset |
| `router_mean_max_prob` | Yes | Stable after warmup | Growth = sharpness / collapse onset |
| `moe/expert_cv_pct` | Yes | Low (< 30%) | High = load imbalance |
| `moe/expert_active_frac` | Yes | 1.0 | Decrease = dead experts |
| `moe/expert_max_frac` | Yes | Near 1/num_experts | High = hot-spot expert |
| `moe/expert_mean_entropy` | Yes | Near log(num_experts) | Decrease = load concentration |
| `moe/expert_min_entropy` | Yes | Near mean entropy | Gap from mean = layer-specific collapse |
| `moe/expert_e_eff_at_min` | Yes | Near num_experts | Low = effective collapse in one layer |
| `moe/expert_dead_count` | Yes | 0 | Any non-zero growing value = dead experts |
| `moe/tokens_per_expert_layer_{i}` | With per_layer_logging | Flat histogram | Peaked histogram = collapsed layer |
