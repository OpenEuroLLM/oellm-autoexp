---
name: megatron-moe-metrics
description: Explain, audit, or implement Megatron-LM MoE metric semantics and distributed reductions. Use when asked what a router or expert-viability metric means, whether a metric is correctly reduced across TP/CP/PP/DP/EP, where it is logged, or how to add a safe MoE diagnostic.
---

# Megatron MoE Metrics

Use this skill for metric semantics and implementation audits, not broad run diagnosis.

## Workflow

1. Identify the metric family: routing/load, parameter/gradient, routed contribution, optimizer
   update, or paired causal validation.
2. Read the relevant section in `references/metric-catalog.md`, then
   `references/distributed-semantics.md` for any distributed question.
3. For source audits, verify the configured logging gate, accumulation point, process groups,
   numerator/denominator reduction, reset point, and output key.
4. State whether a metric is currently emitted, planned, or unavailable in the supplied run.

## Rules

- Do not call routing/load health proof of functional usefulness.
- Preserve sum/count pairs through reduction; reduce before taking ratios or RMS.
- For optimizer-specific estimates, omit unsupported paths rather than inventing a proxy.
- Treat compatibility mappings in `references/compatibility.md` as revision-specific.

## Resources

- `references/metric-catalog.md`
- `references/distributed-semantics.md`
- `references/compatibility.md`
