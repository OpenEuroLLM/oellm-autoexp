---
name: megatron-moe-health
description: Diagnose a Megatron-LM MoE training run from TensorBoard scalars, resolved configs, and logs. Use when asked whether experts are healthy, starved, collapsed, weakly learning, or functionally silent; when comparing MoE layers; or when producing an evidence-backed run-health brief. Keep analysis read-only unless the user explicitly requests an action.
---

# Megatron MoE Health

Build an evidence-backed health brief, not a universal health score.

## Workflow

1. Obtain a normalized run context. Use `$megatron-run-context` if the user supplied a run root
   or artifacts rather than an existing context file.
2. Extract scalar evidence with `$megatron-tensorboard` when a normalized scalar artifact is not
   already available.
3. Read `references/diagnosis-matrix.md` and only the relevant sections of
   `references/metric-catalog.md`.
4. Classify every observed MoE layer independently as:
   `healthy`, `imbalanced`, `starved`, `static`, `collapsed`, `weak_learning`,
   `small_contribution`, `functionally_silent`, or `insufficient_evidence`.
5. Write a Markdown brief and a machine-readable findings document using the schema in
   `references/finding-schema.md`.

## Rules

- Treat missing metrics as missing evidence. Never substitute zero or infer a healthy state.
- Separate selected routing, dispatched routing, parameter movement, gradient signal, routed
  contribution, and paired-mask causal evidence.
- State alternate explanations before calling a layer functionally silent. A near-zero routed
  ratio alone is not causal proof.
- Do not modify a run, submit jobs, or change code. Hand off requested cluster inspection to
  `$megatron-lumi` or `$megatron-tensorwave`.
- Cite metric names, layer, time window, and values for every nontrivial conclusion.

## Resources

- `scripts/render_health_summary.py`: turn normalized scalar JSON into a first-pass report.
- `references/metric-catalog.md`: names, availability, and semantics for this fork.
- `references/diagnosis-matrix.md`: evidence combinations and next checks.
- `references/finding-schema.md`: output contract.
