---
name: megatron-tensorboard
description: Extract and normalize Megatron-LM TensorBoard scalar events, especially MoE router and expert-viability metrics. Use when given event files or a TensorBoard directory and asked to inspect, export, compare, or prepare MoE training metrics for diagnosis.
---

# Megatron TensorBoard

Normalize scalar data before interpreting it.

## Workflow

1. Accept an event file or TensorBoard directory. Do not require a live TensorBoard server.
2. Run `scripts/extract_scalars.py` with an explicit tag or prefix allowlist.
3. Return its JSON artifact to `$megatron-moe-health` or the requesting workflow.
4. Record requested-but-absent tags in `missing_tags`; do not synthesize series.

## Rules

- Preserve `step`, `wall_time`, and `value` for every event.
- Merge event files by tag and sort by step/wall time; retain duplicates rather than averaging them.
- Use `--prefix moe/` for a first MoE pass, then request specific non-prefixed loss metrics as needed.
- Explain dependency failures clearly: extraction requires the Python `tensorboard` package.

## Resources

- `scripts/extract_scalars.py`
- `references/output-schema.md`
