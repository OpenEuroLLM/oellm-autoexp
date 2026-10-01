---
name: megatron-run-context
description: Build a portable, read-only context document for a Megatron-LM training run from a run root, config, TensorBoard files, logs, and optional scheduler metadata. Use before cross-cluster MoE analysis or whenever run artifacts need to be located and normalized.
---

# Megatron Run Context

Create a portable artifact manifest; do not infer health here.

## Workflow

1. Run `scripts/build_run_context.py --run-root PATH` or pass explicit artifact paths.
2. Set `--cluster` to `local`, `lumi`, or `tensorwave`; include an optional scheduler ID only if
   supplied or discovered by a read-only adapter.
3. Validate the resulting document against `references/context-schema.md`.
4. Pass the context JSON to the requested analysis skill.

## Rules

- Mark absent files as unavailable, never as empty measurements.
- Record absolute artifact paths and the discovery time.
- Do not connect to a cluster, submit a job, or mutate a run.

## Resources

- `scripts/build_run_context.py`
- `references/context-schema.md`
