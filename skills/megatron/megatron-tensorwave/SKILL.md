---
name: megatron-tensorwave
description: Read-only discovery and inspection of Megatron-LM training runs on TensorWave. Use when asked to locate a TensorWave run, inspect SLURM status, identify TensorBoard/log artifacts, or prepare a portable run context without submitting, cancelling, or modifying jobs.
---

# Megatron TensorWave

Resolve TensorWave artifacts into the portable `$megatron-run-context` contract.

## Workflow

1. Read `references/read-only-protocol.md` before issuing remote commands.
2. Prefer existing `$tw:cluster` or `$tw:catchup` capabilities when available; otherwise use only
   read-only scheduler and filesystem inspection.
3. Identify run root, resolved config, TensorBoard event directory, training logs, and scheduler ID.
4. Emit a context document; hand it to `$megatron-moe-health` for diagnosis.

## Rules

- Never submit, cancel, requeue, or alter a TensorWave job in this skill.
- Do not hard-code hostnames, scratch roots, partitions, or usernames.
- Report unavailable `tw` capabilities or connectivity as unresolved context fields.
