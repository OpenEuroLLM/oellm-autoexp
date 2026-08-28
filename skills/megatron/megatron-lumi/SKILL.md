---
name: megatron-lumi
description: Read-only discovery and inspection of Megatron-LM training runs on LUMI. Use when asked to locate a LUMI run, inspect SLURM status, identify TensorBoard/log artifacts, or prepare a portable run context without submitting, cancelling, or modifying jobs.
---

# Megatron LUMI

Resolve LUMI artifacts into the portable `$megatron-run-context` contract.

## Workflow

1. Read `references/read-only-protocol.md` before issuing remote commands.
2. Use read-only scheduler and filesystem commands only: `squeue`, `scontrol show job`, `sacct`,
   `find`, `rg`, and `tail`.
3. Identify run root, resolved config, TensorBoard event directory, training logs, and scheduler ID.
4. Emit a context document; hand it to `$megatron-moe-health` for diagnosis.

## Rules

- Never submit, cancel, requeue, or alter a LUMI job in this skill.
- Do not assume a partition, filesystem root, launcher, or username; discover or ask.
- Report permission and connectivity failures as unresolved context fields.
