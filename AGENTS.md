# Slurm allocation policy

The user explicitly instructed on 2026-11-03:

> DO NEVER RESCHEDULE, UPDATE, REQUEUE OR CANCEL slurm jobs without my permission, we will loose all priority.

Preserve existing Slurm job IDs and submission age. Permission to submit additional jobs does not authorize changing existing ones. This includes automatic queued exclusion updates, pool replacement/cancellation, and requeue/restart actions that change Slurm jobs. Use read-only inspection and new submissions within the authorized scope; obtain explicit user permission for changes to existing Slurm jobs.
