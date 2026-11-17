# Slurm allocation policy

The user explicitly instructed on 2026-11-03:

> DO NEVER RESCHEDULE, UPDATE, REQUEUE OR CANCEL slurm jobs without my permission, we will loose all priority.

Preserve existing Slurm job IDs and submission age. Permission to submit additional jobs does not authorize changing existing ones. This includes automatic queued exclusion updates, pool replacement/cancellation, and requeue/restart actions that change Slurm jobs. Use read-only inspection and new submissions within the authorized scope; obtain explicit user permission for changes to existing Slurm jobs.

# Commit and snapshot discipline

The user instructed on 2026-11-17:

> Commits should happen targeted (relevant files) and frequently.

- One concern per commit, and only the files of that concern. Split unrelated edits.
- Commit submodules before the superproject gitlink that records them.
- Never commit `.pi/` or coverage artifacts (`.coverage`).
- Do not edit a synced snapshot (`$R/autoexp_<name>`) in place. Re-run `sync_to_marenostrum.sh <name>` so `SYNC_STAMP` matches its files, and read that stamp before trusting a run's provenance.
- A run reads the code of the snapshot it was enqueued from, not the live checkout. Sync before enqueueing anything that depends on new code.
