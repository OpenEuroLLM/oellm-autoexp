# TensorWave read-only protocol

Prefer `$tw:cluster` and `$tw:catchup` for status and artifact discovery when those skills are available.
Otherwise use only read-only `squeue`, `scontrol show job`, `sacct`, `find`, `rg`, `ls`, `stat`, and bounded
`tail`. Do not submit, cancel, requeue, hold, release, modify, or delete anything. Return unresolved fields
in the portable run-context document instead of guessing.
