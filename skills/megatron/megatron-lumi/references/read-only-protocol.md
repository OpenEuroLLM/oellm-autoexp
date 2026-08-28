# LUMI read-only protocol

Use `squeue`, `scontrol show job`, `sacct`, `find`, `rg`, `ls`, `stat`, and bounded `tail` only.
Discover account, partition, filesystem root, and job identifiers from user-provided context or command
output. Do not submit, cancel, requeue, hold, release, modify, or delete anything. Return unresolved
fields in the portable run-context document instead of guessing.
