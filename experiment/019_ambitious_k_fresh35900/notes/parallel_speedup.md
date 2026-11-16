# Parallel checkpoint surgery and load plan

User steering, 2026-11-16: the full checkpoint is about 1.1 TB; execution
should be parallel, and a slow path should lead to a planned faster version.

## Current attempt

Training is distributed over 128 nodes / 512 GPUs. Surgery is a separate,
single-node CPU step using `add_ckda_branch.py`; it reads only the selected
source attention weights and target input norms. Unchanged `.distcp` files
are hard-linked, not copied. At about 3.5 minutes, the derive process had read
24.5 GB (`/proc/PID/io` rchar) and used about 5 GB RSS. This is logical read
traffic, not a measurement of physical storage traffic or 1.1-TB copying.

## Measure before changing

Record elapsed time for derive/read, delta write, read-back, independent
verify, training metadata/plan/data loading, and final synchronous save.
Compute requested tensor bytes from metadata and compare with logical read
traffic to expose read amplification. Keep source metadata hash and resulting
tensor hashes as correctness checks.

## Faster surgery design

1. Use six workers, one target/source layer pair per worker, preferably one
   worker per node if single-node read bandwidth limits throughput.
2. Each worker reads its source Q/gate/output/Q-norm/input norms and writes
   only its delta tensors and optimizer states with unique file names.
3. A coordinator reads base metadata once, checks disjoint keys, merges the
   six deltas, and hard-links the unchanged source shards once.
4. Verify worker hashes, tensor shapes/dtypes, gamma zero, finite copied
   weights, and source metadata unchanged. Publish the tracker atomically
   only after every check passes.
5. Test on a small checkpoint with both equal and changed load world sizes;
   compare all resulting tensors with the sequential implementation.

Do not create six full serialized copies of the original checkpoint.

## Distributed training I/O

Inspect the actual snapshot's FSDP load path and timing instrumentation.
Keep tensor reads/writes distributed; look for coordinator plan gathering,
repeated metadata parsing, and replicated optimizer-state reads as possible
serialization points. Rank-local planning is a candidate only when the
snapshot supports it and correctness has been checked. Apply any implementation
to a new attempt/snapshot rather than editing a running process's source files.
