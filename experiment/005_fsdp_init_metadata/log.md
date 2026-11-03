# Batched FSDP initialization metadata

## Objective
Reduce model setup latency without changing weights, optimizer state, or validation.

## Hypothesis
FSDP parameter initialization gathers uneven-shard metadata twice and reduces boundary flags per parameter. Gathering once per mesh/group and batching boundary reductions removes thousands of sequential collectives.

## Setup
- Baseline: Megatron `5ecd322ea` (rowmeta optimizer batching), remote snapshot `autoexp_rowmeta2`.
- Candidate: isolated `autoexp_fsdpmeta005`, never synced over a live training snapshot.
- Small tests: uneven/empty shards, replicated and multidimensional meshes, rejected invalid metadata, collective-count regression.
- Integration: distinct suite `fsdp_meta005`; baseline 4-node ref saves rolling iteration 20 and trains to 30; candidate 4-node cont must load iteration 20 and train 21–30. Separate outputs from amb_rs6.
- Pool: retained `mn5_pool_o134e`; packing alongside other work on disjoint nodes.

## Run Log
- 2026-11-03: inspected live stack and source; found metadata gather in validation, boundary all-reduce, then another metadata gather for attachment. Implementation preserves TP construction checks and batches only uneven-shard finalization.

## Results
Pending execution; exact commands in scripts/.

## Interpretation
No production deployment until tests pass. The helper requires identical tensor order across ranks, as the existing metadata collectives do.

- 2026-11-03 01:21 CET: focused unit job `mn5-pool-o134e.0017` completed, 6 tests passed per rank (4 ranks, CPU/gloo within the training container). Tests cover bounds and missing coverage errors, empty shards, replicated and multi-shard meshes, metadata reuse, and group-level collective counts. See `results/unit.log`. Candidate Megatron commit `a1f185341`; hashes in `results/provenance.json`.
- 2026-11-03 01:21 CET: baseline reference `fsdp_meta005_ref_2c579a` enqueued in o134e using `autoexp_rowmeta2`. Fresh output suite, 30 total iterations, rolling save at 20.

- 2026-11-03: added a separate 4-node/16-GPU NCCL benchmark over 64 uneven-shard tensors on full-DP and expert-like inner meshes. It compares old validate+attach (128 metadata gathers, 64 boundary reductions per rank) with batched finalization (2+2), checks identical metadata, and records slowest-rank elapsed time.
