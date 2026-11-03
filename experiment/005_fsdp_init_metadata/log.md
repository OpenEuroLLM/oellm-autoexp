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
