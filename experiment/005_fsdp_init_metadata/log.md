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
- [Focused unit output](results/unit.log): 6 tests per rank, 4 ranks, passed.
- [Exact-source NCCL benchmark](results/benchmark.json): 34.36x metadata-finalization speedup on 16 GPUs; 192 collectives reduced to 4.
- [Four-node restore and continuation](results/comparison.json): matching FP32 parameter/optimizer-state diagnostics on all 16 ranks; steps 21–30 completed and final checkpoint saved.
- [Unchanged-code control](results/trajectory_control.json): restore diagnostics match, but subsequent trajectories vary in both arms. Statistical trajectory equivalence is not established.
- [Source provenance](results/provenance.json); exact driver commands in scripts/.

Full 128-node startup improvement has not been measured. The implementation is committed as Megatron `a1f185341` and remains isolated from the live training snapshot.

## Interpretation
No production deployment until tests pass. The helper requires identical tensor order across ranks, as the existing metadata collectives do.

- 2026-11-03 01:21 CET: focused unit job `mn5-pool-o134e.0017` completed, 6 tests passed per rank (4 ranks, CPU/gloo within the training container). Tests cover bounds and missing coverage errors, empty shards, replicated and multi-shard meshes, metadata reuse, and group-level collective counts. See `results/unit.log`. Candidate Megatron commit `a1f185341`; hashes in `results/provenance.json`.
- 2026-11-03 01:21 CET: baseline reference `fsdp_meta005_ref_2c579a` enqueued in o134e using `autoexp_rowmeta2`. Fresh output suite, 30 total iterations, rolling save at 20.

- 2026-11-03: added a separate 4-node/16-GPU NCCL benchmark over 64 uneven-shard tensors on full-DP and expert-like inner meshes. It compares old validate+attach (128 metadata gathers, 64 boundary reductions per rank) with batched finalization (2+2), checks identical metadata, and records slowest-rank elapsed time.

- 2026-11-03 01:24 CET: four-node NCCL benchmark `mn5-pool-o134e.0019` produced identical metadata. Slowest-rank time: old validate+attach 0.188033 s vs batch 0.003759 s (50.02x on 64 tensors). Per-rank collective counts: 128 gathers + 64 boundary reductions -> 2 + 2. This is finalization-path timing, not total startup speedup. See `results/benchmark.json`.

- 2026-11-03 01:34–01:36 CET: candidate restored rolling iteration 20. All 15,345 summed FP32 parameter/optimizer-state entries and per-rank weighted checksums match across 16 ranks (zero differences). First resumed loss matches reference. Later trajectories show differences (through step 27: max loss delta 0.08456; max grad-norm delta 20.0), so an unchanged-code continuation control is added to measure baseline variability rather than declaring trajectory equivalence.

- 2026-11-03 01:38 CET: candidate continuation `mn5-pool-o134e.0021` completed with rc `0 completed`; resumed every step 21–30, then successfully saved persistent step 30. The strict driver reported `FSDP_META005_VALIDATION_PASSED`. Restore diagnostic checks match exactly across 16 ranks. Later trajectory diagnostics: max loss delta 0.08825 (RMS 0.03962); max grad-norm delta 20.0 (RMS 6.521). Control continuation `mn5-pool-o134e.0022` uses unmodified code to assess this variability. See `results/comparison.json` and `results/comparison.log`.

- 2026-11-03: tightened the timing benchmark to import the actual unmodified baseline validator from `autoexp_rowmeta2`, rather than running the old per-tensor call pattern through the candidate's single-tensor compatibility wrapper. The prior timing includes the wrapper's extra indexing kernels; collective counts are unchanged. Repeating the benchmark gives a fairer source-to-source timing comparison.

- 2026-11-03 01:41 CET: exact-source NCCL timing rerun: 0.147678 s baseline vs 0.004298 s candidate, 34.36x for the metadata-finalization microbenchmark, identical metadata and 128+64 -> 2+2 collectives. This supersedes the 50x wrapper-based timing; the earlier result is retained as `results/benchmark_common_validator.json`.
- 2026-11-03 01:44 CET: unchanged-code control also restored matching state summaries on all 16 ranks and completed steps 21–30 with a successful step-30 save. Control vs ref loss max/RMS delta: 0.034088/0.017107; candidate vs ref: 0.088250/0.039622. Control grad-norm max/RMS: 4.771/2.032; candidate: 20.0/6.521. Both trajectories differ; one control does not establish statistical equivalence, and candidate variability was larger in this sample. **Validated: checkpoint state diagnostics, metadata equivalence, retained validation, continued training/save, collective reduction. Not established: bitwise or statistical trajectory equivalence, or full 128-node startup improvement.** Candidate is isolated; live training still uses its previous snapshot.
- 2026-11-03 01:42 CET: user requested retiring the one-day allocation. LDM production is stable near 29.36 s/step after excluding as07r2b24. Set o134e `min_hosts=0 keep_pending=0 max_own_hosts=0`, drained host001, and cancelled its keepalive. Existing tests (including another agent's amb_rs7_cont) finish before release.

- 2026-11-03 01:47:36 CET: o134e host001 exited after `drain requested: pool_ctl drain`. All running tests completed, including the other agent's `amb_rs7_cont` (step-30 save successful, run rc `0 completed`). No replacement allocation will be submitted (`min_hosts=0`, `keep_pending=0`, `max_own_hosts=0`). LDM remains on 48291324 with as07r2b24 excluded.
