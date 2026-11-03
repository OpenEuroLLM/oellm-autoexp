# Faster HF export: identical checkpoint and exported bytes

## Objective
Reduce the full export-and-validation latency, checking exported values one-to-one against the frozen baseline.

## Hypothesis
Repeated deserialization of 485 MB of DCP metadata and serial reads dominate much of the layer-by-layer export. Cache metadata for the pinned checkpoint, skip unnecessary flattening for flat CPU templates, and read different shard files concurrently. Preserve conversion arithmetic, FP8 quantization and safetensors writing.

## Setup
- Input: pinned main_mhc4 checkpoint 3700, identical config/tokenizer and FP8 settings.
- Baseline: experiment 006, snapshot autoexp_mhc_eval3700, run mn5-pool-p132.0233; retain all outputs.
- Candidate: separate snapshot and output root; opt-in HF_EXPORT_READER=fast, configurable read threads.
- Validation: small DCP round trips, controlled real-layer reader comparisons on one node, full streamed bitwise comparison of every exported tensor, and identical shape/scoring/paired-validation stages.
- Full-run wall times must include validation; record node placement and cache limitations explicitly.

## Run Log
- 2026-11-03: inspected the active PyTorch FileSystemReader and DefaultLoadPlanner. The default planner's tensor commit is a no-op, enabling independent shard-file reads into disjoint CPU tensor slices. Non-default planners and byte objects retain serial fallback.

## Results
Pending.

## Related
- Previous: ../006_mhc_v4_downstream_3700/log.md

- A/B scope: full candidate pipeline uses the same checkpoint, config, tokenizer, FP8 settings, container and one MN5 node as baseline. It writes `hf_exports_fast007`, leaving experiment 006 and its evaluation gates untouched. Baseline and candidate full pipelines run on equivalent but different spare nodes; full wall-time comparison is observational, not cache-controlled. The real-layer loader AB/BA benchmark runs on one node to control hardware and order effects. Full tensor comparison streams raw bytes including FP8 weights/scales and giant Engram tables, regardless of safetensors shard layout.
- 2026-11-03 10:59 CET: submitted `kernel_tests_fast_hf_export007_6cec00`, run `mn5-pool-p132.0234` on as05r5b07. Both unit tests passed. Six AB/BA real-layer comparisons passed bitwise: attention baseline 18.45–72.93 s / fast 0.345–0.403 s; CKDA 28.88–76.00 s / 1.668–3.078 s; MoE 107.83–162.68 s / 5.941–7.380 s (1,093 tensors). Benchmark startup metadata reads: 12.24 / 10.45 s. Global cache not flushed.
- 2026-11-03 11:10 CET: full pipeline did not start because `/usr/bin/time` is absent in the training container. Replaced it with a standard-library subprocess timer, retaining completed reader comparisons for the retry (`SKIP_READER_BENCHMARK=1`). No candidate export files were produced by this failed launch.
- 2026-11-03 11:10:40 CET: retry `kernel_tests_fast_hf_export007b_59727c`, run `mn5-pool-p132.0235` on as05r5b07. Conversion completed 11:27:07 (987 s from pool launch vs baseline 3,756 s); shape checks completed 11:27:39, all 1,949 tensors matched names/shapes. Reader metadata was read once (11.369 s), tensor reads totaled 749.259 s with 8 workers. Scoring then exposed a missing launch dependency: generic kernel-test config lacks hf_export_mn5's external `pylibs_hfx` path for accelerate. Add the same path and resume from successful conversion/check markers. Preserve the failed pipeline log and timer. Operational end-to-end comparison will include the retry delay, avoiding an inflated speed claim; subprocess timer on resume covers validation only.
