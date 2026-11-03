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

Complete. Same checkpoint 3700, config, tokenizer, expert quantization and validation inputs.

| Variant | Export incl. launch | Export + checks + ordinary/paired validation |
|---|---:|---:|
| Frozen baseline | 62.61 min | 119.21 min |
| Cached/parallel reader, sequential checks | 16.46 min | 75.40 min (includes dependency retry delay) |
| v2 reader + checks on separate GPU pairs | 10.35 min | 43.28 min |

- v2 operational end-to-end speedup: **2.754x**, reserving the same one four-GPU node per run. Its directly timed pipeline took 42.43 min; operational total includes container launch and capacity preflight.
- Both candidates: **1,949 tensors / 248,189,032,600 bytes, bit-for-bit equal** to baseline, including FP8 expert bytes/scales, full Engram tables and all other model tensors. Config/tokenizer/model source files also equal.
- Both candidates: ordinary score fields and every paired sample loss **exactly equal** to baseline. Mean NLL 1.5175073530138918; paired loss 1.487030614801938; no nonfinite documents/samples.
- v2 ordinary scoring took 760.52 s and ran concurrently with paired validation. Validation wall time fell from 56.13 to 32.60 min.
- Two reader unit tests passed; all six same-node AB/BA real-layer reads matched bitwise; reusable config resolves to one node/four GPUs with the baseline accelerate path.
- Full-run timings are observational: equivalent different nodes, shared GPFS, no cache flushing. The reader-only repeat took 16.46 min versus v2's 10.35 min despite the same reader settings, showing I/O/cache variation. The controlled AB/BA results separately establish the reader improvement. A brief v2 profiler attempt timed out without usable output; no conclusions rely on it, and its overhead remains included.
- One-time exhaustive byte checks took 13.87 min (v2) / 13.44 min (sequential candidate), excluded from the production pipeline comparison because baseline does not perform a reference-to-reference comparison.

Artifacts: [parallel summary](results/parallel_summary.json), [sequential summary](results/sequential_summary.json), [reader benchmark](results/reader_benchmark.json), [source/input provenance](results/provenance.json), [unit tests](results/unit_test_result.txt), [config validation](results/config_validation.json), [commands](notes/commands.md).

Reusable entry point: `config/experiments/korbi/production/hf_export_v2_mn5.yaml`, defaults to ambitious_v4 + FP8, fast reader with 8 workers and parallel validation. Existing legacy entry point remains unchanged.

## Related
- Previous: ../006_mhc_v4_downstream_3700/log.md

- A/B scope: full candidate pipeline uses the same checkpoint, config, tokenizer, FP8 settings, container and one MN5 node as baseline. It writes `hf_exports_fast007`, leaving experiment 006 and its evaluation gates untouched. Baseline and candidate full pipelines run on equivalent but different spare nodes; full wall-time comparison is observational, not cache-controlled. The real-layer loader AB/BA benchmark runs on one node to control hardware and order effects. Full tensor comparison streams raw bytes including FP8 weights/scales and giant Engram tables, regardless of safetensors shard layout.
- 2026-11-03 10:59 CET: submitted `kernel_tests_fast_hf_export007_6cec00`, run `mn5-pool-p132.0234` on as05r5b07. Both unit tests passed. Six AB/BA real-layer comparisons passed bitwise: attention baseline 18.45–72.93 s / fast 0.345–0.403 s; CKDA 28.88–76.00 s / 1.668–3.078 s; MoE 107.83–162.68 s / 5.941–7.380 s (1,093 tensors). Benchmark startup metadata reads: 12.24 / 10.45 s. Global cache not flushed.
- 2026-11-03 11:10 CET: full pipeline did not start because `/usr/bin/time` is absent in the training container. Replaced it with a standard-library subprocess timer, retaining completed reader comparisons for the retry (`SKIP_READER_BENCHMARK=1`). No candidate export files were produced by this failed launch.
- 2026-11-03 11:10:40 CET: retry `kernel_tests_fast_hf_export007b_59727c`, run `mn5-pool-p132.0235` on as05r5b07. Conversion completed 11:27:07 (987 s from pool launch vs baseline 3,756 s); shape checks completed 11:27:39, all 1,949 tensors matched names/shapes. Reader metadata was read once (11.369 s), tensor reads totaled 749.259 s with 8 workers. Scoring then exposed a missing launch dependency: generic kernel-test config lacks hf_export_mn5's external `pylibs_hfx` path for accelerate. Add the same path and resume from successful conversion/check markers. Preserve the failed pipeline log and timer. Operational end-to-end comparison will include the retry delay, avoiding an inflated speed claim; subprocess timer on resume covers validation only.
- 2026-11-03 11:31:45 CET: dependency-corrected resume `kernel_tests_fast_hf_export007c_ab35d9`, run `mn5-pool-p132.0236`, again on as05r5b07. Imports and repeated unit tests passed. Existing export/check markers retained; only model-code refresh, remaining validation and byte comparison run. Baseline paired validation continued normally (sample 24/64 at 11:32).
- 2026-11-03 11:37 CET: validation now dominates elapsed time. Prepare a distinct v2 full export on the third spare node, with a new output root: identical conversion and validation inputs, but ordinary score on GPUs 0/1 concurrently with paired score on GPUs 2/3. Current mapping uses only 81.4 GiB of layer weights, so each validation fits a two-GPU pair; test capacity before conversion. This uses the same four-GPU-node resource budget per full run. Keep the sequential candidate for the original controlled comparison. Capture per-check timers and compare every exported byte for v2 as well. Baseline export/evaluation pipeline remains untouched.

- 2026-11-03 12:40 CET: both candidate jobs completed rc=0 after exhaustive byte verification. All exported bytes, ordinary score fields and paired per-sample losses match baseline exactly. Full v2 benchmark and reusable configuration complete; downstream evaluation continues independently.
