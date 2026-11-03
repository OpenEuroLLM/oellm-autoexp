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
