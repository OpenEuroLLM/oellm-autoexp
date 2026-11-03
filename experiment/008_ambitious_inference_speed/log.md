# Ambitious inference: HF and vLLM

## Objective
Reduce inference/evaluation latency for the ambitious MHC/LDM models, keeping checkpoint 3700 and fixed prompts as a numerical reference. Measure startup, prefill and decode separately.

## Hypotheses
- HF's per-expert Python loop, CUDA nonzero synchronization and per-forward FP8 dequantization cost far more than the active parameter count suggests. Grouped GPU matmuls and one-time dequantization can remove this overhead.
- vLLM already uses TP4+EP and mmap-backed Engram. Pure EP4 with TP1 and DP4 can remove TP collectives/replicated lookup work within a node.
- Sparse Engram reads from GPFS, CPU sorting and host/device synchronization need measurement. Compare mmap policies and node-local storage where actual disk capacity permits.

## Setup
- Frozen baseline: autoexp_mhc_eval3700, checkpoint ambitious_v4_2ph_fp8_3700.
- Export changes from experiment 007 are verified bitwise; preserve old inference implementations for comparison.
- First test: one acc_debug four-GPU node, two hours, independent of live training allocations.
- vLLM AGENTS.md read: use uv and a node-local .venv/bin/python for Python work; no upstream PR requested.
- Original eval shards were externally canceled at 12:43; p132 allocation released. Completed arc_challenge_25 / winogrande_0 rows have done markers. The runner resumes completed rows, but unfinished rows restart (no response cache enabled).
- Commit drivers before submission; record allocation, runtime versions, source revisions, exact prompts and numerical differences.

## Run Log
- 2026-11-03: inspection found HF loops over torch.unique(idx).tolist(), torch.where per expert and dequantizes per expert on every forward. vLLM uses fused experts but Engram hashes/state handling still loop over sequences and host lookups synchronize GPU indices. Disk mmap is already default.

## Results
Pending.

## Related
- ../007_fast_hf_export/log.md
