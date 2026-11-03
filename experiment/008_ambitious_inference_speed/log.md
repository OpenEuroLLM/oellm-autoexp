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
- 2026-11-03 15:32: debug job 48455101 started on as03r3b15 (4 x H100 64 GiB; /tmp is a 252 GiB tmpfs, not local disk). Runtime: torch 2.13.0+cu130, vLLM 0.32 fork. UV creates a node-local virtual environment and installs only from the existing offline wheel store.
- HF real MoE layer 1: at 512 tokens reference 76.47 ms, cached dequantization 55.93 ms (bitwise equal), grouped matmuls 2.554 ms (RMS output error 2.124e-5; maximum 9.766e-4). At 128 tokens: 60.83 -> 1.970 ms; at one token 1.795 -> 0.663 ms. One-time dequantization 23.58 ms for this layer. Full-model numerical validation remains required.
- Fused 4-stream Sinkhorn: all 16 token-count/logit-scale cases passed maximum error <3e-6; observed worst <4.5e-7. At 4096 tokens eager ~1.75 ms vs fused ~0.02 ms. This is a component measurement, not a full-model speed claim. Microbenchmark ran as a small overlapping step while the full engine was loading.
- 2026-11-03 15:53: pure EP4/TP1/DP4 successfully initialized the real 3700 model. Startup 995.39 s. Same 32 x 256-token prompts, 8 generated tokens each: first-use 161.62 s (50.69 input tok/s), repeat 6.096 s (1343.88 input tok/s). First-use includes JIT compilation and mmap faults; do not attribute it entirely to storage. TP4+EP comparison follows on the same node.
- vLLM changes are opt-in: local expert loading avoids dequantizing non-owned experts; optional TP-leader Engram lookup, mmap readahead advice and offload-root override; fused mHC normalization. Submodule lint/type/pre-commit checks passed after bringing the existing port to the required formatting/type checks. vLLM commit 5bd602f81c. Existing baseline snapshot remains unchanged.
