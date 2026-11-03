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
- Baseline TP4+EP: startup 284.49 s, first batch 13.41 s, repeat 9.795 s (836.35 input tok/s). EP4 repeat was 1.61x faster, but the original probe's token budget was 512 per DP rank (2048 aggregate in EP4, 512 in TP4). Do not attribute the full difference to topology. Updated comparisons use a matched 2048 aggregate token budget and include a reference-kernel TP4 batching control.
- 2026-11-03 16:10: optimized debug job 48456575 started on as01r4b32. HF trained-checkpoint reference warm prefill 92.035 s vs new 6.495 s on the same 32 prompts (14.17x); generation 4.82x; 32/32 generated tokens match. Mean NLL change -0.0007672, mean absolute token logprob difference 0.02050, maximum 0.43247. Numerical differences remain and require task-level checking; do not label this bit-exact. Load times 147.41 / 117.26 s. Grouped experts now retain ascending-expert BF16 summation order; remaining differences include grouped GEMM and fused Sinkhorn rounding.
- Durable cache interruption/reopen test passed using lm-eval's actual CachingLM: three requests persisted, only five of eight recomputed, outputs equal uninterrupted reference. Added optional finite request batches and versioned per-rank SQLite caches.
- Prepared a true TP1/DP4/EP4 lm-eval integration using the harness's distributed request splitting and CPU Gloo aggregation, alongside vLLM's independent DP engine group. The smoke test uses 17 examples (uneven across ranks), MC plus generation. Reuse job 48456575 by appending this independent test stage after its existing benchmark stages; model/weight implementations of those benchmarks remain unchanged.
- Quality allocation 48458754 on as02r5b10: HF grouped-only ablation and 17-example reference-mHC task comparison. Append an independent final stage (no model implementation changes): production wrapper/done-row resume validation, then matched 8192-token aggregate budget tests for EP4 and TP4. Earlier comparisons used 2048. Production defaults remain at the measured 2048 setting until these finish.
- Matched 2048-token budget results (same optimized node): reference-mHC TP4 warm mixed 5.575 s; fused-mHC TP4 4.557 s; fused-mHC EP4 2.856 s. Warm prefill + one output: 2.572 / 1.496 / 2.573 s respectively. Thus EP wins this mixed decode workload, while TP4 wins prefill alone; topology choice is workload-dependent. Do not extrapolate mixed-batch speedups to every FLAG task.
- HF grouped-only / reference-mHC ablation: warm prefill 11.915 s (7.72x), generation 2.43x, all 32 generated tokens match. Mean NLL delta -0.0005493, mean absolute token-logprob difference 0.01018, maximum 0.41471. Grouped GEMM alone is not bitwise equivalent. Full fast HF is 14.17x with 4.82x generation speedup.
- Uneven distributed lm-eval smoke passed: exactly 17 unique documents each for HellaSwag and GSM8K, with four per-rank SQLite DBs. GSM8K has 34 sample records because it logs two filters for each document. Fast: HellaSwag acc 7/17, acc_norm 10/17; GSM8K exact match 10/17. The reference task run is separate, retaining these fast artifacts.
- Production config dry-run passed: `scripts/run_autoexp.py --dry-run --dry-run-root /tmp/infer008_configcheck --yes --no-monitor --config-name experiments/korbi/production/flag_eval_fast_mn5 aux.export=ambitious_v4_2ph_fp8_3700 'aux.shards=[0]' aux.num_shards=1 aux.limit=17 'aux.flag_names="vllm/hellaswag/0 vllm/gsm8k/4"'`. Exactly one node/shard rendered, with TP1/DP4, EP enabled and bounded request cache. Shell syntax and Python compilation checks passed.
- 2026-11-03 17:06: reference-mHC task comparison completed successfully (all four evaluator ranks rc0): HellaSwag raw acc 8/17 versus fast 7/17; normalized acc 10/17 in both. GSM8K exact match 10/17 in both filters and both implementations. Saved prompt hashes match for every document. Task runtimes reference/fast: HellaSwag 14 / 8 s, GSM8K 92 / 46 s (different nodes; small smoke sample, not a controlled full-suite speed claim). Reference startup 506–507 s; fast smoke startup 108 s on the already warmed prior node. Cold/startup cost is variable and must be reported separately.
- The actual production wrapper/runtime successfully installed from offline wheels and resumed the completed smoke output: all four frontends reported 2 rows / 0 to run, then exited successfully without initializing another engine. Larger token-budget probes now running as final stages.
