# Reproducing and using the fast paths

All speed comparisons use the same checkpoint `ambitious_v4_2ph_fp8_3700` and the fixed prompts saved in `results/ep4/prompts.json`. H100 64 GiB x4 per node. This model has ~3B active parameters but ~248 GB of exported tensors; active parameter count alone does not predict loading time or host-table traffic.

## HF

Refresh an independent export's model code using the existing `refresh_export` workflow; preserve the original checkpoint as reference. Enable:

```
OELLM_HF_MOE=grouped OELLM_HF_ENGRAM_LOOKUP=sorted OELLM_MHC_IMPL=triton
```

Use `score_on_flag.layer_device_map` and `load_engram_tables_cpu` so that expert FP8 weights are expanded once on CPU before device placement. Grouped experts require BF16 storage and ~3 GPUs for this checkpoint. The old FP8 reference fits on fewer GPUs. Keep `OELLM_MHC_IMPL=reference` for the reference-normalization variant. Defaults remain reference. These are inference-only optimizations.

The export-v2 concurrent validation tests reserve only two GPUs per process. Keep their default reference HF path; do not enable grouped experts for those concurrent two-GPU checks.

## vLLM / lm-eval

Production entry point:

```bash
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --yes --no-monitor \
  --config-name experiments/korbi/production/flag_eval_fast_mn5 \
  aux.export=ambitious_v4_2ph_fp8_3700
```

This submits four independent one-node shards using the inherited sweep. To use a running pool, add the existing `--pool <pool-directory>` argument; to test on one debug node, set `aux.dev_queue=true 'aux.shards=[0]' aux.num_shards=1 aux.limit=17`. This document is an invocation recipe, not a record that a full suite was submitted.

Fast config: TP1 + DP4 + EP4, aggregate max 256 sequences, aggregate 32768 scheduled tokens, finite request batches, per-rank durable response cache, local-expert-only loading, RAM-prefetched mmap tables, batched Engram decode, fused mHC. TP4 remains available with `aux.tp=4 aux.dp=1`.

The runner starts real distributed vLLM engines; it does not use lm-eval's Ray multi-model data-parallel mode. CPU Gloo combines uneven evaluator shards. Model traffic and EP are confined to the node. No training allocation was preempted by these tests.

Engram `mmap` already offloads tables to the export's storage. `ram` reads the same mappings once; the OS page cache is shared across ranks, without a private pinned table copy per rank. `OELLM_ENGRAM_MODEL_DIR` can point to identical shards on actual local disk. MN5 `/tmp` is tmpfs: staging there uses RAM, so no local-NVMe speed claim is made. A different offload root must contain matching table tensor values, shapes and dtypes.

A completed task has a done marker. Interrupted tasks resume saved responses from SQLite when the export/runtime/batch identity matches. Responses in the current unfinished request batch may be recomputed. Old runs without a response cache can only resume completed tasks. Changing batch layout or implementation intentionally selects a fresh cache. Use a new output directory for numerical comparisons; done markers intentionally skip completed tasks.

## Drivers and provenance

- `run_debug.sh`: frozen baseline, allocation 48455101, as03r3b15.
- `run_optimized.sh`: HF and matched-budget vLLM tests, allocation 48456575, as01r4b32; independent `run_eval_smoke.sh` appended after benchmark stages.
- `run_quality_check.sh`: HF grouped-only and reference-mHC lm-eval; allocation 48458754, as02r5b10. Independent `run_final_checks.sh` appended after initial submission, without changing the model implementation used by earlier stages.
- Baseline snapshot: `/gpfs/projects/ehpc533/users/korbip/hybrid_v019/autoexp_infer008`.
- Optimized snapshot: same parent, `autoexp_infer008_fast`.
- Optimized vLLM source: submodule commit `5bd602f81c`; offline wheel 0.32.0 native libraries, fork Python overlay, Torch 2.13.0+cu130.
- Remote logs: `/gpfs/scratch/ehpc533/hybrid_nucleus/kernel_tests/{infer008_debug,infer008_fast,infer008_quality}/slurm-<jobid>.log`.
- Full lm-eval artifacts: `/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals_infer008_{smoke,reference}/ambitious_v4_2ph_fp8_3700_limit17`.
- `summarize.py` regenerates `results/comparison.json` from the downloaded JSON artifacts with the standard library. The 17-example tests are smoke checks, not sufficient evidence of benchmark-level quality equivalence.

## Current layout and batching recommendation

Use TP1/DP4/EP4 with a 256-sequence upper cap and 32768-token aggregate budget for this checkpoint. All 64/128/256 caps passed the saturated 1024-request tests. Pure EP outperformed TP4 for both scoring and generation under this queued workload; the older 32-prompt probe was too small to establish a general layout recommendation. Do not extrapolate its TP4 prefill advantage to the saturated case.

At budget 8192, original EP4 generation throughput was 168 / 262 / 382 / 494 output tokens/s at concurrency 32 / 64 / 128 / 256. The same-loaded-model Engram optimization improved batch-256 generation from 478 to 1059 output tokens/s at budget 32768, and its decode-window rate from 533 to 1427 tokens/s. All figures are aggregate across the four-GPU node. They exclude model loading; end-to-end request throughput includes prefill, whereas the decode-window metric excludes first-token latency.

Single-request decode remains ~6.7 tokens/s, far below the user's ~168-token/s target. Batched Engram fixes the dominant batch bottleneck, while a delayed single-request profile distributes worker wall time across MoE, CKDA and MHC. Reaching the single-request target needs further GPU/launch/collective profiling and fusion work; this experiment does not claim that target is met.

256 is the largest tested cap, not the absolute hardware maximum. The saturation test uses 256-token inputs and 32 outputs; scaling diagnostics use 128 outputs. Long contexts may lower actual cache admission. TP4/256/65536 used essentially all device memory and was slower, so it is not recommended. Tables and per-case memory are in `results/batching/summary.md`.

The original reference entry point is still available. Within the fast config, `aux.engram_decode_impl=reference` selects the old Engram path. Use the same batch/token settings for numerical comparisons. Real-module states/outputs matched bitwise in all 96 component cases. Full EP generations are not bit-reproducible: 84.5% token agreement across the optimization versus 84.8% between separate reference-mode controls. The 17-document smoke test matched aggregate metrics, but two HellaSwag items changed correctness; it does not establish full-suite quality equivalence.

HF mapped-table validation: setting `OELLM_ENGRAM_MODEL_DIR` to the original GPFS export produced exactly identical 8160 scored logprobs and 32 greedy tokens to the optimized RAM-table run. Warm prefill was 6.613 versus 6.495 s. This loader currently materializes the CPU model before replacing its Engram tensors with file-backed mappings; it does not eliminate the initial peak host-memory requirement. The operating system can keep recently used mapped pages resident, so this test does not establish cold-disk throughput.
