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

Fast config: TP1 + DP4 + EP4, aggregate max 32 sequences, aggregate 2048 scheduled tokens, finite request batches, per-rank durable response cache, local-expert-only loading, RAM-prefetched mmap tables, fused mHC. TP4 remains available with `aux.tp=4 aux.dp=1`.

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

## Layout selection from the measured probes

Use the default TP1/DP4 for mixed/decode workloads. The matched larger-budget test gave TP1/DP4 2.777 s versus TP4/DP1 3.506 s for 32 prompts with 8 outputs each. For prefill-heavy multiple-choice scoring, TP4 was better: 1.234 s versus EP4's 2.630 s for prefill plus one output. Override the fast config with `aux.tp=4 aux.dp=1 aux.token_budget=8192` for that profile. Both profiles still use EP across the four GPUs of a single node. No cross-node collectives are required.

HF mapped-table validation: setting `OELLM_ENGRAM_MODEL_DIR` to the original GPFS export produced exactly identical 8160 scored logprobs and 32 greedy tokens to the optimized RAM-table run. Warm prefill was 6.613 versus 6.495 s. This loader currently materializes the CPU model before replacing its Engram tensors with file-backed mappings; it does not eliminate the initial peak host-memory requirement. The operating system can keep recently used mapped pages resident, so this test does not establish cold-disk throughput.
