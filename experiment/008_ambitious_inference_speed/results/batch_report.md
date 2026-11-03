# Batch-size tests and Engram optimization

Checkpoint: `ambitious_v4_2ph_fp8_3700`. Four H100 64-GiB GPUs per node. All throughput figures are aggregate per node, not per GPU.

All 16 baseline cases passed. The largest tested sequence cap is **256**, equivalent to 64 per DP rank in pure EP4. This is not an absolute hardware maximum. Tests use 1,024 requests (996 unique prefixes), 256 input tokens, and 32 output tokens; scaling diagnostics generate 128 tokens. Prefix caching is disabled. Longer contexts can lower actual admitted concurrency.

## Saturated baseline, 8,192-token aggregate budget

| Concurrent sequences | EP4 score input tok/s | EP4 generation output tok/s | TP4 score input tok/s | TP4 generation output tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 14135 | 168 | 10806 | 103 |
| 64 | 20524 | 262 | 10882 | 134 |
| 128 | 20332 | 382 | 11091 | 147 |
| 256 | 20442 | 494 | 10948 | 172 |

EP4/256/32768 reached ~26k scoring input tok/s. All EP4/256 cases stayed at or below 58.8 GiB sampled device memory. TP4/256/65536 reached ~63.3 GiB and was slower; it has essentially no device-memory margin.

## Same-loaded-model Engram comparison

256 sequences, 32,768 aggregate scheduled tokens. Only the Engram decode implementation changes; checkpoint and prompts are identical.

| Metric | Reference | Batched Engram |
| --- | ---: | ---: |
| Warm generation, including prefill (output tok/s) | 479 | 1059 |
| Decode window, batch 1 (output tok/s) | 6.6 | 6.7 |
| Decode window, batch 32 (output tok/s) | 167.4 | 200.4 |
| Decode window, batch 256 (output tok/s) | 532.7 | 1427.0 |

Warm generation improved **2.21x**. Single-request decode remains **~6.7 tok/s**, far below the ~168 tok/s target. That target is not met.

## Correctness and interpretation

- All 96 real-module checks matched outputs, hash rows and updated states bitwise, including mixed prefill/decode.
- Full generated-token agreement: 84.55% across reference/batched modes, versus 84.81% between separate reference-mode controls. Full EP generation is not bit-reproducible.
- Mean scored NLL change: -0.0001001.
- The 17-document smoke test matched aggregate accuracy: HellaSwag 7/17 raw, 10/17 normalized; GSM8K 10/17. HellaSwag correctness matched on 15/17 documents; GSM8K matched all document/filter correctness records. This is not full-suite quality-equivalence evidence.
- Decode-tail worker samples at batch 256 were ~71% Engram. Batched hashing/convolution removes the per-request loop from this path.
- Delayed single-request sampling after the fix distributes worker wall time across MoE (~28%), CKDA (~24%), MHC (~20%) and Engram (~8%). Frontend/core largely wait for worker execution. These are CPU wall-time samples; GPU timing is needed to separate compute, launch and collective costs.

## Using the result

`config/experiments/korbi/production/flag_eval_fast_mn5.yaml` now defaults to TP1/DP4/EP4, max 256 sequences, 32768 scheduled tokens, and batched Engram. `aux.engram_decode_impl=reference` selects the prior Engram path. The original reference eval entry point remains available.

Baseline vLLM source: 5bd602f81c. Candidate: 0f684e9f4c; runtime source ID `7055a88eda71d9f1`. Submodule pre-commit/type checks, real-checkpoint parity, performance and distributed lm-eval smoke checks passed. Allocation 48461862 completed 0:0 in 1h44m29s; all experiment GPU work has finished.

Raw measurements: [batch table](batching/summary.md), [paired comparison](engram_candidate/comparison.json), [completion](batching/completion.txt). Protocol and commands: [usage](../notes/usage.md), [lab book](../log.md).
