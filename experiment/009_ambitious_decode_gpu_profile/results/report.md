# Ambitious decode: GPU timeline findings

The current serving path has substantial eager launch overhead and synchronization between ranks. Switching EP4 to TP4 does not fix single-request latency. Native FP8 inference is not enabled by the FP8 checkpoint format.

Job **48469958** completed **0:0** on **as01r5b21**, four H100s, in **26m16s**. No production allocation was preempted. This profiles vLLM; the HF implementation was not GPU-profiled in this experiment.

## Capture-disabled controls

Same checkpoint `ambitious_v4_2ph_fp8_3700`, vLLM source `0f684e9f4c`, runtime ID `7055a88eda71d9f1`. Batched Engram, RAM-warmed host tables, Triton Sinkhorn, eager execution, prefix cache off, maximum 256 sequences and 32768 scheduled tokens. Inputs are the fixed experiment-008 tokenized prompts (256 input tokens); each control generates 128 output tokens, ignoring EOS.

All output throughput figures are **aggregate over the node**, not per GPU. Decode-window throughput is `requests * 127 / (last final-token timestamp - earliest first-token timestamp)`. Mean request ITL is the mean of each request's `(last-first)/127`. Staggered first tokens and batch tails mean these are not interchangeable quantities. These are finite batch controls, not a new saturated evaluation throughput benchmark.

| Layout | Requests | Mean request ITL, ms | Decode-window output tok/s | Traced step, ms | Trace/control ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1/DP4/EP4 | 1 | 150.37 | 6.65 | 208.79 | 1.39 |
| TP1/DP4/EP4 | 32 | 176.80 | 177.69 | 202.45 | 1.15 |
| TP1/DP4/EP4 | 256 | 184.56 | 1267.50 | 214.21 | 1.16 |
| TP4/DP1/EP4 | 1 | 147.59 | 6.78 | 188.74 | 1.28 |
| TP4/DP1/EP4 | 32 | 173.05 | 175.75 | 190.86 | 1.10 |
| TP4/DP1/EP4 | 256 | 184.13 | 1161.33 | 193.38 | 1.05 |

The ~120 tok/s dense reference and ~168 tok/s single-request target remain unmet. EP4 remains a reasonable evaluation default; this one paired profile does not establish statistical significance for the small single-request difference.

## What the GPUs actually do

[Per-step timelines](analysis/decode_timeline.png) and [activity accounting](analysis/gpu_activity.png) show thousands of short kernels separated by gaps. Each timeline panel has its own time scale. The figures use actual CUDA timestamps; orange is collective execution, which can include waiting for peers. Blue includes all other kernels, including conversions/reductions, not only matrix multiplication.

For the single-request EP4 trace, each GPU executes roughly **9,600 kernels per decode step**. GPU 0's annotated breakdown is:

| Owner | Kernels/step, excluding collectives | Summed kernel ms/step, excluding collectives |
| --- | ---: | ---: |
| mHC read | 2646 | 4.95 |
| mHC write | 588 | 1.39 |
| RMSNorm | 1150 | 1.86 |
| CKDA | 1764 | 5.15 |
| MoE | 2304 | 7.55 |
| Engram | 940 | 1.18 |
| Attention | 126 | 0.39 |

Module rows assign each kernel to its innermost annotation using its launch API's process/thread and correlation ID. The MoE row excludes its **96 NCCL kernels per step**. Kernel sums are not utilization and should not be treated as independent critical-path costs.

The exported model has **98 serial sublayers**: 48 MoE, 42 CKDA, six attention and two Engram. There are 360 experts, top-5 routing, hidden size 2048. Even after the previous Sinkhorn fusion, mHC read/write and normalization remain mostly eager PyTorch expressions.

Single-request EP4 activity, measured over the same complete middle steps:

| GPU | Non-collective kernels, union ms/step | Collectives, union ms/step | Idle ms/step |
| ---: | ---: | ---: | ---: |
| 0 | 23.23 | 83.69 | 101.79 |
| 1 | 21.04 | 7.79 | 179.89 |
| 2 | 21.06 | 157.84 | 29.81 |
| 3 | 21.11 | 159.76 | 27.86 |

GPU 1 spends ~122 ms/step idle before the next kernel's launch API has even begun. Its annotated CPU wall times include MoE 66.65 ms, CKDA 58.62 ms, mHC read 39.73 ms and RMSNorm 14.10 ms. These CPU scopes are inclusive where nested and must not simply be added. Other GPUs spend much longer in NCCL while this rank submits work. TP4 has a similar pattern: GPU 1 is ~169 ms idle in a ~189 ms traced single-request step while its peers spend ~152–157 ms in collective kernels.

This supports launch overhead and peer-readiness imbalance as major costs. It does **not** prove a network bandwidth bottleneck or a defective node. Tracing adds overhead and can amplify rank imbalance. TP4 CPU affinity was checked: each worker is bound to its GPU's NUMA CPU domain. CPU thread count is eight; a same-model 1-versus-8 thread control remains worth testing.

At EP4 batch 256, non-collective kernel activity rises to ~68 ms/GPU/step while total traced step time is ~214 ms. Batching amortizes dispatch, but substantial gaps and collective waiting remain. RAM-resident Engram transfer activity is small: total copy-engine time is below 0.1 ms/GPU/step at batch 1. That does not remove its CPU lookup/synchronization cost, and it says nothing about uncached disk-offloaded tables.

## FP8 and startup

The serving log explicitly selects **TRITON Unquantized MoE**, with `quantization=None`. The loader dequantizes the checkpoint's E4M3 expert tensors to BF16. FP8 on disk therefore does not imply native FP8 expert execution. The exact `E=90,N=768,H100` tuning file is also missing, so default MoE launch parameters are used.

Cold EP4 startup took 636.47 seconds and warmed the Engram mmap from GPFS. A stack sample showed `_host` scanning the table while RSS/page cache grew; this was separate from decode. TP4 model initialization took 107.42 seconds using the warmed cache, then its first request spent ~3 minutes compiling the fallback FlashInfer TRT-LLM all-reduce workspace. The control timings above follow warmup and exclude both costs.

## Next implementation priorities

1. **Fuse complete mHC read/write and RMSNorm**, preserving normalization precision, zero-centered weights and state/output semantics. Together they account for ~4384 launches per token on GPU 0, about 45% of its current launches. Validate module outputs and end-to-end scoring before enabling either in HF or vLLM.
2. **Fuse routing/shared-expert elementwise operations and Engram hash generation**, then reduce CKDA state gathering/cloning and small launch chains. The batched Engram change removed the request loop but still launches ~940 kernels for its two modules.
3. **Capture graph-compatible decode regions** once host lookup/state boundaries are handled. Engram's CPU lookup prevents assuming that simply clearing `enforce_eager` will work. Piecewise capture/compiled regions may be practical.
4. Test CPU thread count and model-shape-specific MoE tuning. Investigate native FP8 as a separate backend/numerics change, preserving the per-expert post-activation bound and validating checkpoint rounding behavior.

These changes can benefit the HF path too, but this trace does not quantify HF gains. A launch-count reduction is not a promised proportional speedup. The measured non-collective GPU work also needs fusion/optimization to approach an 8 ms/token target; removing host gaps alone does not establish 120 tok/s.

## Capture limits and reproduction

Nsight Systems 2026.5.1 traced CUDA/NVTX only, with CPU sampling and context-switch tracing off. Before model loading, a four-process smoke test proved positive kernel timestamps on every GPU. Each model capture records 12 steps after eight decode warmup steps; only complete middle intervals are analyzed.

Non-controller CUDA buffers have incomplete leading/trailing coverage. The analyzer intersects all four GPUs' timestamp coverage, requires complete root-to-next-root intervals and at least two middle steps, and rejects >2% cross-GPU kernel-count spread. This retains steps 11–14 in five traces and 12–14 in TP4 batch 256. Missing buffers are never counted as idle. Capture-disabled controls remain under Nsight injection; comparison to the previous independent measurements supports the single-request baseline, but profiler overhead cannot be fully removed from the trace. Eight-step statistics and hardware SM counters were not obtained.

Scalar aggregates are [JSON files](analysis/ep4_b1.json); compact timeline replay data is stored beside them as `.timeline.json.gz`. [Timing summary](timing_summary.json), per-rank controls in `ep4/` and `tp4/`, [checkpoint metadata](checkpoint_config.json), and [raw trace manifest with hashes](trace_manifest.json) preserve provenance. All six downloaded SQLite files matched the remote SHA-256 hashes. Large raw `.nsys-rep`/SQLite files remain at the exact paths in that manifest.

From the repository root, replay analysis with Python's standard library; plotting additionally needs matplotlib (this run used 3.11.2):

```bash
python experiment/009_ambitious_decode_gpu_profile/scripts/analyze_trace.py /path/to/ep4/capture.*.sqlite /path/to/tp4/capture.*.sqlite --out experiment/009_ambitious_decode_gpu_profile/results/analysis
python experiment/009_ambitious_decode_gpu_profile/scripts/summarize.py experiment/009_ambitious_decode_gpu_profile/results
python experiment/009_ambitious_decode_gpu_profile/scripts/plot_trace.py experiment/009_ambitious_decode_gpu_profile/results/analysis
```
