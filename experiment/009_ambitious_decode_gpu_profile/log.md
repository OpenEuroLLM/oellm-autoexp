# Ambitious decode CPU/GPU timeline

## Objective
Explain the ~6.7 tok/s single-request gap and limited batched throughput using actual CPU/CUDA timelines. Separate kernel execution, launch gaps, synchronization, collectives, and host offload. Previous Python wall-time profiles in experiment 008 cannot establish GPU utilization or kernel bottlenecks.

## Protocol
- Checkpoint ambitious_v4_2ph_fp8_3700; vLLM source 0f684e9f4c, batched Engram and fused mHC enabled. Same pinned offline runtime, eager mode, prefix cache off, 256-sequence/32768-token upper caps.
- One independent debug node, four H100s, 45-minute allocation. Compare TP1/DP4/EP4 and TP4/DP1/EP4 sequentially.
- Batches 1/32/256; 256-token inputs; warmup, separate unprofiled 128-token generation, then instrumented 64-token workload. Single EP request uses one-token startup placeholders on idle frontends, which leave the steady decode interval.
- Nsight Systems 2026.5.1, CUDA and NVTX trace, CPU sampling/context-switch tracing disabled. Capture only 12 decode iterations after 8 decode warmup steps, controlled by expert rank zero; other workers are captured as process-tree descendants. Module/mHC method NVTX scopes identify launch ownership.
- Capture ends at the next root forward after the twelve steps, including the preceding step's LM head/sampling. Boundary device synchronization is explicit; analyze middle iterations to avoid capture boundary effects. Compare against unprofiled timing to assess observer overhead.
- Before loading the real model, verify a tiny multiprocess smoke capture contains real kernels from all four GPUs. Fail early if CUDA timestamps are unavailable.

## Artifacts
Scripts in scripts/, named worker extension scripts/korbi/vllm/gpu_profile.py. Raw .nsys-rep and SQLite exports stay on the cluster; ingest aggregate tables, a trace manifest and selected plots. Model weights and math are unchanged.

## Run log
- 2026-11-03: user requested a proper profile against the 7B dense single-request reference. Nsight verified available in the image. Drivers committed before submission. No live training/eval job preempted.

## Related
- ../008_ambitious_inference_speed/log.md
- Initial allocation 48466108 started 20:39:41 CET on as04r3b09. Four-process smoke passed: all four GPUs recorded 16 kernels each with positive device timestamps. The allocation was externally cancelled at 20:47:57 during model initialization, before the actual decode capture; no decode timeline was obtained. Raw partial artifacts preserved in the remote `results_attempt1/` directory, with the smoke result/inventory ingested under local `results/attempt1/`.
- User explicitly requested continuation at 23:27 CET. Restart the same committed profiling protocol in a new independent debug allocation. Parallel experiment `009_mhc256_sharding` has a separate folder; this experiment retains its full existing pathname and history.
- Retry allocation 48469958 started on as01r5b21 at 23:28 CET. Its four-GPU smoke passed. Added SQLite analysis for eight middle decode intervals, per-GPU busy unions, separate compute/collective/copy unions, kernel/launch ownership via process-aware CUDA correlation IDs and nested NVTX ranges. CPU API sums and GPU kernel sums are explicitly separate from union-based activity accounting.
- 23:36 CET: startup diagnostic via py-spy dump on rank zero shows `OellmEngram._host` scanning the RAM-backed mmap during `determine_available_memory/profile_run`. GPU idle during this phase; host page-cache and worker RSS grow, so this is cold GPFS table warming, not a decode hang. No live inference timing has yet been collected.
