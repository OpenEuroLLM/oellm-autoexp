# Ambitious fused kernels and decode CUDA graphs

## Objective
Reduce the ~9600 eager kernels/token measured in experiment 009. Implement complete mHC read/write and FP32 RMSNorm fusion, and CUDA graph replay around host Engram lookups and collective boundaries. Validate real-checkpoint outputs/state transitions and matched workloads before enabling the fast evaluation defaults.

## Constraints
- No mutation of existing Slurm jobs (root AGENTS.md). Only new debug submissions.
- Reference math, eager execution and existing fast flags remain selectable.
- Preserve BF16 intermediate rounding in mHC and FP32 RMSNorm semantics.
- Capture must never advance recurrent state an extra time. Static graph inputs must refresh slot indices on every replay. Host lookup and distributed collectives remain eager boundaries.
- New graph cache bounded by configured row sizes; unsupported/mixed prefill paths remain eager.

## Run log
- 2026-11-04: User authorized implementing/enabling fusion and CUDA graphs. Initial scope is the vLLM path profiled in experiment 009; reusable fusion math can be applied to HF subsequently.

## Related
- ../009_ambitious_decode_gpu_profile/results/report.md
- ../008_ambitious_inference_speed/log.md
