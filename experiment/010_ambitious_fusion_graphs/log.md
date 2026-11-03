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
- 00:18 CET: submitted one new acc_debug allocation 48470754 (one node, one-hour limit), snapshot autoexp_fusion010, runtime source edfebe66a7fc49a1. vLLM source 0e4675b997; all pre-commit and mypy checks passed before submission. No existing Slurm job mutated.
- User additionally requested fast evals of amb_v4_2ph_mhc at step5000 after validation, using existing configs. Existing hf_export_v2_mn5 selects model ambitious_v4 and persistent/rolling checkpoint discovery; flag_eval_fast_mn5 keeps the standard 21 FLAG rows and four shards. Persistent ckpt/iter_0005000 is complete (latest marker5000, metadata238211192 bytes), no step5000 HF export exists. Preserve it in ckpt_eval via hard links, then use the established export + checks and fast-eval path after graph correctness/performance are established.
- Step5000 pinned successfully: ckpt_eval/iter_0005000/.metadata has two hard links. Preparing the existing v2 export pipeline independently of graph validation; downstream evaluation remains gated on both export readiness and validated optimized serving. Pool p129 host004 is held, runs production plus existing single-node work, and permits five concurrent pool runs. Enqueue only a new single-node export record; do not preempt, release hosts, or change Slurm jobs.
- Existing pool host uses account cns144; the first export enqueue was rejected for account mismatch (ehpc533), without creating a run. Retried only the new submission with slurm.sbatch.account=cns144. Export started as mn5-pool-p129.0353 on spare node as06r5b24 at00:24:27. No allocation state/priority changed. Export/check outputs go to the existing hf_exports root, name ambitious_v4_2ph_fp8_5000.
- Numerical evaluation criteria before results: real CKDA graph outputs and complete updated states must match eager exactly under changing slots; full scored NLL must remain finite and mean delta <0.002, with per-token differences and A/A generation noise reported. The 17-document HellaSwag/GSM8K smoke is required before fast-eval rollout; it is not full quality equivalence evidence.
- Prepare a second frozen candidate (274f52ebfb) while the first runs: share graph scratch through vLLM's global graph pool, retain strong output references, replay islands sequentially in model order. This avoids one recurrent temporary pool per CKDA layer/batch. Restrict root graph eligibility to powers of two (up to256 rows), cache at most8 shapes per island, and add Engram hash/sign graph islands (16 bounded entries). Arbitrary/mixed batch sizes remain eager with fused kernels, avoiding capture churn as generation batches shrink. Test real Engram output/state parity in addition to CKDA. Keep first candidate/results untouched; new node-constrained debug submission follows its natural completion and uses the warm page cache.
- First candidate's real CKDA tests passed on all4 GPUs: batches1/8/64, five changing-slot transitions each, outputs and full updated caches bitwise equal. The v2 plan is session1793748740 (no Slurm ID yet: acc_debug max_active=1 guard); it must be submitted from that original plan after job48470754 completes naturally. Planned new job requests the same node and afterany:48470754; no existing job is modified.
- Before actual v2 submission, strengthen CKDA state validation to test both reference and fused mHC/norm modes, and copy only that checked-in driver update to the still-unsubmitted v2 snapshot. No running candidate code changed.
