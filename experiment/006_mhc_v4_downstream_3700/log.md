# v4 two-phase mHC downstream evaluation, checkpoint 3700

## Objective
Run the established downstream suite on the latest complete checkpoint requested by the user.

## Setup
- Model: `ambitious_v4_2ph`, mHC-4, production main_mhc4.
- Step: 3700 (latest completed rolling save captured before submission; step 3600 was also pinned during initial inspection).
- Pin: `/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph/ckpt_eval/iter_0003700`.
- Snapshot: `autoexp_mhc_eval3700`, isolated copy of `autoexp_hfexp13` (latest export memory fix, vLLM 7993918d04).
- Export: FP8 routed experts, matching production's expert parameter-gather representation and prior step-2000 evaluation.
- Suite: default 21 FLAG lm-eval tasks, full datasets (`limit=0`), four shards. At most three run concurrently on p132's spare nodes; training retains its 128 nodes.
- Prerequisite: successful HF export, shape checks, fixed-validation scoring, and paired validation when available. The export wrapper writes `.downstream_ready` only on success; the pool monitor gates all eval shards on that marker.
- No new allocation submissions.

## Run Log
- 2026-11-03: user requested latest-checkpoint downstream evaluation. Initial latest was 3600; 3700 completed during setup, so the evaluation selects 3700. Preserve it with hard links so rolling cleanup cannot remove the evaluation input.
- Commands: `scripts/launch.sh`, using shared configs `hf_export_mn5` and `flag_eval_mn5`.

## Results
- Export: `/gpfs/scratch/ehpc533/hybrid_nucleus/hf_exports/ambitious_v4_2ph_fp8_3700`.
- Downstream: `/gpfs/scratch/ehpc533/hybrid_nucleus/flag_evals/ambitious_v4_2ph_fp8_3700`.
- Per-task JSON and per-shard TSV summaries are written by the existing evaluator.

- 2026-11-03 09:48 CET: submitted export `kernel_tests_hf_export_ambitious_v4_3700_fp8_9d0466` and four full-suite eval shard records. Export is RUNNING as `mn5-pool-p132.0233` on spare node `as05r5b06` in allocation 48335386. All four eval records have the `.downstream_ready` FileExistsCondition and are waiting; the pool monitor will start them automatically after successful export/validation. Launch manifest: `results/launch.json`. LDM retains its 128 nodes in run .0232; no new allocation was submitted.

- 2026-11-03 10:35 CET: investigated export latency without changing the running job. Exporter is CPU-only (all 4 GPUs report 0 MiB/0% utilization); process average CPU ~188%. Observed logical read counter ~1.38 TB (not physical GPFS disk traffic). Checkpoint `.metadata` is 484,968,871 bytes. Source `fsdp_dtensor.iter_layers` calls `dcp.load` separately for 98 layers; PyTorch's active `FileSystemReader.read_metadata` unconditionally opens and unpickles the complete metadata on each call. A stack sample caught metadata unpickling; a separate 12-second/20-Hz py-spy profile captured 239 samples, all in checkpoint tensor reads. Per-layer completion interval medians: MoE 47.2 s, CKDA 19.8 s, attention 17.2 s, Engram 135.9 s (2 tables). These intervals include loading, transformation and occasional shard flushes, not isolated quantization timing. Therefore no exact FP8 fraction is established. Clear optimization candidates: cache export-reader metadata; reduce repeated loading/planning and small shard reads; profile and then accelerate FP8 conversions separately. Profile: `results/export_profile.json`.
