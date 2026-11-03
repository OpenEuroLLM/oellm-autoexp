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
