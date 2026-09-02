# Cross-checkpoint parameter comparison

This workflow expands the existing four-matrix audit to every named model
parameter and separates three questions:

1. What changes within each run?
2. Which changes are localized to a tensor, layer, row, or input column?
3. Which changes differ between the matched FP8 and BF16 trajectories?

The workflow has two stages. `scan_checkpoint_stats.py` performs expensive
checkpoint I/O once and emits normalized CSVs. `compare_checkpoint_stats.py`
then performs cheap coverage checks, rebasing, ranking, paired comparisons, and
plotting without reopening checkpoints.

## JUPITER launch

From the `oellm-autoexp` repository root on JUPITER, dry-render the default
model-weight scan before submitting it:

```bash
scripts/oellm_32b.sh \
  --config-name experiments/oellm_32b_dense/checkpoint_stats_jupiter \
  --dry-run

scripts/oellm_32b.sh \
  --config-name experiments/oellm_32b_dense/checkpoint_stats_jupiter \
  --submit-and-exit
```

The default `model_all` mode scans the flagship FP8 and BF16 trajectories and
then compares them. Run the larger anonymous optimizer-bucket pass separately:

```bash
scripts/oellm_32b.sh \
  --config-name experiments/oellm_32b_dense/checkpoint_stats_jupiter \
  --dry-run aux.mode=optimizer_all

scripts/oellm_32b.sh \
  --config-name experiments/oellm_32b_dense/checkpoint_stats_jupiter \
  --submit-and-exit aux.mode=optimizer_all
```

The job inherits the production TE 2.18/FA3 image, `/opt/venv`, `/e` and
`/dev/shm` binds, JUPITER account/partition/exclusions, and the trusted FP8
extra-state setting from `oellm_32b_dense_production.yaml`. Artifacts are
written to:

```text
/e/project1/e-sta-openeurollm/production_training/
  oellm_32b_checkpoint_scan/runs/<SLURM_JOB_ID>/
```

Available `aux.mode` values are `model_flagship`, `model_bf16`, `model_all`,
`optimizer_flagship`, `optimizer_bf16`, `optimizer_all`, and `all`.

## 1. Inventory before loading

Run this first for one checkpoint from each tree and confirm that the embedding,
output head, four transformer matrix families, and all norm gains appear:

```bash
apptainer exec <training.sif> python3 scripts/scan_checkpoint_stats.py \
  <checkpoint>/iter_0064000 \
  --optimizer off \
  --metadata-only
```

Do not infer complete coverage from the number of selected tensors alone.
Explicitly search the output for the embedding and untied output-layer keys.

## 2. Scan matched model checkpoints

Use a common 60k baseline and matched 64k/68k checkpoints. Pass explicit
checkpoint directories for BF16 because that trajectory spans the `cont4` and
`cont4b` trees.

```bash
apptainer exec <training.sif> python3 scripts/scan_checkpoint_stats.py \
  <flagship>/checkpoints \
  --iterations 60000,64000,68000,72000,75126 \
  --run flagship \
  --optimizer off \
  --comparison-sample-elements 16384 \
  --output-dir checkpoint_stats/flagship

apptainer exec <training.sif> python3 scripts/scan_checkpoint_stats.py \
  <cont4>/checkpoints/iter_0060000 \
  <cont4b>/checkpoints/iter_0064000 \
  <cont4b>/checkpoints/iter_0068000 \
  --run bf16 \
  --optimizer off \
  --comparison-sample-elements 16384 \
  --output-dir checkpoint_stats/bf16
```

The scalar and channel summaries are exact except for the explicitly labelled
whole-tensor quantiles and elementwise drift samples. Sampling uses stable
indices, so cosine, sign-flip, and relative-delta measurements compare the same
weights at every checkpoint.

After each scan, require:

- `nonfinite_tensors: 0` in the manifest;
- no unexpected entries in `skipped.csv`;
- identical selected tensor identities and shapes across checkpoints;
- expected embedding and output-head keys in `metadata.csv`.

## 3. Compare trajectories and runs

```bash
python3 scripts/compare_checkpoint_stats.py \
  --scan flagship=checkpoint_stats/flagship \
  --scan bf16=checkpoint_stats/bf16 \
  --pair bf16:flagship \
  --baseline-iteration 60000 \
  --output-dir checkpoint_comparison
```

Outputs:

- `coverage.csv` and `coverage_gaps.csv`: missing, extra, skipped, or non-finite
  tensors before any scientific interpretation.
- `tensor_trajectories.csv`: baseline and adjacent-checkpoint changes in RMS,
  extrema, tail factor, sampled elementwise delta, cosine, and sign flips.
- `channel_trajectories.csv`: row/input-column lower and upper tails, dead-channel
  fractions, and baseline-relative movement.
- `family_summary.csv`: layer-aggregated summaries for embeddings, output head,
  norms, and every matrix family.
- `ranked_changes.csv`: screening list of the strongest localized movements.
- `paired_tensor_effects.csv` and `paired_channel_effects.csv`: difference in
  change between BF16 and FP8 from the same baseline.
- `overview.png` and `report.md`: compact navigation into the detailed CSVs.

## 4. Read the results in this order

1. **Coverage:** stop if tensors are missing or skipped.
2. **Finite-state health:** check model parameters and anonymous optimizer
   buckets for non-finite or invalid values.
3. **Global versus localized drift:** compare parameter-weighted global RMS with
   the p95 and maximum layer/channel movements.
4. **Direction-sensitive drift:** use sampled delta and cosine to catch large
   weight movement hidden by unchanged RMS.
5. **Paired FP8/BF16 effects:** prioritize changes that emerge after the common
   baseline and differ consistently across neighboring checkpoints.
6. **Causal validation:** correlate candidates with fixed-data evaluation,
   activations, FP8 scales, gradient/update norms, data source, and loss timing.

The rankings are discovery tools, not significance tests. A striking checkpoint
statistic is not causal unless it precedes the held-out loss degradation and is
altered by a controlled intervention.

## 5. Optimizer pass

Run optimizer analysis separately because production `dp_reshardable`
checkpoints expose anonymous flat buckets and the I/O is large:

```bash
apptainer exec <training.sif> python3 scripts/scan_checkpoint_stats.py \
  <flagship>/checkpoints \
  --iterations 60000,64000,68000,72000,75126 \
  --include optimizer \
  --optimizer on \
  --optimizer-states exp_avg,exp_avg_sq \
  --no-channels \
  --output-dir checkpoint_stats/flagship_optimizer
```

This can detect non-finite values, negative second moments, zeros, and bucket
drift. It cannot attribute a bad bucket offset to a named parameter; that needs
a model-aware optimizer loader or future fully reshardable checkpoints.
