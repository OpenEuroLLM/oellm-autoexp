# Checkpoint 3700 export comparison

Shared launcher prefix on MN5, from `$R/autoexp_hf_fast007` (`R=/gpfs/projects/ehpc533/users/korbip/hybrid_v019`):

```bash
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --yes --no-monitor \
  --monitor-state-dir "$R/pool_state" \
  --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p132 \
  --config-name experiments/korbi/hybrid_speed/kernel_tests_mn5 \
  aux.tests= slurm.sbatch.qos=acc_xehpc slurm.sbatch.time=06:00:00 \
  aux.name=fast_hf_export007parallel \
  'aux.bench="bash experiment/007_fast_hf_export/scripts/run_parallel.sh"'
```

Variants submitted with that prefix:

| aux.name | aux.bench | Run |
|---|---|---|
| fast_hf_export007 | `bash experiment/007_fast_hf_export/scripts/run.sh` | .0234: unit tests and AB/BA benchmark passed; missing GNU timer |
| fast_hf_export007b | `SKIP_READER_BENCHMARK=1 bash experiment/007_fast_hf_export/scripts/run.sh` | .0235: conversion/shape checks passed; missing external accelerate path |
| fast_hf_export007c | same as 007b, after adding baseline's pylibs_hfx path | .0236: resume validation on same node |
| fast_hf_export007parallel | `bash experiment/007_fast_hf_export/scripts/run_parallel.sh` | .0237: fresh full conversion and concurrent validation |

Input is pinned at `/gpfs/scratch/ehpc533/hybrid_nucleus/production/main_mhc4/ambitious_e360_eng80_v4_2ph/ckpt_eval/iter_0003700`; resolved configuration is `config-mn5-pool-p129.0338.yaml` in that run. Source hashes and checkpoint metadata digest are captured by `scripts/record_provenance.py`.

The baseline frozen snapshot is `$R/autoexp_mhc_eval3700`, run .0233. Candidate source is that snapshot plus the explicit opt-in reader changes committed at c9c032f8 / 9def4359. FP8 conversion, model implementation and score code are identical.

Drivers are resumable via completion markers. Use a fresh output root to repeat a full timing comparison; an existing root skips completed stages. Ensure `<fresh-root>/paired` links to the baseline `hf_exports/paired` input so paired validation is included. The v2 pipeline can be invoked directly from a one-node/four-GPU test allocation:

```bash
export HF_EXPORT_READER=fast HF_EXPORT_READ_THREADS=8 EXPERT_DTYPE=fp8
export PYTHONPATH="$PYTHONPATH:$R/pylibs_hfx"
bash experiment/007_fast_hf_export/scripts/export_main_run_v2.sh ambitious_v4 3700 <fresh-root>
```

v2 is scoped to the tested ambitious_v4 architecture, with FP8 routed experts. It uses the same four-GPU node budget as baseline and runs ordinary scoring on one two-GPU pair and paired scoring on the other.

## Reusable v2 entry point

The reusable pipeline is `scripts/korbi/hf_export/export_main_run_v2.sh`; its timer is a byte-identical copy of the tested standard-library helper. Compared with the tested experimental pipeline, the reusable script changes the timer path, defaults the reader to fast/8 threads and experts to FP8 (the tested settings), and rejects unsupported model/dtype combinations early. The existing legacy entry point is unchanged.

```bash
scripts/korbi/mn5_autoexp.sh scripts/run_autoexp.py --yes --no-monitor \
  --monitor-state-dir "$R/pool_state" \
  --pool /gpfs/scratch/ehpc533/hybrid_nucleus/pools/mn5_pool_p132 \
  --config-name experiments/korbi/production/hf_export_v2_mn5 \
  aux.step=<checkpoint-step>
```

This uses the normal export root and completion markers, with the baseline's external accelerate dependency inherited from `hf_export_mn5`.
