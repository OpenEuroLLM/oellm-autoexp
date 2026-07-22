## Multilingual scaling experiments configurations

To launch experiments:
```
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/multilingual_scaling/<stage>/<cluster>/<B>_ne
```
where:
- `<stage>` is `training` or `validation`
- `<cluster>` is `leo`, `lumi`, or `mn5`
- `<B>` denotes the NE model size (e.g., `0.1B`, `0.2B`, `0.4B`, `0.9B`)

### Layout

Each leaf file under `training/<cluster>/` or `validation/<cluster>/` is a thin composition (usually just a
`defaults:` list plus a `job.base_output_dir` and any genuinely experiment-specific override) of
shared building blocks:

- `common/base.yaml` — settings shared by every multilingual_scaling experiment regardless of
  cluster, size, or stage (wandb identity, job-name/index placeholders).
- `cluster/{leo,lumi,mn5}.yaml` — per-cluster infra + policy shared by every experiment on that
  cluster (backend/container/slurm group selection, data cache path, node-count formula,
  account/qos, cluster-specific env vars, whether periodic eval runs during training).
- `stage/validation.yaml` — settings shared by every validation-stage experiment regardless of
  cluster or size (skip_train, eval-only batch/iters formulas, single-node override).
  Training has no equivalent file since, once cluster + common are factored out, there was
  nothing left that's common to training across all three clusters.
- `/backend/megatron/multilingual_scaling/models/<size>.yaml` — per-size model architecture.
- `/sweep/multilingual_scaling/...` — per-(cluster, size, stage) hyperparameter grids (left
  untouched by this reorg; genuine experiment design, not boilerplate).

Dataset settings (data_args_path, tokenizer, split, num_workers, ...) are inlined directly in each
leaf file's `backend.megatron` block rather than pulled from a shared config group — they're
duplicated per size within a cluster/stage directory (e.g. all 4 files under `training/leo/` repeat
the same block) by design, so everything about one experiment lives in one file. The one exception:
LUMI's `num_workers` is set once in `cluster/lumi.yaml` (overriding the dataset's own default of 7
down to 6) rather than repeated in every LUMI leaf file, since it's a hardware-tuning value owned by
the cluster, not the dataset.

When adding a new (size, cluster, stage) combination, a leaf file should need only: the sweep
pointer, the model group pointer, the dataset block, and `job.base_output_dir`. If you find
yourself repeating something else across multiple leaf files in the same cluster directory, it
likely belongs in `cluster/<cluster>.yaml` instead.
