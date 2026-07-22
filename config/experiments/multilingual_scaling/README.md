## Multilingual scaling experiments configurations

To launch experiments:
```
PYTHONPATH=. python scripts/run_autoexp.py --config-name experiments/multilingual_scaling/<stage>/<B>_ne_<cluster>
```
where:
- `<stage>` is `training` or `validation`
- `<B>` denotes the NE model size (e.g., `0.1B`, `0.2B`, `0.4B`, `0.9B`)
- `<cluster>` is `leo`, `lumi`, or `mn5` (MareNostrum training/validation leaf files omit the
  cluster suffix for historical reasons, e.g. `training/0.1B_ne_mn5.yaml`)

### Layout

Each leaf file under `training/` or `validation/` is a thin composition (usually just a
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
- `/backend/megatron/multilingual_scaling/data/<dataset>.yaml` — per-cluster/per-split dataset
  paths, built on `data/common/base.yaml` for the shared dataloader schema.
- `/sweep/multilingual_scaling/...` — per-(cluster, size, stage) hyperparameter grids (left
  untouched by this reorg; genuine experiment design, not boilerplate).

When adding a new (size, cluster, stage) combination, a leaf file should need only: the sweep
pointer, the model/data group pointers, and `job.base_output_dir`. If you find yourself repeating
something else across multiple leaf files, it likely belongs in one of the shared layers above.
