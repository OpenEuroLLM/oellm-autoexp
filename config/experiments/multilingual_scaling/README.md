## Multilingual scaling experiments configurations

To launch experiments:
```
python scripts/run_autoexp.py --config-name experiments/multilingual_scaling/<stage>/<cluster>/<B>_ne_sweep_<version>
```
where:
- `<stage>` is `training` or `validation`
- `<cluster>` is `leo`, `lumi`, or `mn5`
- `<B>` denotes the NE model size (e.g., `0.1B`, `0.2B`, `0.4B`, `0.9B`)
- `<version>` denotes the sweep version: v1: initial HP grid derived from the English scaling experiments or v2: adjusted HP grid following initial results, since the initial
  centers were not optimal
### Layout

Each leaf file under `training/<cluster>/` or `validation/<cluster>/` is a thin composition (usually just a
`defaults:` list plus a `job.base_output_dir` and any genuinely experiment-specific override) of
shared building blocks:

- `common/base.yaml` — settings shared by every multilingual_scaling experiment regardless of
  cluster, size, or stage (wandb identity, job-name/index placeholders).
- `cluster/{leo,lumi,mn5}.yaml` — per-cluster infra + policy shared by every experiment on that
  cluster (backend/container/slurm group selection, node-count formula,
  account/qos, cluster-specific env vars).
- `validation/common.yaml` — settings shared by every validation-stage experiment regardless of
  cluster or size (skip_train, eval-only batch/iters formulas, single-node override).
- `/backend/megatron/multilingual_scaling/models/<size>.yaml` — per-size model architecture.
- `/sweep/multilingual_scaling/...` — per-(cluster, size, stage) hyperparameter grids.


