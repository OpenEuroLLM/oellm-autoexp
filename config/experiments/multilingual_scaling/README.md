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

### What "NE" means

`<B>_ne` is **N_NE**, the non-embedding parameter count: transformer-block-only params, excluding *both* the
input embedding and the output (unembedding) matrix. It's the Chinchilla-style convention, and it's a nominal
label — the `hidden_size`/`num_layers` in `backend/megatron/multilingual_scaling/qwen3_dense_<B>_ne.yaml` are
picked to land close to that round number, not to hit it exactly.

N_NE is not the count to use for compute (FLOPs) estimates: the input embedding is a lookup (no matmul, ~free),
but the output embedding does cost a full matmul to produce logits, so a compute-relevant parameter count should
keep it. Following [Porian et al. 2024](https://arxiv.org/abs/2406.19146):

**N_Porian = N_NE + N_output_embedding**

Every `qwen3_dense_<B>_ne.yaml` here sets `untie_embeddings_and_output_weights: False`, i.e. input and output
embeddings are tied to a single matrix, already counted once in the model total — so for this model family
**N_Porian = N_total** (there's no separate input-embedding term left to subtract once the output one is kept).

Actual N_NE / N_total per size, read from the Megatron startup logs, live in the
[multilingual scaling results README](../../../../dense_multilingual_models_scaling_results/README.md#model-sizes).

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


