In these sweeps we suggest the HP grid per model size and token budget.

### Layout

- `training/v1/`, `validation/v1/` — initial HP grid derived from the English scaling experiments.
- `training/v2/`, `validation/v2/` — adjusted HP grid following initial results, since the initial
  centers were not optimal.
- `training/mn5/` — same HP grid as `training/v1/`, but with `num_workers` set per (lr, gbsz) pair,
  since on MN5 the number of dataloader workers matters a lot for throughput and that optimum
  depends on model size and global batch size.
- `training/leo/` — reserved for a grid matching `training/v1/` but adjusted for the Leonardo data
  split inconsistencies (see `dense_multilingual_models_scaling_results/training_challenges/data_split_issue`).
  Not yet created; Leonardo experiments currently point at `training/v1/` directly (see the TODO in
  `config/experiments/multilingual_scaling/training/leo/*.yaml`).

Each file is named `qwen3_dense_<size>_ne.yaml` (no version/cluster suffix — that's carried by the
directory). Consumers select a grid via the `defaults:` path, e.g.
`/sweep/multilingual_scaling/training/v2@sweep: qwen3_dense_0.1B_ne`.

TODO: mention something about the low micro_batch_size on mn5 (and leo?) due to memory consumption.
