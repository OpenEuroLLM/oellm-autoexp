In these sweeps we suggest the HP grid per model size and token budget.

### Layout

- `training/v1/`, `validation/v1/` — initial HP grid derived from the English scaling experiments.
- `training/v2/`, `validation/v2/` — adjusted HP grid following initial results, since the initial
  centers were not optimal.

Each file is named `qwen3_dense_<size>_ne.yaml`. Consumers select a grid via the `defaults:` path, e.g.
`/sweep/multilingual_scaling/training/v2@sweep: qwen3_dense_0.1B_ne`.
