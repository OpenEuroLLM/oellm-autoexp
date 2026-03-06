# Titan/TorchTitan Backend (YAML-First)

This repo now supports a TorchTitan/Titan‑OELLM backend where **the YAML config is the single source of truth**. The backend generates a TorchTitan TOML on the fly and launches `torchtitan.train` via `torchrun`.

## How It Works

1. **Hydra/compoconf parses the YAML** into the Titan schema (generated from TorchTitan/Titan‑OELLM dataclasses).
2. The backend **emits a TOML** (`config.titan.toml` by default) from that YAML.
3. The backend runs:

```
torchrun ... -m torchtitan.train --job.config_file=<generated toml>
```

This means you can express everything in YAML, use Hydra sweeps, and still run TorchTitan without maintaining TOML files manually.

## Cluster Path Resolution

By default, **no Titan‑OELLM cluster path resolution is used**. That avoids double‑config and removes any dependency on `TITAN_USER` or `user/<name>/cluster_paths.toml`.

If you *do* want Titan‑OELLM to fill in dataset/tokenizer paths, you can enable it:

```yaml
backend:
  cluster_args_source: titan_oellm
  validate_cluster_paths: true
  dataset: slimpajama_627b
  tokenizer: neox
  cluster: juwels
  config_file: base_plus.toml
  project_root: "."
```

### Recommended (YAML‑only) Path Overrides

Set these directly in YAML under `backend.titan`:

```yaml
backend:
  titan:
    model:
      tokenizer_path: /path/to/tokenizer
    data:
      data_prefix: /path/to/train/prefix
      chunks_dir: /path/to/chunks
      dataloader: MMapDataset
      min_doc_len: 10
    validation:
      data_prefix: /path/to/val/prefix
    benchmarks:
      wikitext2_path: /path/to/wikitext2
      wikitext103_path: /path/to/wikitext103
      lambada_path: /path/to/lambada
```

You can also use `backend.cluster_args` (string) or `backend.cluster_args_overrides` (dict) for late overrides. The apply order is:

1. Titan‑OELLM `get_cli_args` (if enabled)
2. `backend.cluster_args` (string)
3. `backend.cluster_args_overrides` (dict)

## Custom Config Modules

You can still use TorchTitan custom config modules:

```yaml
backend:
  custom_config_module: my_pkg.my_custom_config
  require_custom_module: true
```

If importable, the backend merges it into the strict schema and filters YAML to the merged schema before emitting TOML.

## TOML Output

The TOML is written to:

```
${job.base_output_dir}/config.titan.toml
```

You can override this with `backend.toml_output_path`.

Optional provenance JSON:

```yaml
backend:
  write_toml_provenance: true
```

This writes `config.titan.toml.provenance.json` alongside the TOML with backend metadata.

## Quick Start (YAML‑Only)

```yaml
# config/experiments/titan_speed_test.yaml

defaults:
  - /backend: titan
  - /slurm: base
  - /job: default
  - _self_

backend:
  titan:
    model:
      name: llama3
      flavor: debugmodel
    training:
      steps: 200
      local_batch_size: 4
    parallelism:
      tensor_parallel_degree: 1
      pipeline_parallel_degree: 1
    data:
      data_prefix: /path/to/train
      chunks_dir: /path/to/chunks
      dataloader: MMapDataset
      min_doc_len: 10
    validation:
      data_prefix: /path/to/val

slurm:
  sbatch:
    nodes: 1
    gpus_per_node: 4
    time: "00:30:00"
```

Run:

```
OELLM_LOG_LEVEL=INFO python scripts/run_autoexp.py --config-ref experiments/titan_speed_test
```

## Notes
- TorchTitan/Titan‑OELLM must be available in the runtime environment.
- `PYTHONPATH` should include `submodules/titan-oellm` and `submodules/titan-oellm/torchtitan` (configured in `config/backend/titan.yaml`).
- You will need a custom container, the build instructions and definition file integration into `oellm-autoexp` are WIP