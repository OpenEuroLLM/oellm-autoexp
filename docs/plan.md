# Plan: Integrate Titan-OELLM / TorchTitan Backend

## Goals
- Add a new backend adapter that can launch TorchTitan/Titan-OELLM runs via the `submodules/titan-oellm` subtree.
- Provide Hydra config defaults for the new backend.
- Ensure backend registration is wired into config loading and orchestration.
- Add minimal tests to validate command construction and optional cluster-args resolution.

## Scope & Assumptions
- TorchTitan training entry point is `torchrun -m torchtitan.train`.
- Titan-OELLM provides CLI args via `titan_oellm.cluster_config.get_cli_args(...)` and expects `--key=value` formatting.
- Planning/launch may happen on hosts without Titan-OELLM deps; backend should allow skipping live cluster-path validation and/or using explicit `cluster_args`.

## Implementation Steps
1. **Backend Adapter**
   - Add `oellm_autoexp/backends/titan_backend.py` with:
     - `TitanBackendConfig` (NonStrict + BaseBackendConfig) to capture TorchTitan/Titan-OELLM parameters.
     - A safe `build_launch_command()` that builds:
       - `torchrun` args (from config),
       - `-m torchtitan.train`,
       - resolved cluster args from `titan_oellm.cluster_config.get_cli_args(...)` **or** a user-supplied `cluster_args` string,
       - optional extra CLI overrides (`extra_cli_args`) formatted as `--key=value`.
     - `validate()` to optionally run `validate_paths` when enabled and module import is available.
   - Ensure optional imports are guarded with a clear error if resolution is requested but the module is unavailable.

2. **Registry Wiring**
   - Register the backend via `@register` (compoconf).
   - Update `oellm_autoexp/config/loader.py` and `oellm_autoexp/orchestrator.py` to import the new backend for registration side effects.

3. **Hydra Config**
   - Add `config/backend/titan.yaml` (or `titan_torchrun.yaml`) with defaults for:
     - `torchrun_args`, `launcher_module`, `dataset`, `tokenizer`, `config_file`, `project_root`, `cluster`,
     - `resolve_cluster_args` and `validate_cluster_paths` flags,
     - `env.PYTHONPATH` including `submodules/titan-oellm`.
   - Optionally add a minimal example experiment config under `config/experiments/` if that’s useful for validation.

4. **Tests**
   - Add unit tests for `TitanBackend` command construction:
     - With explicit `cluster_args` (no import needed).
     - With `resolve_cluster_args=True` using a monkeypatched `titan_oellm.cluster_config.get_cli_args`.
   - Confirm registry instantiation via config loader if applicable.

5. **Docs (Optional but Recommended)**
   - Brief README note on selecting the new backend and required environment variables (e.g., `TITAN_USER`).

## Deliverables
- New backend module and config.
- Updated registry imports.
- Tests covering command assembly and optional cluster args resolution.
- Optional doc update.
