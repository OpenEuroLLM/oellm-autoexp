# Plan: YAML-First TorchTitan/Titan-OELLM Config (Single Source of Truth)

## Goal
Make a single YAML config (Hydra/compoconf) the source of truth for TorchTitan/Titan‑OELLM runs, including user-extended config classes, while keeping plan/submit flows working on hosts without Titan dependencies.

## Key Principles
- **YAML is authoritative**: one config controls backend, slurm/env/container, sweep, and job management.
- **Schema parity**: TorchTitan config structure is represented as nested dataclasses in compoconf.
- **Schema-only vs full validation**: can plan without Titan deps, but optionally validate against TorchTitan when available.
- **Extensibility**: user extensions are first-class, not an afterthought.
- **Disambiguation via Union + class_name**: compoconf uses `class_name` to select strict vs generic schema.

## Design Overview

### 1) Schema Source
- Use TorchTitan’s nested dataclasses as the “ground truth” schema:
  - `torchtitan.config.JobConfig` (and Titan‑OELLM extensions, e.g. `titan_oellm.configs.sci_job_config.JobConfig`).
- Provide a **generated snapshot** in `oellm_autoexp/backends/titan/config_schema.py` to enable schema-only mode.

### 2) Handling User-Extended Config Classes
TorchTitan already supports user extensions via `job.custom_config_module` and a merge routine (`ConfigManager._merge_configs`). We can emulate that.

**Proposed mechanism**
- YAML exposes:
  - `backend.titan.custom_config_module: "my_pkg.my_custom_config"`
- On load/validation/build:
  1. Determine **base config class** (TorchTitan default or Titan‑OELLM default).
  2. If `custom_config_module` is set and importable, **merge dataclasses** using the same algorithm as TorchTitan.
  3. Use the merged class for parsing/validation/CLI generation.

**Fallback when module is not importable**
- If running in schema-only mode and the module isn’t available:
  - Warn and **skip** merging (still allow YAML to carry unknown keys under generic mode).
  - Optional strict flag: `backend.titan.require_custom_module` to fail fast if missing.

This mirrors TorchTitan behavior while allowing planning on login nodes.

### 3) YAML → CLI / TOML
- **Primary**: build CLI args directly from the YAML dataclass instance (`--section.key=value`), matching Tyro format.
- **Intermediate TOML**: generate a TOML artifact from the YAML tree on backend init and pass `--job.config_file=...`.

### 4) Strict vs Generic Schema (Union + class_name)
- `backend.titan` is typed as a Union of two dataclasses:
  - **Strict**: generated schema, with `class_name: TitanJobConfig`.
  - **Generic**: `NonStrictDataclass` fallback, with `class_name: TitanJobConfigGeneric`.
- compoconf selects based on `backend.titan.class_name` (or default).
- This allows:
  - **standard models** to validate on login nodes,
  - **custom/hacky models** to opt into generic parsing.

### 5) Avoid Config Explosion
- Use Hydra defaults to split Titan config into reusable parts:
  - `config/backend/titan/base.yaml` (core defaults)
  - `config/backend/titan/model/*.yaml`
  - `config/backend/titan/data/*.yaml`
  - `config/backend/titan/training/*.yaml`
  - `config/backend/titan/parallelism/*.yaml`
- Compose them in `config/backend/titan.yaml` with defaults list.
- This mirrors existing Megatron setup without combinatorial explosion.

## Implementation Steps

1. **Generator scripts**
   - Add `scripts/generate_titan_dataclass.py`:
     - Imports TorchTitan (and optional Titan‑OELLM extensions).
     - Merges custom config module if provided (CLI flag).
     - Emits `oellm_autoexp/backends/titan/config_schema.py` with nested dataclasses + `class_name`.

2. **Backend config schema**
   - Create `oellm_autoexp/backends/titan_backend.py` with:
     - `TitanBackendConfig` including a `titan` field typed to the Union (strict/generic).
     - Fields for `custom_config_module`, `require_custom_module`, `full_schema_validation`.
     - `validate()` in two modes:
       - Schema-only (generated snapshot).
       - Full validation via TorchTitan `ConfigManager` if available.

3. **Custom config merge helper**
   - Implement a small helper that mirrors TorchTitan’s `_merge_configs` logic (reuse code with attribution or call into TorchTitan when available).

4. **CLI generation**
   - Serialize the nested dataclass into `--section.key=value` arguments.
   - Ensure lists/booleans match Tyro expectations.
   - Allow `backend.titan.extra_cli_args` for overrides.

5. **Hydra config layout**
   - Add `config/backend/titan.yaml` (composed defaults).
   - Add `config/backend/titan/{data,model,training,parallelism}.yaml` presets.

6. **Tests**
   - Schema-only validation and command build.
   - Custom config module merging (mock module with a dataclass).
   - TOML emission + CLI assembly sanity checks.

## User Experience
- Users write a **single YAML** config.
- Optional custom config modules are set via `backend.titan.custom_config_module`.
- All overrides and sweeps are done via Hydra overrides.
- The backend emits the correct TorchTitan/Titan‑OELLM CLI, using YAML as the canonical config.

## Open Questions
- Should the generator accept multiple extension modules and merge them in order? (TorchTitan currently supports only one.)

