# OELLM Auto Experimentation Tool — Specification

## Goals
- Unify the capabilities of `autoexperiment`, `megatron-train`, and `oellm_pretrain` in a single mono-repo while keeping the pieces modular.
- Provide a single source-of-truth configuration describing sweeps, launch parameters, backend arguments, monitoring rules, and restart policies.
- Allow swapping Megatron with other training backends via a thin adapter layer while keeping Megatron support first-class (including argument validation).
- Automate 
  1. sweep expansion (Hydra-style),
  2. SBATCH script generation/validation,
  3. submission/monitoring/restart logic for SLURM jobs.
- Ship a Python package `oellm_autoexp` that exposes a CLI and Python API. The package ships an optional extra `[megatron]` and keeps the Megatron repo under `submodules/megatron` to record the exact version.

## Repository Layout
```
oellm-autoexp/
├── oellm_autoexp/
│   ├── __init__.py
│   ├── cli.py                        # entry points (click/typer)
│   ├── config/
│   │   ├── schema.py                 # compoconf-based dataclasses defining the config tree
│   │   ├── loader.py                 # parse YAML via compoconf.parse_config into typed objects
│   │   ├── evaluator.py              # resolve backends/options, produce runtime plan context
│   │   └── resolvers.py              # hydra-style resolvers for derived params (optional)
│   ├── sweep/
│   │   ├── expander.py               # hydra-like sweep expansion
│   │   └── planner.py                # builds job plan objects from sweeps
│   ├── slurm/
│   │   ├── template_renderer.py      # render sbatch scripts from templates + args
│   │   ├── validator.py              # lint & dry-run templates, type-check sbatch args
│   │   └── fake_sbatch.py            # mock sbatch/squeue/scancel used in tests
│   ├── monitor/
│   │   ├── watcher.py                # log/time-based freeze detection
│   │   └── policy.py                 # restart/abort decisions encapsulated here
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                   # abstract interface (arg dataclasses, command builder)
│   │   └── megatron.py               # concrete adapter + argument extraction via megatron parser
│   ├── orchestrator.py               # glue: sweep → job plan → submission → monitoring loop
│   ├── job_state.py                  # job model, enums for status/error type
│   └── utils/
│       └── shell.py                  # wrappers for invoking sbatch, squeue, etc.
├── submodules/
│   └── megatron/                     # git submodule pointing to Megatron-LM
├── tests/
│   ├── unit/
│   │   └── ...
│   └── integration/
│       ├── conftest.py
│       ├── test_end_to_end.py        # full sweep -> submission -> monitoring with fake SLURM
│       └── fake_slurm_backend.py     # reusable mock objects shared with package
├── SPEC.md
├── pyproject.toml
├── README.md
└── docs/
    ├── architecture.md
    └── usage.md
```

*Note:* actual module naming can be adjusted, but keep the separation of concerns.

## Configuration Model
- Hydra configuration tree under `config/` provides reusable defaults (`project`, `slurm`, `backend`, `monitoring`, `sweep`, `restart`, `scheduler`). Top-level `autoexp.yaml` composes these via `group@key` syntax so the CLI can load configurations by name.
- CLI still accepts plain YAML paths; Hydra references resolve via `load_config_reference`, which flattens nested structures (e.g., `restart.policies` → `restart_policies`).
- Single YAML file consumed by the CLI, with named sections:
  - `project`: descriptive metadata, default output root, global environment settings.
  - `sweep`: Hydra-compatible structure; support product & list semantics similar to `autoexperiment.product_recursive`.
  - `slurm`: defaults for template path, submission command, partition/account, SBATCH keyword args.
  - `launcher` / `srun`: optional wrappers injected before the backend command and extra `srun` flags.
  - `monitoring`: log file pattern, inactivity threshold, retry limits, termination criteria (string, command, metric-based).
  - `backend`: name (`megatron` initially), CLI overrides, pointer to backend-specific configs.
  - `restart_policies`: list of rules keyed by `error_mode` (`stall`, `crash`, `timeout`, `success`), each specifying whether to restart, mutate args (e.g., bump `time_limit`, switch `checkpoint_path`), or abort.
- YAML is read into compoconf-driven dataclasses:
  - Define `ConfigInterface` subclasses for each section (e.g., `ProjectConfig`, `SlurmConfig`, `MonitoringConfig`, `BackendConfig`) so they inherit parsing/validation primitives.
  - Use interface registrations (`@register_interface`, `@register`) where multiple implementations are possible (e.g., backend adapters, restart policies, monitoring strategies). This allows backend swaps without changing the schema.
  - `loader.parse_config` (thin wrapper around `compoconf.parse_config`) materialises the tree; `config/evaluator.py` can then instantiate concrete components using compoconf’s `instantiate` helpers.
- Validation should flag:
  - missing mandatory template placeholders (`{name}`, `{output_file}`, …),
  - type mismatches when expanding sweeps,
  - invalid combinations (e.g., both `termination_str` and `termination_cmd` absent).
- Provide Hydra resolver interop by registering the same custom resolvers used in `megatron-train/extract_hydra.py` so we retain support for expressions like `oc.mul` etc.
- Expose compoconf-based registries for backend-level options (e.g., `BackendInterface`, `RestartPolicyInterface`, `MonitorInterface`) so downstream modules can request instantiated implementations without bespoke factories.

## Workflow Overview
1. **Load + Validate Config**: Use `compoconf.parse_config` (wrapped in `loader.py`) to map YAML into typed dataclass instances. The resulting objects implement `ConfigInterface`, enabling lazy instantiation. For `backend=megatron`, the evaluator resolves the `BackendConfig` to a concrete adapter and invokes Megatron argument validation.
2. **Sweep Expansion**: Use Hydra-like resolver & Cartesian product semantics to expand `sweep` block into concrete job parameter sets. Each job receives a unique deterministic ID + job name.
3. **Plan Generation**: Feed expanded configs into `config/evaluator.py`, which leverages compoconf’s registry-instantiated components (backend adapters, monitoring policies) to assemble a `JobPlan` (command, env, expected log paths). Persist optional `sweep.json` for compatibility.
4. **SBATCH Rendering**: Fill a template (Jinja/Omega templating) with plan values. Validate script statically (e.g., ensure `#SBATCH --job-name` matches) and optionally run `sbatch --test-only` when available.
5. **Submission**: Invoke `sbatch`, capturing job IDs. Support job arrays when all jobs are compatible; fall back to per-job submission otherwise.
6. **Monitoring Loop** (derived from `autoexperiment.manager`):
   - Poll fake or real `squeue`, track runtime and last log modification time.
   - Detect states: `pending`, `running`, `stalled`, `failed`, `completed`.
   - On stall/crash, consult compoconf-registered restart policy implementations: optionally mutate job plan (e.g., update `--load` to latest checkpoint) and requeue by instantiating new run directives.
   - Record transitions in a lightweight SQLite or JSONL history for post-mortem/CLI queries.
7. **Extensibility**: Backends expose `validate(config)`, `build_launch_command(expanded_cfg)`, and optional `postprocess(log_dir)` hooks. New backends register via entry points for discoverability.

## Command-Line Interface
Expose a single `oellm-autoexp` console script with subcommands:
- `plan CONFIG.yaml`: load+validate, output sweep summary, write artifacts without submitting.
- `submit CONFIG.yaml [--dry-run]`: plan + render + submit (supports `--dry-run` to print commands and exit).
- `monitor CONFIG.yaml [--job JOB_NAME|JOB_ID]`: attach to running plan; can resume after process restarts.
- `status [--config CONFIG.yaml]`: display table of job states.
- `cleanup CONFIG.yaml`: optional helper to archive logs or cancel jobs.

CLI should share state via a project directory (`.oellm-autoexp/`) containing serialized plans, submission metadata, restart counters, etc., enabling resume/restarts akin to `autoexperiment`'s name-based recovery. Persist the compoconf-evaluated configuration (e.g., via `dump_config`) so repeated invocations reuse the same resolved objects.

## Megatron Integration
- Vendored as Git submodule under `submodules/megatron`; package installs optional dependencies with `pip install .[megatron]`.
- Backend adapter should:
  - mirror `megatron-train/config.py` functionality (arg parser extraction, cmdline serialization),
  - support Hydra config ingestion by leveraging `extract_hydra.py` logic,
  - allow future swap by isolating Megatron-specific resolvers and command building.
- Keep Megatron checks optional to avoid import costs when backend != megatron, but include a lightweight stub pathway so unit tests can run without the real dependency.
- Implement `MegatronBackendConfig(ConfigInterface)` registered under a `BackendInterface`. The evaluator instantiates `MegatronBackend` via the compoconf registry, ensuring backends can be swapped by config without changing orchestration code.

## Monitoring & Restart Logic
- Track three signals: SLURM job state (`squeue`), wall-clock runtime, and log progress (mtime + optional regex heartbeat).
- Define error modes:
  - `stall`: log unchanged for N seconds while job is `RUNNING` → restart with same plan, increment `stall_retries`.
  - `crash`: SLURM exits non-zero without completion marker → restart if retries remain.
  - `timeout`: job cancelled due to wall-clock (detected via SLURM reason codes) → restart with updated `time_limit` or gracefully stop depending on policy.
  - `completed`: termination string or command detected → mark done, no restart.
- Implement restart policies as `RegistrableConfigInterface` derivatives. They decide whether to `restart`, `adjust`, or `abort` and can be instantiated from the config via compoconf. Provide hooks for backend to adjust restart command (e.g., set `--load {latest_ckpt}`) through the same interface contracts. SLURM templates should accept aggregated directive/env strings so cluster-specific SBATCH, launcher, and `srun` settings live entirely in config rather than templated code.

## Testing Strategy
- **Unit tests**: cover config parsing, sweep expansion semantics, SLURM template rendering, Megatron arg validation (use pared-down parser fixtures).
- Include unit tests ensuring compoconf parsing honors required/optional fields, registry instantiation, and nested backend selection. Add golden tests validating `dump_config` round-trips for sample YAMLs.
- **Integration tests**:
  - Build a fake SLURM layer (`tests/integration/fake_slurm_backend.py`) that simulates `sbatch`, `squeue`, `scontrol show job`, `scancel`, with controllable timelines and log file updates.
  - Scenario coverage:
    1. Happy path: multi-axis sweep → array submission → jobs complete.
    2. Stall detection: log stops updating, ensure restart triggered once, log mutated, job completes.
    3. Crash without restart policy → job marked failed, no rerun.
    4. Timeout with policy increasing limit or aborting after retry budget.
    5. Resume monitoring after process restart (load persisted state, pick up existing job IDs).
  - Validate that sbatch templates include required placeholders and `--job-name` consistency.
- Provide fixtures for config YAMLs mirroring patterns from legacy repos to avoid regressions.

## Tooling & Packaging Notes
- `pyproject.toml` with Poetry or Hatch for optional extras (`[project.optional-dependencies]` includes `megatron` entry).
- Re-export main CLI in `oellm_autoexp/__init__.py` for programmatic use.
- Document workflow in `README.md` (quickstart, example config) and `docs/` (architecture, backend guide).
- Ensure linting & type checking (ruff + mypy) to keep interfaces tight.

## Open Questions / Follow-ups
- Decide whether to keep Hydra as dependency or reimplement minimal resolver support. (Initial plan: depend on Hydra to reuse resolver ecosystem.)
- Consider storing state in SQLite vs. plain JSON; JSONL is simpler, SQLite offers concurrent safety.
- Evaluate whether job array submission should be default or opt-in based on homogeneous sweeps.
- Determine policy for cleaning old log files/checkpoints when jobs restart frequently.
- Expand container validation to mirror `megatron-train` behaviour (dry-run + optional fake submission) so images can be smoke-tested without a real cluster.
