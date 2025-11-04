# OELLM Auto Experimentation Tool — Specification

## Goals
- Unify the capabilities of `autoexperiment`, `megatron-train`, and `oellm_pretrain` in a single mono-repository while keeping the pieces modular.
- Provide a single source-of-truth configuration describing sweeps, launch parameters, backend arguments, monitoring rules, and restart policies.
- Allow swapping Megatron with other training backends by way of a thin adapter layer while keeping Megatron support first-class (including argument validation).
- Automate
  1. sweep expansion (Hydra-style),
  2. SBATCH script generation/validation,
  3. submission/monitoring/restart logic for SLURM jobs.
- Ship a Python package `oellm_autoexp` that exposes a CLI and Python API. The package ships an optional extra `[megatron]` and keeps the Megatron repository under `submodules/megatron` to record the exact version.

## Repository Layout
```
oellm-autoexp/
├── oellm_autoexp/
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py                 # compoconf-based dataclasses defining the config tree
│   │   ├── loader.py                 # parse YAML by way of compoconf.parse_config into typed objects
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
│   │   └── megatron.py               # concrete adapter + argument extraction by way of megatron parser
│   ├── orchestrator.py               # glue: sweep → job plan → submission → monitoring loop
│   ├── workflow/
│   │   ├── manifest.py               # serialisable plan manifests shared between container and host
│   │   └── host.py                   # host-side helpers for submission/monitoring from manifests
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
- Hydra configuration tree under `config/` provides reusable defaults (`project`, `slurm`, `backend`, `monitoring`, `sweep`, `restart`, `scheduler`). Top-level `autoexp.yaml` composes these by way of `group@key` syntax so the CLI can load configurations by name.
- CLI still accepts plain YAML paths; Hydra references resolve by way of `load_config_reference`, which flattens nested structures (for example, `restart.policies` → `restart_policies`).
- Single YAML file consumed by the CLI, with named sections:
  - `project`: descriptive metadata, default output root, and optional persistent state location.
  - `sweep`: Hydra-compatible structure; support product & list semantics similar to `autoexperiment.product_recursive`.
  - `slurm`: defaults for template path, submission command, partition/account, SBATCH keyword args, launch wrappers, and cluster-level environment exports.
  - `launcher` settings split between `launch.env` (env exports) and `launcher_cmd` (prefix command, for example container exec).
  - `container`: optional container configuration specifying the image path and runtime (apptainer/singularity), enabling containerized execution that persists in config for reproducibility.
- `monitoring`: log file pattern, inactivity threshold, retry limits, termination criteria (string, command, metric-based).
- `monitoring.output_paths`: optional list of templates resolved per job (for example `{output_dir}/train.log`) that are tailed alongside the SLURM log to detect activity or checkpoints.
- `backend`: concrete backend config (`megatron` initially). Config files such as `config/backend/megatron.yaml` extend other presets by way of Hydra `defaults` and expose Megatron arguments directly; there are no wrapper keys like `implementation` or manual merge steps.
  - `restart_policies`: list of rules keyed by `error_mode` (`stall`, `crash`, `timeout`, `success`), each specifying whether to restart, mutate args (for example, bump `time_limit`, switch `checkpoint_path`), or abort.
- YAML is read into compoconf-driven dataclasses:
- Define `ConfigInterface` subclasses for core sections (for example, `ProjectConfig`, `SlurmConfig`) and rely on compoconf registries for backend/monitor entries so YAML maps straight onto registrable configs without wrapper layers.
  - Use interface registrations (`@register_interface`, `@register`) where multiple implementations are possible (for example, backend adapters, restart policies, monitoring strategies). This allows backend swaps without changing the schema.
  - `loader.parse_config` (thin wrapper around `compoconf.parse_config`) materialises the tree; `config/evaluator.py` can then instantiate concrete components using compoconf’s `instantiate` helpers.
- Validation should flag:
  - missing mandatory template placeholders (`{name}`, `{output_file}`, …),
  - type mismatches when expanding sweeps,
  - invalid combinations (for example, both `termination_str` and `termination_cmd` absent).
- Provide Hydra resolver interop by registering the same custom resolvers used in `megatron-train/extract_hydra.py` so we retain support for expressions like `oc.mul` etc.
- Expose compoconf-based registries for backend-level options (for example, `BackendInterface`, `RestartPolicyInterface`, `MonitorInterface`) so downstream modules can request instantiated implementations without bespoke factories.

## Workflow Overview
1. **Load + Validate Config**: `load_config_reference` hydrates YAML into compoconf-backed dataclasses. Evaluator objects remain lazy until instantiated during plan generation, allowing schema-only validation on hosts where heavy dependencies (Megatron/Torch) are unavailable.
2. **Sweep Expansion**: Hydra-style expansion creates deterministic job parameter sets (`JobPlan`) with derived names and output paths. Optional `sweep.json` is still emitted for compatibility and tooling.
3. **Plan Generation & Rendering**: Inside the container (or host when dependencies are installed), `scripts/plan_autoexp.py` renders SBATCH scripts, sweep metadata, and a SLURM array wrapper when allowed. The step serialises a **plan manifest** (`oellm_autoexp/workflow/manifest.py`) capturing rendered artifacts, job metadata, instantiated monitor/restart/slurm components, and the fully resolved config. By default manifests are written to `<project.base_output_dir>/manifests/plan_<timestamp>_<token>.json`, so concurrent runs do not clobber each other. (`scripts/run_autoexp.py` wraps steps 3–6 for the common single-shot workflow.)
4. **Manifest Persistence**: The manifest is a JSON document shared between container and host. It includes:
   - job entries (script path, log path, start conditions, inactivity thresholds),
   - component specs (monitor implementation, restart policies, SLURM client config),
   - project/context metadata (base output dir, monitoring state dir, container image/runtime hints),
   - the resolved configuration (paths coerced to strings) for reproducibility and follow-up actions.
5. **Host Submission**: `scripts/submit_autoexp.py --manifest …` consumes the manifest without re-instantiating the backend. It loads monitor/slurm components by way of the manifest specs, submits pending jobs (arrays or individual), and records session state in `<base_output_dir>/monitoring_state/<plan_id>.json`. Restart attempts reuse the stored script path; new actions can request fresh renders by calling the plan step again.
6. **Monitoring Loop**: The same host runtime spins `MonitorController` from manifest data (`submit_autoexp.py` or `monitor_autoexp.py`).
   - Polls SLURM state and log/output timestamps.
   - Applies restart policies configured in the manifest.
   - Records every cycle into the monitoring session; actions emitted by the monitor are written to an `actions/` queue directory derived from the manifest.
   - Monitoring is resumable: `scripts/monitor_autoexp.py --manifest plan.json` rehydrates the manifest, attaches to existing SLURM job IDs, and continues observing without new submissions.
7. **Extensibility**: Backends still expose `validate()` and `build_launch_command()`; new runtimes only need to add component specs into the manifest builder so host-side code can instantiate them without reaching into backend modules.

### Monitoring Sessions and State Persistence
- Each monitoring session receives a unique session ID (8-character UUID prefix) for tracking
- Session files are stored in **visible location**: `<base_output_dir>/monitoring_state/<session_id>.json` (not hidden)
- Each session file contains:
  - Full resolved configuration for reproducibility
  - Project name and creation timestamp
  - Active job state for runtime tracking
- Session files enable:
  - Resuming monitoring after process restarts without re-parsing config
  - Inspecting historical runs and their configurations
  - Managing multiple concurrent monitoring sessions
- Workflow:
  - **On submission**: host `submit` command creates/updates a session file in `monitoring_state/` and writes a pointer to the manifest path.
  - **Monitoring**: `submit_autoexp.py` keeps monitoring unless `--no-monitor` is provided. A later invocation of `monitor_autoexp.py --manifest plan.json` rehydrates the manifest and resumes monitoring without touching the backend.
  - **Session discovery**: `scripts/manage_monitoring.py list/show/remove` remain the primary tooling for enumerating sessions; the manifest path stored in each JSON descriptor makes it trivial to restart monitoring even after long gaps.
- Cleanup behavior:
  - Session files persist for inspection even after jobs complete
  - Manual cleanup by way of `scripts/manage_monitoring.py remove <session_id> [--force]`
- Management commands (all require `--monitoring-state-dir` parameter):
  - `manage_monitoring.py list`: Show all monitoring sessions
  - `manage_monitoring.py show <id>`: Display session details and active jobs
  - `manage_monitoring.py remove <id>`: Clean up session (with --force for active jobs)
  - `manage_monitoring.py dump-config <id>`: Extract full config from session

### Container Integration
- `scripts/run_autoexp_container.py` orchestrates the full flow:
  1. Execute `scripts/plan_autoexp.py` inside the container (render scripts + manifest).
  2. Execute `scripts/submit_autoexp.py --manifest …` on the host (submit + optional monitoring).
- `--manifest <path>` controls where the plan manifest is written/read (default `output/autoexp_plan.json`).
- `--no-submit` stops after planning; useful when subsequent actions (evaluation, conversions) will consume the manifest later.
- `--no-monitor` submits jobs but exits immediately; monitoring can be resumed by way of `scripts/monitor_autoexp.py --manifest …`.
- `--monitor-only` skips planning/submission and forwards straight to the host `monitor` command.
- Container autodiscovery honours top-level `container.image` / `container.runtime`, but explicit overrides remain available by way of `--image`/`--apptainer-cmd`.

## Megatron Integration
- Vendored as Git submodule under `submodules/megatron`; package installs optional dependencies with `pip install .[megatron]`.
- Backend adapter should:
  - mirror `megatron-train/config.py` functionality (arg parser extraction, cmdline serialization),
  - support Hydra config ingestion by leveraging `extract_hydra.py` logic,
  - allow future swap by isolating Megatron-specific resolvers and command building.
- Keep Megatron checks optional to avoid import costs when backend != megatron, but include a lightweight stub pathway so unit tests can run without the real dependency.
- Implement `MegatronBackendConfig(ConfigInterface)` registered under a `BackendInterface`. The evaluator instantiates `MegatronBackend` by way of the compoconf registry, ensuring backends can be swapped by config without changing orchestration code.

## Monitoring & Restart Logic
- Track three signals: SLURM job state (`squeue`), wall-clock runtime, and log progress (mtime + optional regex heartbeat).
- Monitoring config exposes `log_signals` that match regex/substring patterns in SLURM logs and emit named signals. Each signal can map to a restart `mode` (feeding into restart policies) or to an arbitrary `action` collected by the controller (for example, enqueue checkpoint conversion jobs or trigger evaluation workflows). The standalone monitor CLI pre-registers generic signals for errors, run lifecycle (pending → running → ended), completion markers, and checkpoint publication so downstream tooling can attach commands without backend knowledge.
- Monitor compares both SLURM log output and configured training output files; inactivity detection keys off the latest change across either source so freezes without log updates are still caught, mirroring the legacy `autoexperiment` behaviour.
- **Event Logging and Visibility**: The `MonitorController` logs all monitoring events, SLURM state transitions, signal detections, and policy decisions at INFO level to ensure full visibility into the restart system's behavior. This includes:
  - Monitor outcomes (status, last_update, signal counts)
  - SLURM state transitions (PENDING → RUNNING → CANCELLED/FAILED/COMPLETED)
  - Signal detection with mode and action
  - Policy decisions (restart/stop) with reason and attempt count
  - Job restart operations (old job_id → new job_id)
- **Mode Classification with Metadata**: When classifying job state into restart modes, the controller enriches the mode with contextual metadata:
  - `crash` mode for CANCELLED jobs includes `error_type: cancelled` and `subsystem: slurm` to enable selective restart policies
  - `crash` mode for FAILED jobs includes `error_type: slurm_failure`
  - `timeout` mode includes `error_type: timeout` and reason codes
  - This metadata allows restart policies to make informed decisions (for example, restart on CANCELLED but not on OOM)
- Define error modes:
  - `stall`: log unchanged for N seconds while job is `RUNNING` → restart with same plan, increment `stall_retries`.
  - `crash`: SLURM exits non-zero or is CANCELLED without completion marker → consult restart policy based on error_type metadata.
  - `timeout`: job cancelled due to wall-clock (detected by way of SLURM reason codes) → restart with updated `time_limit` or gracefully stop depending on policy.
  - `completed`: termination string or command detected → mark done, no restart.
- Implement restart policies as `RegistrableConfigInterface` derivatives. They decide whether to `restart`, `adjust`, or `abort` based on mode and metadata (error_type, subsystem, signal_name). The `SelectiveRestartPolicy` supports:
  - `restart_on_error_types`: List of error types to restart (for example, `cancelled`, `hang`, `nccl`)
  - `restart_on_subsystems`: List of subsystems to restart (for example, `slurm`, `distributed`)
  - `exclude_error_types`: List of permanent errors to never restart (for example, `oom`, `exception`)
  - `default_action`: Fallback action when no rules match (`restart` or `stop`)
- Provide hooks for backend to adjust restart command (for example, set `--load {latest_ckpt}`) through the same interface contracts. SLURM templates should accept aggregated directive/env strings so cluster-specific SBATCH, launcher, and `srun` settings live entirely in config rather than templated code.

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
    5. Manual `scancel` → CANCELLED state → restart triggered with proper logging.
    6. CUDA OOM in log → crash mode with `error_type: oom` → no restart (excluded).
    7. Resume monitoring after process restart (load persisted state, pick up existing job IDs).
  - Validate that sbatch templates include required placeholders and `--job-name` consistency.
  - **Event Logging Coverage**: Integration tests should verify that all monitoring events are logged:
    - SLURM state transitions are logged with old and new states
    - Log signal detection is logged with signal name and mode
    - Policy decisions are logged with action, reason, and attempt count
    - Job restart operations are logged with old and new job IDs
- Provide fixtures for config YAMLs mirroring patterns from legacy repositories to avoid regressions.
- **Unit Test Coverage for Monitor**: Ensure unit tests cover:
  - CANCELLED job state → crash mode with `error_type: cancelled` metadata
  - Selective restart policy decision based on error_type and subsystem
  - Logging output contains expected transition and decision messages

## Tooling & Packaging Notes
- `pyproject.toml` with Poetry or Hatch for optional extras (`[project.optional-dependencies]` includes `megatron` entry).
- Re-export main CLI in `oellm_autoexp/__init__.py` for programmatic use.
- Document workflow in `README.md` (quickstart, example config) and `docs/` (architecture, backend guide).
- Ensure linting & type checking (ruff + mypy) to keep interfaces tight.
- Ship container helpers (for example, `scripts/run_autoexp_container.py`) so images can execute plan generation with all backend dependencies, while SLURM submission happens on the host. Support both unified (submit + monitor) and separated workflows (submit-only, monitor-only) to enable flexible deployment and resumable monitoring.

## TODO / Backlog
- Extend the standalone `monitor` CLI with persisted state and (optional) real SLURM integration, plus queue wiring for automated follow-up tasks.
- Flesh out remaining CLI surface (`status`, `cleanup`) and persist plan state so processes can resume monitoring after restarts. ✓ (Partially done: monitoring can resume from state)
- Teach orchestrator to persist/reload job history (JSONL/SQLite) and consume `MonitorController.drain_actions()` to enqueue downstream tasks (HF conversion/eval) automatically.
- Implement first-class real SLURM submission path (drop `--fake` guard) including container-friendly sbatch invocation. ✓ (Container workflow separates plan generation from sbatch execution)
- Extend integration tests to cover log-signal driven actions/restarts, post-processing hooks, and container smoke tests.
- Add scheduler throttling (`max_jobs`, rate limits) and restart policy variants that mutate launch arguments (for example, checkpoint substitution) using signal metadata.
- Provide richer container assets (constraint tooling, requirements reduction) and document validation workflows.
- Add `slurm.container` configuration schema to persist container image and runtime settings in config for full reproducibility. ✓ (Container config added at top level)
- Implement monitoring session descriptors for config persistence and management. ✓ (Session descriptors with full config tracking)

## Open Questions / Follow-ups
- Decide whether to keep Hydra as dependency or reimplement minimal resolver support. (Initial plan: depend on Hydra to reuse resolver ecosystem.)
- Consider storing state in SQLite versus plain JSON; JSONL is simpler, SQLite offers concurrent safety. -> Use JSON
- Evaluate whether job array submission should be default or opt-in based on homogeneous sweeps. -> Job Array as Default for sweeps
- Determine policy for cleaning old log files/checkpoints when jobs restart frequently.
- Expand container validation to mirror `megatron-train` behaviour (dry-run + optional fake submission) so images can be smoke-tested without a real cluster.
