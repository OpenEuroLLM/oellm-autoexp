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
  - `project`: descriptive metadata, default output root, and optional persistent state location.
  - `sweep`: Hydra-compatible structure; support product & list semantics similar to `autoexperiment.product_recursive`.
  - `slurm`: defaults for template path, submission command, partition/account, SBATCH keyword args, launch wrappers, and cluster-level environment exports.
  - `launcher` settings split between `launch.env` (env exports) and `launcher_cmd` (prefix command, e.g. container exec).
  - `container`: optional container configuration specifying the image path and runtime (apptainer/singularity), enabling containerized execution that persists in config for reproducibility.
- `monitoring`: log file pattern, inactivity threshold, retry limits, termination criteria (string, command, metric-based).
- `monitoring.output_paths`: optional list of templates resolved per job (e.g. `{output_dir}/train.log`) that are tailed alongside the SLURM log to detect activity or checkpoints.
- `backend`: concrete backend config (`megatron` initially). Config files such as `config/backend/megatron.yaml` extend other presets via Hydra `defaults` and expose Megatron arguments directly; there are no wrapper keys like `implementation` or manual merge steps.
  - `restart_policies`: list of rules keyed by `error_mode` (`stall`, `crash`, `timeout`, `success`), each specifying whether to restart, mutate args (e.g., bump `time_limit`, switch `checkpoint_path`), or abort.
- YAML is read into compoconf-driven dataclasses:
- Define `ConfigInterface` subclasses for core sections (e.g., `ProjectConfig`, `SlurmConfig`) and rely on compoconf registries for backend/monitor entries so YAML maps straight onto registrable configs without wrapper layers.
  - Use interface registrations (`@register_interface`, `@register`) where multiple implementations are possible (e.g., backend adapters, restart policies, monitoring strategies). This allows backend swaps without changing the schema.
  - `loader.parse_config` (thin wrapper around `compoconf.parse_config`) materialises the tree; `config/evaluator.py` can then instantiate concrete components using compoconf’s `instantiate` helpers.
- Validation should flag:
  - missing mandatory template placeholders (`{name}`, `{output_file}`, …),
  - type mismatches when expanding sweeps,
  - invalid combinations (e.g., both `termination_str` and `termination_cmd` absent).
- Provide Hydra resolver interop by registering the same custom resolvers used in `megatron-train/extract_hydra.py` so we retain support for expressions like `oc.mul` etc.
- Expose compoconf-based registries for backend-level options (e.g., `BackendInterface`, `RestartPolicyInterface`, `MonitorInterface`) so downstream modules can request instantiated implementations without bespoke factories.

## Workflow Overview
1. **Load + Validate Config**: Use `compoconf.parse_config` (wrapped in `loader.py`) to map YAML into typed dataclass instances. The resulting objects implement `ConfigInterface`, enabling lazy instantiation. For `backend=megatron`, the evaluator instantiates the registered Megatron backend directly from the parsed config and invokes argument validation.
2. **Sweep Expansion**: Use Hydra-like resolver & Cartesian product semantics to expand `sweep` block into concrete job parameter sets. Each job receives a unique deterministic ID + job name.
3. **Plan Generation**: Feed expanded configs into `config/evaluator.py`, which leverages compoconf's registry-instantiated components (backend adapters, monitoring policies) to assemble a `JobPlan` (command, env, expected log paths). Persist optional `sweep.json` for compatibility and emit an aggregate array script when `slurm.array` is enabled so compatible sweeps collapse to SLURM job arrays.
4. **SBATCH Rendering**: Fill a template (Jinja/Omega templating) with plan values. Validate script statically (e.g., ensure `#SBATCH --job-name` matches) and optionally run `sbatch --test-only` when available.
5. **Submission**: Invoke `sbatch`, capturing job IDs. Support job arrays when all jobs are compatible; fall back to per-job submission otherwise. Create monitoring session file in `<base_output_dir>/monitoring_state/<session_id>.json` containing full config and job state.
   - **Container Workflow**: When `container` is configured, the script generation happens inside the container (which has all dependencies), but `sbatch` submission happens on the host (which has SLURM commands). The container runner (`run_autoexp_container.py`) orchestrates this by running plan generation with `--no-submit` inside the container, parsing the generated sbatch command, and executing it on the host.
6. **Monitoring Loop** (derived from `autoexperiment.manager`):
   - Poll fake or real `squeue`, track runtime and last log modification time.
   - Detect states: `pending`, `running`, `stalled`, `failed`, `completed`.
   - On stall/crash, consult compoconf-registered restart policy implementations: optionally mutate job plan (e.g., update `--load` to latest checkpoint) and requeue by instantiating new run directives.
   - Record transitions in session files for post-mortem/CLI queries.
   - **Separate Execution**: Monitoring can run independently from submission. After jobs are submitted, monitoring can be started separately by reading session files:
     - `--monitor-session <id>`: Monitor specific session from `monitoring_state/<id>.json`
     - `--monitor-all`: Discover and monitor all sessions in `monitoring_state/` directory
     - Sessions contain full config, enabling monitoring without re-parsing original config
     - This enables resuming monitoring after process restarts and separating concerns between submission and long-running monitoring.
7. **Extensibility**: Backends expose `validate(config)`, `build_launch_command(expanded_cfg)`, and optional `postprocess(log_dir)` hooks. New backends register via entry points for discoverability.

## Command-Line Interface
Expose a single `oellm-autoexp` console script with subcommands:
- `plan CONFIG.yaml`: load+validate, output sweep summary, write artifacts without submitting.
- `submit CONFIG.yaml [--dry-run]`: plan + render + submit (supports `--dry-run` to print commands and exit). When `--no-monitor` is specified, returns immediately after submission without starting the monitoring loop. Creates session file in `monitoring_state/` directory.
- `monitor CONFIG.yaml [--job JOB_NAME|JOB_ID]`: attach to running plan; can resume after process restarts. Can be run independently after jobs are submitted.
- `monitor --monitor-session <id>`: monitor a specific session by reading from `monitoring_state/<id>.json`.
- `monitor --monitor-all`: discover and monitor all active sessions in `monitoring_state/` directory.
- `status [--config CONFIG.yaml]`: display table of job states.
- `cleanup CONFIG.yaml`: optional helper to archive logs or cancel jobs.
- `monitor MONITOR.yaml`: run monitoring/alerting standalone using only the monitoring config, emitting default signals and dispatching configured actions.

CLI should share state via a project directory containing serialized plans, submission metadata, restart counters, etc., enabling resume/restarts akin to `autoexperiment`'s name-based recovery. Persist the compoconf-evaluated configuration (e.g., via `dump_config`) so repeated invocations reuse the same resolved objects.

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
  - **On submission**: `submit_jobs()` creates session file in `monitoring_state/` directory
  - **On monitoring**: `run_autoexp.py --monitor-session <id>` reads session file and reconstructs config
  - **Monitor all**: `run_autoexp.py --monitor-all` discovers and monitors all active sessions
- Cleanup behavior:
  - Session files persist for inspection even after jobs complete
  - Manual cleanup via `scripts/manage_monitoring.py remove <session_id> [--force]`
- Management commands (all require `--monitoring-state-dir` parameter):
  - `manage_monitoring.py list`: Show all monitoring sessions
  - `manage_monitoring.py show <id>`: Display session details and active jobs
  - `manage_monitoring.py remove <id>`: Clean up session (with --force for active jobs)
  - `manage_monitoring.py dump-config <id>`: Extract full config from session

### Container Integration
- `scripts/run_autoexp_container.py` is the **recommended entry point** for all workflows (containerized or not)
- The script intelligently handles execution based on configuration:
  - `--image <path>`: Use specified container image (explicit override)
  - No `--image` + `container.image` set in config: Auto-detect and use configured container
  - No `--image` + `container: null` or `container: {}`: Execute directly on host without container
- For containerized execution, handles SLURM submission split:
  - Plan generation runs inside container (has all backend dependencies like Megatron)
  - SLURM submission (`sbatch`) runs on host (has SLURM commands)
  - Monitoring runs on host (only needs SLURM commands and file access)
- The container script supports:
  - `--no-run`: Generate scripts inside container, output sbatch command, but don't execute sbatch
  - `--no-monitor`: Execute sbatch on host, but return immediately without monitoring
  - `--monitor-only`: Skip submission, attach to existing jobs and monitor (resuming from persisted state)
- Container configuration persisted at top level (`container.image`, `container.runtime`) makes workflows fully reproducible from config alone
- Lazy imports: OmegaConf/Hydra only imported when auto-detecting config, so `--image` works even without these dependencies installed on host

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
- Monitoring config exposes `log_signals` that match regex/substring patterns in SLURM logs and emit named signals. Each signal can map to a restart `mode` (feeding into restart policies) or to an arbitrary `action` collected by the controller (e.g., enqueue checkpoint conversion jobs or trigger evaluation workflows). The standalone monitor CLI pre-registers generic signals for errors, run lifecycle (pending → running → ended), completion markers, and checkpoint publication so downstream tooling can attach commands without backend knowledge.
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
  - This metadata allows restart policies to make informed decisions (e.g., restart on CANCELLED but not on OOM)
- Define error modes:
  - `stall`: log unchanged for N seconds while job is `RUNNING` → restart with same plan, increment `stall_retries`.
  - `crash`: SLURM exits non-zero or is CANCELLED without completion marker → consult restart policy based on error_type metadata.
  - `timeout`: job cancelled due to wall-clock (detected via SLURM reason codes) → restart with updated `time_limit` or gracefully stop depending on policy.
  - `completed`: termination string or command detected → mark done, no restart.
- Implement restart policies as `RegistrableConfigInterface` derivatives. They decide whether to `restart`, `adjust`, or `abort` based on mode and metadata (error_type, subsystem, signal_name). The `SelectiveRestartPolicy` supports:
  - `restart_on_error_types`: List of error types to restart (e.g., `cancelled`, `hang`, `nccl`)
  - `restart_on_subsystems`: List of subsystems to restart (e.g., `slurm`, `distributed`)
  - `exclude_error_types`: List of permanent errors to never restart (e.g., `oom`, `exception`)
  - `default_action`: Fallback action when no rules match (`restart` or `stop`)
- Provide hooks for backend to adjust restart command (e.g., set `--load {latest_ckpt}`) through the same interface contracts. SLURM templates should accept aggregated directive/env strings so cluster-specific SBATCH, launcher, and `srun` settings live entirely in config rather than templated code.

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
- Provide fixtures for config YAMLs mirroring patterns from legacy repos to avoid regressions.
- **Unit Test Coverage for Monitor**: Ensure unit tests cover:
  - CANCELLED job state → crash mode with `error_type: cancelled` metadata
  - Selective restart policy decision based on error_type and subsystem
  - Logging output contains expected transition and decision messages

## Tooling & Packaging Notes
- `pyproject.toml` with Poetry or Hatch for optional extras (`[project.optional-dependencies]` includes `megatron` entry).
- Re-export main CLI in `oellm_autoexp/__init__.py` for programmatic use.
- Document workflow in `README.md` (quickstart, example config) and `docs/` (architecture, backend guide).
- Ensure linting & type checking (ruff + mypy) to keep interfaces tight.
- Ship container helpers (e.g., `scripts/run_autoexp_container.py`) so images can execute plan generation with all backend dependencies, while SLURM submission happens on the host. Support both unified (submit + monitor) and separated workflows (submit-only, monitor-only) to enable flexible deployment and resumable monitoring.

## TODO / Backlog
- Extend the standalone `monitor` CLI with persisted state and (optional) real SLURM integration, plus queue wiring for automated follow-up tasks.
- Flesh out remaining CLI surface (`status`, `cleanup`) and persist plan state so processes can resume monitoring after restarts. ✓ (Partially done: monitoring can resume from state)
- Teach orchestrator to persist/reload job history (JSONL/SQLite) and consume `MonitorController.drain_actions()` to enqueue downstream tasks (HF conversion/eval) automatically.
- Implement first-class real SLURM submission path (drop `--fake` guard) including container-friendly sbatch invocation. ✓ (Container workflow separates plan generation from sbatch execution)
- Extend integration tests to cover log-signal driven actions/restarts, post-processing hooks, and container smoke tests.
- Add scheduler throttling (`max_jobs`, rate limits) and restart policy variants that mutate launch arguments (e.g., checkpoint substitution) using signal metadata.
- Provide richer container assets (constraint tooling, requirements reduction) and document validation workflows.
- Add `slurm.container` configuration schema to persist container image and runtime settings in config for full reproducibility. ✓ (Container config added at top level)
- Implement monitoring session descriptors for config persistence and management. ✓ (Session descriptors with full config tracking)

## Open Questions / Follow-ups
- Decide whether to keep Hydra as dependency or reimplement minimal resolver support. (Initial plan: depend on Hydra to reuse resolver ecosystem.)
- Consider storing state in SQLite vs. plain JSON; JSONL is simpler, SQLite offers concurrent safety. -> Use JSON
- Evaluate whether job array submission should be default or opt-in based on homogeneous sweeps. -> Job Array as Default for sweeps
- Determine policy for cleaning old log files/checkpoints when jobs restart frequently. 
- Expand container validation to mirror `megatron-train` behaviour (dry-run + optional fake submission) so images can be smoke-tested without a real cluster.
