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
│   │   ├── dag_resolver.py           # DAG-based resolution + sibling interpolation
│   │   ├── planner.py                # job plan helpers
│   │   └── validator.py              # plan validation (deps, interpolations)
│   ├── slurm/
│   │   ├── client.py                 # real/fake SLURM clients
│   │   ├── template_renderer.py      # render sbatch scripts from templates + args
│   │   └── validator.py              # lint & dry-run templates, type-check sbatch args
│   ├── monitor/
│   │   ├── watcher.py                # log/time-based detectors (SLURM + file polling)
│   │   ├── controller.py             # orchestrates event emission, action execution, restarts
│   │   ├── actions.py                # compoconf-registered monitor actions
│   │   ├── conditions.py             # gate/guard primitives shared by monitor + CLI
│   │   ├── events.py                 # EventRecord + persistence helpers
│   │   ├── event_bindings.py         # declarative bindings (event + conditions + actions)
│   │   ├── action_queue.py           # single-file action queue entries per event/action
│   │   └── states.py                 # typed monitor state definitions
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                   # abstract interface (arg dataclasses, command builder)
│   │   └── megatron_backend.py       # concrete adapter + argument extraction by way of megatron parser
│   ├── orchestrator.py               # glue: sweep → job plan → submission → monitoring loop
│   ├── workflow/
│   │   ├── manifest.py               # serialisable plan manifests shared between container and host
│   │   ├── plan.py                   # plan serialization helpers
│   │   └── host.py                   # host-side helpers for submission/monitoring from manifests
│   ├── job_state.py                  # job model, enums for status/error type
│   └── utils/
│       ├── logging_config.py         # logging defaults + CLI flags
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
- Hydra configuration tree under `config/` provides reusable defaults (`project`, `slurm`, `backend`, `monitoring`, `sweep`, `scheduler`, `job`). Top-level `autoexp.yaml` composes these by way of a `defaults` list so the CLI can load configurations by name.
 - CLI still accepts plain YAML paths; Hydra references resolve by way of `load_config_reference`, which composes the config and applies overrides before compoconf instantiation.
- Single YAML file consumed by the CLI, with named sections:
  - `project`: descriptive metadata, default output root, and optional persistent state location.
  - `sweep`: Hydra-compatible structure; support product & list semantics similar to `autoexperiment.product_recursive`.
  - `job`: declarative start/cancel conditions, optional start command/interval, and inactivity thresholds (parsed directly into dataclasses; no extra overrides required).
  - `slurm`: defaults for template path, submission command, partition/account, SBATCH keyword args, launch wrappers, and cluster-level environment exports.
  - `launcher` settings split between `launch.env` (env exports) and `launcher_cmd` (prefix command, for example container exec).
  - `container`: optional container configuration specifying the image path and runtime (apptainer/singularity), enabling containerized execution that persists in config for reproducibility.
- `monitoring`: log file pattern, inactivity threshold, termination criteria (string, command, metric-based), and the declarative event bindings described below.
- `monitoring.output_paths`: optional list of templates resolved per job (for example `{output_dir}/train.log`) that are tailed alongside the SLURM log to detect activity or checkpoints.
- `backend`: concrete backend config (`megatron` initially). Config files such as `config/backend/megatron.yaml` extend other presets by way of Hydra `defaults` and expose Megatron arguments directly; there are no wrapper keys like `implementation` or manual merge steps.
- `monitoring.log_events`: regex/substring detectors (mostly log-derived) that emit metadata/state (`CrashState`, `SuccessState`, `StalledState`, …) and inline actions (instantiate by way of compoconf). `monitoring.state_events`: synthetic SLURM/job events (for example, `stall`, `crash`, `completed`) that directly attach actions executed by the controller.
- `monitoring.event_bindings`: shared schema describing `EventActionConfig` entries (conditions + actions). Both `log_events` and `state_events` embed the bindings rather than referencing a separate policy engine; this keeps the primitive count low (`Event` + `Condition` + `Action`).
- YAML is read into compoconf-driven dataclasses:
- Define `ConfigInterface` subclasses for core sections (for example, `ProjectConfig`, `SlurmConfig`) and rely on compoconf registries for backend/monitor entries so YAML maps straight onto registrable configs without wrapper layers.
  - Use interface registrations (`@register_interface`, `@register`) where multiple implementations are possible (for example, backend adapters, monitoring strategies, action/condition variants). This allows backend swaps without changing the schema.
  - `loader.parse_config` (thin wrapper around `compoconf.parse_config`) materialises the tree; `config/evaluator.py` can then instantiate concrete components using compoconf’s `instantiate` helpers.
- Validation should flag:
  - missing mandatory template placeholders (`{name}`, `{output_file}`, …),
  - type mismatches when expanding sweeps,
  - invalid combinations (for example, both `termination_str` and `termination_cmd` absent).
- Provide Hydra resolver interop by registering the same custom resolvers used in `megatron-train/extract_hydra.py` so we retain support for expressions like `oc.mul` etc.
- Expose compoconf-based registries for backend-level options (for example, `BackendInterface`, `MonitorInterface`) so downstream modules can request instantiated implementations without bespoke factories.

## Workflow Overview
1. **Load + Validate Config**: `load_config_reference` hydrates YAML into compoconf-backed dataclasses. Evaluator objects remain lazy until instantiated during plan generation, allowing schema-only validation on hosts where heavy dependencies (Megatron/Torch) are unavailable.
2. **Sweep Expansion**: Hydra-style expansion creates deterministic job parameter sets (`JobPlan`) with derived names and output paths. Optional `sweep.json` is still emitted for compatibility and tooling. DAG-based sibling resolution is used for dependency ordering and interpolation.
3. **Plan Generation & Rendering**: Inside the container (or host when dependencies are installed), `scripts/plan_autoexp.py` renders SBATCH scripts, sweep metadata, and a SLURM array wrapper when allowed. The step serialises a **plan manifest** (`oellm_autoexp/workflow/manifest.py`) capturing rendered artifacts, job metadata, instantiated monitor/slurm components, and the fully resolved config. By default manifests are written to `<project.base_output_dir>/manifests/plan_<timestamp>_<token>.json`, so concurrent runs do not clobber each other. (`scripts/run_autoexp.py` wraps steps 3–6 for the common single-shot workflow.)
4. **Manifest Persistence**: The manifest is a JSON document shared between container and host. It includes:
   - job entries (script path, log path, start conditions, inactivity thresholds),
   - component specs (monitor implementation + event bindings, SLURM client config),
   - project/context metadata (base output dir, monitoring state dir, container image/runtime hints),
   - the resolved configuration (paths coerced to strings) for reproducibility and follow-up actions.
5. **Host Submission**: `scripts/submit_autoexp.py --manifest …` consumes the manifest without re-instantiating the backend. It loads monitor/slurm components by way of the manifest specs, submits pending jobs (arrays or individual), and records session state in `<base_output_dir>/monitoring_state/<plan_id>.json`. Each `StoredJob` entry now includes the resolved log path (with `%j`/`%A_%a` expanded), `last_monitor_state`, `last_slurm_state`, and `last_updated` to simplify resume scenarios. Once a SLURM ID is known, `log_path_current` is updated by way of a symlink to the resolved log file so monitoring always follows the expected "current" path. Restart attempts reuse the stored script path; new actions can request fresh renders by calling the plan step again.
6. **Monitoring Loop**: The same host runtime spins `MonitorController` from manifest data (`submit_autoexp.py` or `monitor_autoexp.py`).
   - Polls SLURM state and log/output timestamps.
   - Applies event bindings (event-driven restarts, command hooks, downstream automation).
   - Records every cycle into the monitoring session; actions emitted by the monitor are written to an `actions/` queue directory derived from the manifest.
   - Monitoring is resumable: `scripts/monitor_autoexp.py --session <plan_id>` reads the persisted session JSON (which points back to the manifest), rehydrates the runtime, and continues observing without new submissions. The legacy `--manifest` path remains available for backwards compatibility. When invoked with `--all`, sessions without active jobs are skipped unless `--include-completed` is specified.
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
  - **Monitoring**: `submit_autoexp.py` keeps monitoring unless `--no-monitor` is provided. A later invocation of `monitor_autoexp.py --session <plan_id>` rehydrates the manifest (by way of the session file), re-registers the persisted jobs with the SLURM client, and resumes monitoring without touching the backend. Completed sessions are ignored by default when using `--all`, but `--include-completed` can override this.
  - **Session discovery**: `scripts/manage_monitoring.py list/show/remove` remain the primary tooling for enumerating sessions; the manifest path stored in each JSON descriptor makes it trivial to restart monitoring even after long gaps.
- Cleanup behavior:
  - Session files persist for inspection even after jobs complete
- Manual cleanup by way of `scripts/manage_monitoring.py remove <session_id> [--force]`
- Cluster smoke test `scripts/tests/test_monitor_resume.py` plans, submits, interrupts, and resumes monitoring to validate the persisted state end-to-end.
- Management commands (all require `--monitoring-state-dir` parameter):
  - `manage_monitoring.py list`: Show all monitoring sessions
  - `manage_monitoring.py show <id>`: Display session details and active jobs
  - `manage_monitoring.py remove <id>`: Clean up session (with --force for active jobs)
  - `manage_monitoring.py dump-config <id>`: Extract full config from session

### Container Integration
- `scripts/run_autoexp_container.py` orchestrates the full flow:
  1. Execute `scripts/plan_autoexp.py` inside the container (render scripts + manifest).
  2. Execute `scripts/submit_autoexp.py --manifest …` on the host (submit + optional monitoring).
- `--manifest <path>` controls where the plan manifest is written/read (defaults to `<base_output_dir>/manifests/plan_<timestamp>.json`).
- `--no-submit` stops after planning; useful when subsequent actions (evaluation, conversions) will consume the manifest later.
  - `--no-monitor` submits jobs but exits immediately; monitoring can be resumed by way of `scripts/monitor_autoexp.py --session <plan_id>`.
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
- Monitoring config exposes `log_events` that match regex/substring patterns in SLURM logs and emit named events. Each log match materialises a `MonitorEvent` with optional typed state (`MonitorStateInterface`) and one or more actions (`BaseMonitorAction`) so downstream tooling can trigger follow-up work (for example, enqueue checkpoint conversion jobs or downstream evaluations). The standalone monitor CLI pre-registers generic events for errors, lifecycle transitions (pending → running → ended), completion markers, and checkpoint publication so downstream tooling can attach actions without backend knowledge.
- Monitor compares both SLURM log output and configured training output files; inactivity detection keys off the latest change across either source so freezes without log updates are still caught, mirroring the legacy `autoexperiment` behaviour.
- **Event Logging and Visibility**: The `MonitorController` logs all monitoring events, SLURM state transitions, signal detections, and action evaluations at INFO level to ensure full visibility into the restart system's behaviour. This includes:
  - Monitor outcomes (status, last_update, signal counts)
  - SLURM state transitions (PENDING → RUNNING → CANCELLED/FAILED/COMPLETED)
- Event detection with state/action payload
  - Action decisions (restart/commands/logs) with reason and attempt count
  - Job restart operations (old job_id → new job_id)
- **Mode Classification with Metadata**: When classifying job state into restart modes, the controller enriches the mode with contextual metadata:
  - `crash` mode for CANCELLED jobs includes `error_type: cancelled` and `subsystem: slurm` to enable selective event policies
  - `crash` mode for FAILED jobs includes `error_type: slurm_failure`
  - `timeout` mode includes `error_type: timeout` and reason codes
  - This metadata allows event policies to make informed decisions (for example, restart on CANCELLED but not on OOM)
- Define error modes:
  - `stall`: log unchanged for N seconds while job is `RUNNING` → restart with same plan, increment `stall_retries`.
  - `crash`: SLURM exits non-zero or is CANCELLED without completion marker → consult the configured `state_events` (typically bound to `RestartAction` guarded by metadata conditions).
  - `timeout`: job cancelled due to wall-clock (detected by way of SLURM reason codes) → restart with updated `time_limit` or gracefully stop depending on how `state_events` are configured.
  - `completed`: termination string or command detected → mark done, no restart.
- Event bindings live directly inside `log_events` and `state_events`: each action entry lists the conditions and the action to execute. No external policy registry. Inline actions run immediately; queued actions are materialised as JSON files under the action queue directory for out-of-band workers.
- Provide hooks for backend adjustments by emitting metadata that downstream actions can consume (for example, an action that reruns `run_autoexp.py` with revived overrides). SLURM templates should accept aggregated directive/env strings so cluster-specific SBATCH, launcher, and `srun` settings live entirely in config rather than templated code.

## Testing Strategy
- **Unit tests**: cover config parsing, sweep expansion semantics, SLURM template rendering, Megatron arg validation (use pared-down parser fixtures).
- Include unit tests ensuring compoconf parsing honors required/optional fields, registry instantiation, and nested backend selection. Add golden tests validating `dump_config` round-trips for sample YAMLs.
- **Integration tests**:
  - Build a fake SLURM layer (`tests/integration/fake_slurm_backend.py`) that simulates `sbatch`, `squeue`, `scontrol show job`, `scancel`, with controllable timelines and log file updates.
  - Scenario coverage:
    1. Happy path: multi-axis sweep → array submission → jobs complete.
    2. Stall detection: log stops updating, ensure restart triggered once, log mutated, job completes.
    3. Crash without matching restart binding → job marked failed, no rerun.
    4. Timeout with state-event binding that increases limit or aborts after retry budget.
    5. Manual `scancel` → CANCELLED state → restart triggered with proper logging.
    6. CUDA OOM in log → crash mode with `error_type: oom` → no restart (excluded).
    7. Resume monitoring after process restart (load persisted state, pick up existing job IDs).
  - Validate that sbatch templates include required placeholders and `--job-name` consistency.
  - **Event Logging Coverage**: Integration tests should verify that all monitoring events are logged:
    - SLURM state transitions are logged with old and new states
    - Log signal detection is logged with signal name and mode
    - Action decisions are logged with action, reason, and attempt count
    - Job restart operations are logged with old and new job IDs
- Provide fixtures for config YAMLs mirroring patterns from legacy repositories to avoid regressions.
- **Unit Test Coverage for Monitor**: Ensure unit tests cover:
  - CANCELLED job state → crash mode with `error_type: cancelled` metadata
  - Event-action decision based on metadata conditions (for example, restart only on `error_type=cancelled`)
  - Logging output contains expected transition and decision messages

## Tooling & Packaging Notes
- `pyproject.toml` with Poetry or Hatch for optional extras (`[project.optional-dependencies]` includes `megatron` entry).
- Re-export main CLI in `oellm_autoexp/__init__.py` for programmatic use.
- Document workflow in `README.md` (quickstart, example config) and `docs/` (architecture, backend guide).
- Ensure linting & type checking (ruff + mypy) to keep interfaces tight.
- Ship container helpers (for example, `scripts/run_autoexp_container.py`) so images can execute plan generation with all backend dependencies, while SLURM submission happens on the host. Support both unified (submit + monitor) and separated workflows (submit-only, monitor-only) to enable flexible deployment and resumable monitoring.

## TODO / Backlog

### Active TODOs
_None. Add next items here._

### Backlog (Done / Deferred)
- Improve provenance capture for container builds and job executions (commit hash, CLI args, git diff when dirty, resolved/unresolved configs, sanitized env). Stored inside container images and under each job's output directory. ✓ (`build_container_user.py` and `run_autoexp.py` now emit `_provenance` snapshots.)
- Simplify configuration loading by moving Hydra/OmegaConf normalization into a single module so contributors only learn one pathway before compoconf parsing. ✓ (Merged into `oellm_autoexp/config/loader.py`.)
- Extend the standalone `monitor` CLI with persisted state, real SLURM integration, and visibility into queued actions built from the primitives above. ✓ (Resume/tail/events/queue/actions subcommands + worker implemented.)
- Extend integration tests to cover log-event driven actions/restarts, checkpoint-gated evaluations, and container provenance. (Unit coverage exists; SLURM integration run still on the roadmap.)
- Add scheduler throttling (`max_jobs`, rate limits) and restart-policy variants expressed as actions (mutate launch args, swap checkpoints). ✓ (Submission throttling + restart adjustments implemented.)
- Replace the legacy `restart/` configs with inline event bindings driving `RestartAction`/`RunCommandAction`. ✓ (Removed `config/restart` and routed controller decisions through monitor-managed events/actions; the remaining work is described in the active TODO.)
- Complete `log_events` migration + naming cleanup. ✓ (Configs/docs/tests updated, `LogAction` replaces `LogMessageAction`, loader rejects `monitoring.log_signals` / `monitoring.policies`.)
- Array log/output provenance includes SLURM identifiers. ✓ (Monitor defaults now emit `%j`, job metadata carries `job_id`, and README/docs highlight the requirement.)
- Ship runnable follow-up configs/tests. ✓ (`megatron_checkpoint_eval` + new `megatron_followup` demonstrate queued `RunCommandAction`/`RunAutoexpAction`, with config tests ensuring validation errors are clear.)
- Simplify the action queue to `{state_dir}/session.actions/{event_id}/{action_id}.json` so only active actions exist on disk, and update the worker/tests to operate on the per-file layout. ✓
- Ship a concrete checkpoint-follow-up config (`monitoring/megatron_checkpoint_eval.yaml`) with `RunCommandAction` + `RunAutoexpAction`, plus config-parsing tests under `tests/config`. ✓
- Provide richer container assets (constraint tooling, requirements reduction) and document validation workflows. (Still pending; provenance work done, asset curation outstanding.)
- Document the event/condition/action primitives across README/docs (architecture diagrams, cookbook recipes). (Docs still to be updated.)
- Improve action queue UX. ✓ (`monitor_autoexp.py --cmd queue` now supports filtering, JSON inspection, and retries; README/TESTING document the workflow.)
- Add end-to-end coverage for queued `RunAutoexpAction`. ✓ (Integration test now writes checkpoint logs, inspects rendered queue entries, and simulates worker success/failure with the fake SLURM client.)

## Core Automation Primitives

Simplicity = few, composable primitives. The monitor is nothing more than “detect event → persist state → run bound actions (optionally gated by conditions)”. Every feature request should slot into one of these concepts.

### High-Level Flow
1. **Detection** — Monitors (currently `SlurmLogMonitor`) read job state + log/output files and emit `MonitorEvent` objects whenever a rule hits.
2. **Event Record** — `MonitorController` persists or updates an `EventRecord` (status `triggered`, `count += 1`, metadata merged).
3. **Binding Resolution** — The event configuration already embeds its `actions`. No indirection layers or policy registries.
4. **Conditions** — Each binding can list `conditions`. Blocking conditions (for example, `FileExistsCondition(blocking=true)`) set the event to `pending` and keep re-checking. Non-blocking conditions simply skip the action.
5. **Action Execution** — Inline actions run immediately and update the event status. Queued actions materialise as single JSON files under `{session}/actions/{event_id}/{action_id}.json` so the queue is inspectable and corruption stays isolated.
6. **State Updates** — Actions call `update_event` with `success`, `retry`, or `failed`. The controller also updates job attempts when restarts occur.

### Event Specs & Records
- `LogEventConfig`: `name`, `pattern`, `pattern_type` (`regex` | `substring`), optional `state`, optional `metadata`, optional `extract_groups`, and `actions`. Configs must specify either a state or metadata so downstream logic has a payload to work with. `LogSignalConfig` exists only as a temporary alias for backwards compatibility.
- `StateEventConfig`: synthetic events triggered by SLURM transitions or controller heuristics (stall, crash, timeout, success, manual cancellation). Shares the same `state` + `metadata` + `actions` schema.
- `EventRecord`: persisted structure with `event_id`, `name`, `source`, `status` (`triggered`, `pending`, `processed`, `failed`), `count`, `metadata`, `payload`, timestamps, and per-action notes. Re-emitting an event with the same logical identifier increments `count` and updates `last_seen`.

### Conditions
- Interface: `ConditionInterface.check(context) -> ConditionResult` returning `{status: "pass" | "wait" | "fail", reason}`.
- Context merges job metadata (SLURM IDs, attempts, log paths), event metadata/payload, manifest hints (output dirs, workspace), and sanitized env.
- Built-ins: `MetadataCondition`, `FileExistsCondition`, `GlobExistsCondition`, `MaxAttemptsCondition`, `ElapsedTimeCondition`, `CommandCondition`, `AlwaysTrueCondition`.
- **Run-level preliminaries**: configuration may specify `start_condition_cmd` (or future `start_condition`) executed before the job launches. This covers Option B from the earlier event-queueing proposal where a run waits for a prerequisite (checkpoint, dataset upload) before starting.
- **Event-level gates**: each `EventActionConfig` includes `conditions`. All conditions share the same interface; blocking ones bubble up `wait` so the controller keeps the event in `pending` without spawning duplicate actions.

### Actions
- Interface: `ActionInterface.execute(context) -> ActionResult` where result contains `status` (`success`, `retry`, `failed`), `message`, and optional `metadata`.
- Implementations shipped now: `RestartAction`, `RunCommandAction`, `RunAutoexpAction`, `LogAction` (successor of `ErrorNoteAction`/`LogMessageAction`), and `PublishEventAction`.
- Actions control their own event-status updates by way of `update_event(event, result)`, so adding a new action never requires controller tweaks.
- Inline vs queued: an action can request `mode: inline` (default) or `mode: queue`. Queued actions create `{session.actions}/{event_id}/{action_id}.json` entries which workers execute by way of CLI commands; inline actions finish inside the monitor cycle. The queue must stay short, so we delete each JSON file as soon as it reaches a terminal status.

### Action Queue Layout
- Path: `{monitoring_state_dir}/session.actions/{event_id}/{action_id}.json`.
- File content: `{ "event_id": "...", "action": "RunCommandAction", "config": {...}, "job_metadata": {...}, "status": "queued", "attempt": N }`.
- Because each action is a standalone file, `ls` + `cat` is enough to debug the system. Partial writes can be retried by simply rewriting the single file. No monolithic queue files exist.

### Event + Condition-driven Runs
- `RunAutoexpAction` can be gated by `FileExistsCondition(blocking=true)` to wait for checkpoints before scheduling evaluations (Option B). The same primitive can also block `RunCommandAction` for cooldown/capture tasks. No bespoke “preliminary action” concept is needed beyond the blocking conditions.
- Waiting runs record their condition/progress inside the action JSON, making it easy to see “waiting for /ckpt/iter_1000” at a glance.

### Array Sub-Job Reruns
- `run_autoexp.py --array-subset …` reruns individual indices. To keep outputs unique and inspectable, log/output templates must include `%j` (single job) or `%A_%a` (array). Also store `{job_id, array_master_id, array_task_id}` inside each event’s metadata so operators can correlate attempts.
- Tests must assert that reruns generate new log files, that monitor events receive the new IDs, and that actions referencing `{job_id}` render the fresh identifier.

## Implementation Plan

1. **Event Emission** — Refactor the watcher to emit `MonitorEvent` objects only, register detectors by way of compoconf, and persist per-event counters/last-seen timestamps.
2. **Condition Engine** — Implement the condition interface, add blocking/non-blocking modes, and cover built-ins (MaxAttempts, FileExists, etc.) with unit tests (including `waiting` semantics).
3. **Action Catalog** — Implement the action interface + catalog; ensure RunAutoexpAction writes provenance snapshots and sanitizes env. Unit-test restart attempt increments, command failure handling, and event-status updates.
4. **Controller Simplification** — Keep all bindings inline with the events. `MonitorController` should take the emitted event, instantiate its bindings, run conditions, and either execute inline actions or enqueue queued ones. No separate policy registry; the controller just iterates over the bindings already attached to the event.
5. **Action Queue & Worker** — Keep the queue as `{state_dir}/session.actions/{event_id}/{action_id}.json`, one file per action. Ship `scripts/manage_actions.py` (or reuse `monitor_autoexp.py --cmd=actions`) to list/retry/delete items. No global JSONL/Redis queue.
6. **Provenance Capture** — Container build scripts emit `/oellm/provenance.json`; `run_autoexp.py` writes `full_config.yaml` (resolved/unresolved configs, overrides, sanitized env, git metadata, container digest), and also a resolved `config.yaml` (or more for sweeps) that can be reloaded directly with `run_autoexp.py` as `--config-ref`.
7. **Config Loader Simplification** — Move normalization into `oellm_autoexp/config/normalize.py`, document the single pipeline.
8. **Monitor CLI Enhancements** — Subcommands (`--cmd=resume`, `--cmd=tail`, `--cmd=events`, `--cmd=queue`, `--cmd=actions`) show live event counts, condition status, and queued-action backlog. Integrate with real SLURM clients when available. There are still general omegaconf/hydra-like overrides possible as for `run_autoexp.py`.
9. **Testing Strategy** — Unit tests for each condition/action, queue serialization, and provenance writer. Integration tests for stall→RestartAction, checkpoint+FileExists→RunAutoexpAction, and cooldown-gated RunCommandAction. CLI tests ensure `--array-subset` reruns only targeted indices with unique logs.
10. **Documentation** — Update `docs/architecture.md` with Event→Condition→Action diagrams and provide cookbook recipes (simple restart, cooldown restart, checkpoint-triggered evaluation, chained autoexp run).

### Detailed Plans (Upcoming Work)

#### Checkpoint-Triggered Follow-ups & Preliminary Conditions
1. Define `PreliminaryConditionConfig` schema reused by both action bindings and run manifests (for example, `wait_for_checkpoint`).
2. Example config:
   - Event: `checkpoint_ready`
   - Preliminary: `FileExistsCondition(path="${event.metadata.checkpoint_path}")`
   - Actions: `[RunAutoexpAction(config_ref="configs/eval.yaml", overrides=["checkpoint=${event.metadata.checkpoint_path}"]), LogAction(message="Eval scheduled")]`
   - Optional follow-up: `RunCommandAction(command="python cooldown.py --ckpt ...")`
3. Tests:
   - Config parsing ensures compoconf surfaces missing `config_ref`.
   - Monitor unit test fakes a checkpoint event and asserts the action queue receives a `RunAutoexpAction`.
   - Condition test verifies the preliminary condition blocks until the checkpoint path exists.

#### Config + CLI Polish
1. Update all configs to `log_events`, annotate ambiguous sections with `# UNCLEAR: ...`.
2. Add a sample config that schedules another autoexp run or shell command upon manual event emission (demonstrates reusable primitives).
3. Expand `tests/config` to load each shipping config and ensure it fails fast on invalid references (bad action names, missing metadata, etc.).
