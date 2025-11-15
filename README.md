# OELLM Auto Experimentation Tool

Work-in-progress monorepo consolidating sweep planning, SLURM submission, container workflows, and monitoring for OpenEuroLLM projects. The Megatron backend reuses Megatron-LM's parser so all sweeps stay compatible with upstream flags. See `SPEC.md` for the detailed architecture and outstanding work.

FEEL FREE TO ADD PRs if you find a bug or inconsistent behavior!

## Environment setup

Please add the following environment variables to your `.bashrc`, as they are used to store container images, access slurm, load data etc. Otherwise you need to override the respective parts in the config yamls.

```bash
export SLURM_ACCOUNT="myproject"
export DATA_ACCOUNT="myproject"
export SLURM_PARTITION="booster"
export SLURM_PARTITION_DEBUG="develbooster"
export SLURM_QOS="normal"
export SLURM_QOS_DEBUG="normal"

export WORK=/p/scratch/$DATA_ACCOUNT/poeppel1
export DATA_DIR="$WORK/data"
export OUTPUT_DIR="$WORK/output"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export HF_DATASETS_CACHE="$WORK/data/cache"
export HF_HOME="$WORK/cache"
export APPTAINER_CACHEDIR="$WORK/.apptainer/cache"
export APPTAINER_TMPDIR="$WORK/.tmp"
export CONTAINER_CACHE_DIR="$WORK/container_cache"
export ARCH=$(uname -m)
```

## Installation

```bash
git clone https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
pip install -e .
git submodule update --init --recursive
```

Create your container to run things then, see `container/`.


### Lumi

```bash

module load cray-python/3.11.7 #Pytorch module wont work because it is a singularity container wrapper

# in your $HOME folder
git clone https://github.com/sfantao/rccl-tuner
cd rccl-tuner
make Makefile.lumi
# creates the librccl-tuner.so file needed for the interconnect

git clone https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
python3 -m venv .venv --system-site-packages
pip install -e .[dev]
git submodule update --init --recursive
#TODO: Do a replacement for venv; squashfs, container for launching or easybuild module
```

When running scripts directly from the checkout (without installing the package), make sure Python can see the repository and submodules:

```bash
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/submodules/Megatron-LM"
```



## Script Overview

| Script | Purpose |
| --- | --- |
| `scripts/run_autoexp.py` | One-shot plan → submit → (optional) monitor on the local host. |
| `scripts/plan_autoexp.py` | Render SBATCH scripts and record a manifest (defaults to `<base_output_dir>/manifests`). |
| `scripts/submit_autoexp.py` | Submit/monitor jobs defined by a manifest. |
| `scripts/monitor_autoexp.py` | Resume monitoring by way of session JSONs (or legacy manifest paths). |
| `scripts/run_autoexp_container.py` | Run planning inside a container, submission/monitoring on the host. |
| `scripts/manage_monitoring.py` | List, inspect, and remove monitoring sessions on disk. |
| `scripts/tests/test_monitor_resume.py` | Cluster-side smoke test covering monitor interruption and resume. |

## Configuration Primer

- `config/project/*.yaml` sets descriptive metadata, default output locations, and optional persistent state directories.
- `config/slurm/*.yaml` describes cluster-specific `sbatch` overrides, launcher prefixes, optional `slurm.array` toggles, SLURM client implementation, node-level environment exports, and `launcher_env_passthrough` to forward those exports into containers.
- `config/backend/megatron.yaml` selects the `AutoMegatronBackend`, which expands convenience arguments (`train_tokens`, `lr_decay_fraction`, …) into canonical Megatron flags before validation.
- `config/monitoring/*.yaml` defines log templates, inactivity thresholds, optional start conditions, and log-signal → action mappings.
- Monitoring configs (`config/monitoring/*.yaml`) now own restart + follow-up behaviour through `log_events` + `state_events` with inline action bindings.

Hydra overrides (`key=value`) let you combine these building blocks per run.

## Usage Recipes

### 1. Single Megatron Job

```bash
# One-shot plan+submit+monitor (manifest stored under ./outputs/manifests)
python scripts/run_autoexp.py project=default

# Preview without submission
python scripts/run_autoexp.py project=default --dry-run

# Submit but exit immediately (monitor later)
python scripts/run_autoexp.py project=default --no-monitor

# Resume monitoring later (pick the desired session from ./monitor)
python scripts/monitor_autoexp.py --session <plan_id>
# Legacy manifest path still works
python scripts/monitor_autoexp.py --manifest outputs/manifests/<plan>.json

# Manual plan/submit flow for advanced control
python scripts/plan_autoexp.py --manifest outputs/manifests/custom_plan.json project=default
python scripts/submit_autoexp.py --manifest outputs/manifests/custom_plan.json --use-fake-slurm --dry-run

# Inspect the rendered launch command from the manifest
python scripts/run_sweep_entry.py --sweep output/sweep.json --index 0 --dry-run

# Execute a single sweep entry locally (no SLURM)
python scripts/run_sweep_entry.py --sweep output/sweep.json --index 0
```

### Monitoring Resume Workflow

- Each submission writes a monitoring session file under
  `<monitoring_state_dir>/<plan_id>.json`. The controller now records
  `resolved_log_path`, `last_monitor_state`, `last_slurm_state`, and a
  `last_updated` timestamp so resumed sessions no longer depend on `%j` or
  array placeholders.
- `monitor_autoexp.py` can attach directly to a session file by way of
  `--session <plan_id>` (or `--session path/to/session.json`). It re-registers
  persisted jobs with the SLURM client before the first monitoring cycle, which
  restores accurate `squeue()` snapshots after an interruption.
- Pass `--all` to run one monitor per session concurrently (useful when several
  experiments are active). Completed sessions (no active jobs) are skipped by
  default; add `--include-completed` if you need to re-process archived runs.
- To rehearse the whole flow on a login node, run
  `python scripts/tests/test_monitor_resume.py`. The harness plans, submits,
  interrupts the monitor with SIGINT, and verifies that a second monitor run
  reports restored jobs and keeps concrete log paths in the session file.

Run `ls outputs/manifests` to inspect the manifests created by the helper scripts.

`scripts/run_sweep_entry.py` rehydrates the stored command, exports the configured environment, and can be embedded inside notebooks or custom launchers. The manifest produced in step 1 keeps the resolved configuration, scripts, and monitoring metadata for later reuse.

### 2. Megatron Training (Direct Python versus Container)

**Direct (no container):**

```bash
python scripts/run_autoexp.py \
  -C config \
  --config-ref experiments/megatron_jupiter_speed_test \
  container=none \
  --use-fake-slurm
```

The in-memory SLURM client mimics submission/monitoring without touching a real scheduler. Drop `--use-fake-slurm` once a real client is configured under `config/slurm`.

**Container (Recommended):**

```bash
# Build an Apptainer/Singularity image tailored for Megatron
./container/build_container.sh --backend megatron --definition MegatronTraining --output ./artifacts

# Configure container in your config (for example, config/container/juwels.yaml):
# image: ${oc.env:CONTAINER_CACHE_DIR}/MegatronTraining_x86_64.sif
# runtime: singularity

# Run with auto-detected container from config (recommended)
python scripts/run_autoexp_container.py \
  --config-ref experiments/megatron_jupiter_speed_test \
  container=juwels

# Or explicitly specify container image
python scripts/run_autoexp_container.py \
    --image ./artifacts/MegatronTraining_$(uname -m).sif \
    --config-ref autoexp \
    -C config \
    project=juwels

# Dry-run (render manifest only)
python scripts/run_autoexp_container.py \
  --config-ref autoexp \
  -C config \
  --no-submit \
  project=juwels_speed_test

# Execute with fake SLURM client
python scripts/run_autoexp_container.py \
  --config-ref autoexp \
  -C config \
  --use-fake-slurm \
  project=juwels_speed_test
```

**Key Points:**
- `run_autoexp_container.py` plans inside the container (render scripts + manifest) and submits/monitors on the host.
- By default manifests are written under `<project.base_output_dir>/manifests/plan_<timestamp>_<token>.json`, letting you queue multiple runs without overwriting state.
- Use `--no-submit` to render only, `--no-monitor` to submit and exit immediately, or `--monitor-only` to attach to an existing manifest.
- Plan generation happens in container (needs Megatron), SLURM submission on host (needs sbatch)
- Setting `slurm.launcher_env_passthrough: true` forwards `slurm.env` keys into the launcher by way of `--env VAR=$VAR`


### 3. Megatron Training with SLURM

**Recommended approach (using container wrapper):**

```bash
# Submit with auto-detected container and monitoring
python scripts/run_autoexp_container.py --config-ref autoexp -C config \
  project=juwels slurm=juwels container=juwels

# Submit without monitoring (monitor separately later)
python scripts/run_autoexp_container.py --config-ref autoexp -C config \
  --no-monitor \
  project=juwels slurm=juwels container=juwels

# Resume monitoring later (reads the manifest and session state)
python scripts/monitor_autoexp.py --manifest outputs/manifests/<plan>.json

# Inspect available sessions
python scripts/manage_monitoring.py list --monitoring-state-dir output/monitoring_state
```
### 4. Adapting Behaviour for Errors & Timeouts

Monitoring behaviour is now expressed directly on `log_events` and `state_events` without a separate policy file. Each event can declare inline `EventAction` bindings (with optional conditions) that execute inline or enqueue JSON payloads under the per-session `actions/` directory.

```yaml
monitoring:
  class_name: SlurmLogMonitor
  log_events:
    - name: cuda_oom
      pattern: "CUDA out of memory"
      pattern_type: substring
      state:
        class_name: CrashState
      metadata:
        error_type: oom
      actions:
        - class_name: EventAction
          conditions:
            - class_name: MaxAttemptsCondition
              max_attempts: 2
          action:
            class_name: RestartAction
            reason: "retry oom once"
  state_events:
    - name: success
      actions:
        - class_name: EventAction
          action:
            class_name: LogAction
            message: "run_finished"
```

Conditions have a shared interface (`ConditionInterface.check`) and can block (`FileExistsCondition(blocking=true)`), filter (`MetadataCondition`), or limit retries (`MaxAttemptsCondition`). No extra layers: the monitor sees a `cuda_oom` event, evaluates the bindings, and either restarts the job or leaves the event in `pending` for later.

### 5. Trigger Post-processing from Log Events

```yaml
monitoring:
  class_name: SlurmLogMonitor
  log_events:
    - name: checkpoint_ready
      pattern: "Checkpoint saved: (?P<checkpoint_path>.+\.pt)"
      pattern_type: regex
      metadata:
        kind: checkpoint
      actions:
        - class_name: EventAction
          mode: queue
          conditions:
            - class_name: FileExistsCondition
              path: "{checkpoint_path}"
              blocking: true
          action:
            class_name: RunCommandAction
            command:
              - python
              - scripts/run_downstream_eval.py
              - "--checkpoint"
              - "{checkpoint_path}"
```

When the pattern matches, `MonitorController` records the event and materialises queued actions as `{monitoring_state_dir}/session.actions/<event_id>/<action_id>.json`. Built-in action types live under `oellm_autoexp.monitor.actions`:

- `RunCommandAction`: run arbitrary scripts (downstream evaluation, conversions, cooldown).
- `RestartAction`: request a restart (controller increments attempts and resubmits).
- `RunAutoexpAction`: call `scripts/run_autoexp.py` with stored configs/overrides.
- `LogAction`: attach a human-readable note to the event history.
- `PublishEventAction`: emit follow-up events for chained workflows.

Actions receive monitor metadata (regex groups like `{checkpoint_path}`, job metadata such as `{output_dir}` and `{job_id}`) so templates stay simple. Custom actions inherit from `BaseMonitorAction` and register by way of compoconf.

Need a ready-to-use preset?

- `config/monitoring/megatron_checkpoint_eval.yaml` mirrors the production monitor but adds queued `RunCommandAction` + `RunAutoexpAction` entries whenever a checkpoint lands.
- `config/monitoring/megatron_followup.yaml` adds state-event bindings that schedule a cooldown command and a follow-up autoexp invocation after a clean `success`.

Select them with `-o monitoring=megatron_checkpoint_eval` (or `megatron_followup`) and point a lightweight worker at the `actions/` directory to execute queued payloads.

To trigger downstream evaluations automatically upon checkpoints:

1. Add the `RunCommandAction` configuration shown above to your monitoring config (either in Hydra YAML or as a manifest override).
2. Run planning/submission as usual. The monitor loop will emit JSON payloads under `outputs/<run>/actions/` with the rendered command.
3. Point a lightweight worker (or CI job) at that folder to execute queued evaluations. Each payload is written once, so duplicate execution is easy to avoid.

To exercise the monitoring pipeline locally:

```bash
PYTHONPATH=$PYTHONPATH:. pytest tests/integration/test_workflow_scripts.py::test_plan_submit_monitor_fake_slurm \
  --maxfail=1

# Append simulated CUDA errors to the generated SLURM log and rerun:
python scripts/monitor_autoexp.py --session <plan_id> --use-fake-slurm
echo "CUDA out of memory. Tried to allocate 4GiB" >> output/logs/<job>/slurm-<id>.out
python scripts/monitor_autoexp.py --session <plan_id> --use-fake-slurm
```

The rerun verifies that crash signals (like CUDA OOM) are detected and the restart bindings fire as expected. For real-cluster validation you can run the same monitoring CLI without `--use-fake-slurm`; the compoconf registries ensure only the necessary Megatron configuration modules are imported on the host.

### 7. Integrating a New Backend

```python
from compoconf import register


from oellm_autoexp.backends.base import BaseBackend, BackendJobSpec, LaunchCommand

@register
class ExampleBackend(BaseBackend):
    config: ExampleConfig

    def validate(self, spec: BackendJobSpec) -> None:
        ...

    def build_launch_command(self, spec: BackendJobSpec) -> LaunchCommand:
        return LaunchCommand(argv=["python", "train.py"], environment={})
```

Add a Hydra group under `config/backend/example.yaml` and select it with `-o backend=example`. The orchestration stack (sweeps, monitoring, restart logic) will work unchanged.

### 8. JUWELS LLaMA-1.8B Speed Test (Singularity)
- Ensure `CONTAINER_CACHE_DIR` points to the folder holding the container image
- Load the JUWELS environment exporting `SLURM_ACCOUNT` and `SLURM_PARTITION_DEBUG`
- Update the `templates/juwels.sbatch` module list if your site configuration differs
- Configure `config/container/juwels.yaml` with your container image path

**Recommended (using container wrapper):**

```bash
python scripts/run_autoexp_container.py --config-ref autoexp -C config \
  project=juwels_speed_test \
  slurm=juwels \
  container=juwels \
  backend=megatron_speed_test_juwels \
  sweep=speed_test_juwels \
  --fake-submit
```

## Migration Guides

### autoexperiment ➜ oellm-autoexp
- Swap `autoexperiment build|run|build_and_run` for the helper scripts (`plan_autoexp.py`, `submit_autoexp.py`, `monitor_autoexp.py`, or `run_autoexp.py` for the combined flow).
- Legacy single-file configs map into explicit `project`, `slurm`, `monitoring`, and `sweep` blocks. The bundled `config/autoexp.yaml` composes the same defaults automatically.
- Job recovery data now lives under `.oellm-autoexp/`; keep sweep-driven names deterministic by way of `sweep.name_template` instead of hand-crafted `{name}` strings.

| autoexperiment concept | oellm-autoexp equivalent | Notes |
| --- | --- | --- |
| `template` | `slurm.template_path` | SBATCH template rendered per job. |
| `sbatch_script` | `slurm.script_dir` | Output directory for rendered scripts. |
| `cmd` | `slurm.submit_cmd` | Still defaults to `sbatch`; supports arrays and launch wrappers. |
| `output_file` | `monitoring.log_path_template` | Defaults to `"{output_dir}/slurm-%j.out"` so reruns never overwrite prior logs. |
| `start_condition_cmd` | `monitoring.start_condition_cmd` | Checked before submission until it exits 0. |
| `check_interval_secs` | `monitoring.poll_interval_seconds` | Poll cadence for freeze detection. |
| `termination_str` | `monitoring.termination_string` | Ends restart loops when matched. |
| `termination_cmd` | `monitoring.termination_command` | Optional shell predicate for completion. |
| `name` | `sweep.name_template` | Drives `--job-name` and persisted state keys. |
| nested sweep dictionaries | `sweep.axes` / `sweep.base_values` | Explicit cartesian products with shared defaults. |

**Before** (`autoexperiment`):

```yaml
template: template.sbatch
sbatch_script: "sbatch/{name}.sbatch"
cmd: "sbatch {sbatch_script}"
output_file: "{logs}/{name}/slurm-%j.out"
check_interval_secs: 600
termination_str: "Eval Epoch: {epochs}"
```

**After** (`oellm-autoexp`):

```yaml
slurm:
  template_path: templates/base.sbatch
  script_dir: generated/sbatch
  submit_cmd: sbatch
monitoring:
  log_path_template: "{output_dir}/slurm-%j.out"
  poll_interval_seconds: 600
  termination_string: "Eval Epoch: {epochs}"
sweep:
  name_template: "${project.name}_{index}"
  axes:
    epochs: [1]
```

### oellm_pretrain ➜ oellm-autoexp
- Replace the single `oellm-pretrain CONFIG.yaml` command with `oellm-autoexp submit CONFIG.yaml`; use `plan` for inspection and `monitor` to reattach later.
- Split the old `sbatch_args` between `project` (naming, base output directory) and `slurm` (template path, launcher overrides, log root).
- Keep Megatron flags under the `megatron` block; sweeps reference them by way of `sweep.axes`, letting the orchestrator validate every flag against Megatron-LM's parser.

| oellm_pretrain concept | oellm-autoexp location | Notes |
| --- | --- | --- |
| `sbatch_args.out_dir` | `project.base_output_dir` (root) + `slurm.log_dir` | Separate run outputs from log location. |
| `sbatch_args.job_name` | `project.name` / `sweep.name_template` | Controls job naming and recovery keys. |
| implicit job arrays | `slurm.array` | Toggle arrays explicitly per config. |
| `sweep_args` | `sweep.axes` | Declare cartesian products directly. |
| `megatron_args` scalars | `megatron.<arg>` | Defaults shared by all sweep points. |
| `megatron_args` lists | `sweep.axes` with `megatron.<arg>` | Keeps sweep logic centralized. |
| always writing `sweep.json` | `sweep.store_sweep_json` | Enable/disable persisted sweep manifests. |

**Before** (`oellm_pretrain`):

```yaml
sbatch_args:
  out_dir: /path/to/out
  job_name: oellm_test

sweep_args: [lr]

megatron_args:
  lr: [0.0003, 0.0001]
  global_batch_size: 1024
  seq_length: 2048
```

**After** (`oellm-autoexp`):

```yaml
defaults:
  - project: default
  - backend: megatron
  - slurm: base
  - monitoring: default
  - restart: default
  - scheduler: default
  - _self_

project:
  name: oellm_test
  base_output_dir: /path/to/out

slurm:
  log_dir: ${project.base_output_dir}/logs
  array: true

sweep:
  axes:
    megatron.lr: [0.0003, 0.0001]

megatron:
  global_batch_size: 1024
  seq_length: 2048
```

### megatron-train ➜ oellm-autoexp
- The Hydra defaults now bundle `project`, `backend`, `slurm`, `monitoring`, `restart`, and `scheduler`; you no longer need separate `env`, `launcher`, or `srun` groups unless you add overrides in those sections.
- `extract_hydra.py` bootstrap logic is covered by `oellm_autoexp/config/resolvers.py`, so Megatron's parser metadata is available automatically.
- Cluster launch settings live under the shared `slurm` namespace, making container wrappers, env exports, and `srun` options backend-agnostic.

| megatron-train field | oellm-autoexp field | Notes |
| --- | --- | --- |
| `experiment_name` | `project.name` / `sweep.name_template` | Job naming + persisted state. |
| `output_dir` | `project.base_output_dir` | Per-run directories derived from this root. |
| `slurm.template` | `slurm.template_path` | SBATCH template reference. |
| `launcher.cmd` | `slurm.launcher_cmd` | Prefix command (for example, container runtime). |
| `srun.opts` | `slurm.srun_opts` | Additional `srun` arguments. |
| `env.*` | `slurm.env` | Environment variables exported for each job. |
| Megatron Hydra group | `backend: megatron/...` + `megatron` block | Centralizes defaults and overrides. |
| manual resolver registration | Built-in in `config/resolvers.py` | `oc.*` helpers available out of the box. |

**Before** (`megatron-train`):

```yaml
defaults:
  - megatron: base
  - slurm: base
  - env: base
  - launcher: base
  - srun: base
  - _self_

experiment_name: debug_${oc.select:megatron.aux.model_name,""}
output_dir: ${oc.env:OUTPUT_DIR,"."}/megatron_${oc.env:SUBMIT_TIMESTAMP,${.timestamp}}
```

**After** (`oellm-autoexp`):

```yaml
defaults:
  - project: default
  - backend: megatron/base
  - slurm: base
  - monitoring: default
  - restart: default
  - scheduler: default
  - _self_

project:
  name: debug_${megatron.aux.model_name}
  base_output_dir: ${oc.env:OUTPUT_DIR,"."}

slurm:
  template_path: templates/megatron_array.sbatch
  launcher_cmd: python
  srun_opts: "--cpu-bind=none"
  environment:
    HF_HUB_OFFLINE: 1

megatron:
  aux:
    model_name: ${oc.env:MODEL_NAME,""}
```

## State Persistence and Monitoring Sessions

- Each submission creates a monitoring session file in **visible location**: `<base_output_dir>/monitoring_state/<session_id>.json`
- Session files contain:
  - Full resolved configuration for reproducibility
  - Job state (job IDs, names, attempts, restart history)
  - Project metadata (name, creation timestamp)
- Monitoring workflows:
- **List sessions**: `python scripts/manage_monitoring.py list` (add `--include-completed` to show archived sessions)
- **Show session details**: `python scripts/manage_monitoring.py show <session_id>`
    - The session JSON now records `manifest_path`, making it easy to resume monitoring.
  - **Monitor from session**: `python scripts/monitor_autoexp.py --session <plan_id>`
  - **Remove session**: `python scripts/manage_monitoring.py remove <session_id> [--force]`
- Session files persist even after jobs complete, enabling inspection and re-monitoring
- The visible location (`monitor/` by default, or `<base_output_dir>/monitoring_state/`) makes sessions easy to discover and manage

## Diagnostics & Testing

- `pytest` runs unit + integration suites (fake SLURM, Megatron parser, sweep expansion, monitoring).
- `python scripts/submit_autoexp.py --manifest outputs/manifests/<plan>.json --dry-run` previews scripts, sweep metadata, and array scripts without launching jobs.

## Containers & Validation

- `container/build_container.sh --help` lists backend-specific definition files (currently `megatron`).
- `scripts/run_megatron_container.py` wraps Apptainer/Singularity execution with dry-run and fake-submit toggles.
- `scripts/run_sweep_entry.py` doubles as the SLURM array entrypoint and a convenient local debugger for sweep entries.

Refer to `SPEC.md` for ongoing work such as real SLURM submission paths, scheduler throttling, and orchestrator persistence.
