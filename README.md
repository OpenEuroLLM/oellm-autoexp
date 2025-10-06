# OELLM Auto Experimentation Tool

Work-in-progress monorepo consolidating sweep planning, SLURM submission, container workflows, and monitoring for OpenEuroLLM projects. The Megatron backend reuses Megatron-LM's parser so all sweeps stay compatible with upstream flags. See `SPEC.md` for the detailed architecture and outstanding work.

## Installation

```bash
git clone https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
pip install -e .[dev,megatron]
git submodule update --init --recursive
```

## Configuration Primer

- `config/project/*.yaml` sets descriptive metadata, default output locations, and optional persistent state directories.
- `config/slurm/*.yaml` describes cluster-specific `sbatch` overrides, launcher prefixes, optional `slurm.array` toggles, SLURM client implementation, node-level environment exports, and `launcher_env_passthrough` to forward those exports into containers.
- `config/backend/megatron.yaml` selects the `AutoMegatronBackend`, which expands convenience arguments (`train_tokens`, `lr_decay_fraction`, …) into canonical Megatron flags before validation.
- `config/monitoring/*.yaml` defines log templates, inactivity thresholds, optional start conditions, and log-signal → action mappings.
- `config/restart/*.yaml` contains restart policy definitions keyed by error mode (`stall`, `timeout`, `crash`, `success`).

Hydra overrides (`-o key=value`) let you combine these building blocks per run.

## Usage Recipes

### 1. Single Megatron Job Without SLURM

```bash
# Render scripts + sweep metadata without submitting
python scripts/run_autoexp.py autoexp -C config -o project=default --dry-run

# Inspect the launch command for the first sweep entry
python scripts/run_sweep_entry.py --sweep outputs/sweep.json --index 0 --dry-run

# Execute locally (no SLURM)
python scripts/run_sweep_entry.py --sweep outputs/sweep.json --index 0
```

`scripts/run_sweep_entry.py` rehydrates the stored command, exports the configured environment, and can be embedded inside notebooks or custom launchers.

### 2. Megatron Training (Direct Python vs. Container)

**Direct:**

```bash
python scripts/run_autoexp.py autoexp -C config -o project=default --use-fake-slurm
```

The in-memory SLURM client mimics submission/monitoring without touching a real scheduler. Drop `--use-fake-slurm` once a real client is configured under `config/slurm`.

**Container:**

```bash
# Build an Apptainer/Singularity image tailored for Megatron
./container/build_container.sh --backend megatron --definition MegatronTraining --output ./artifacts

# Dry-run inside the container (render scripts only)
python scripts/run_megatron_container.py --image ./artifacts/MegatronTraining_$(uname -m).sif \
    --config-ref autoexp -C config -o project=juwels --dry-run

# Execute with the fake SLURM client from inside the container
python scripts/run_megatron_container.py --image ./artifacts/MegatronTraining_$(uname -m).sif \
    --config-ref autoexp -C config -o project=juwels --fake-submit
```

Here `slurm.launcher_cmd` is treated as a prefix (e.g. `apptainer exec ...`), while the backend command is rendered separately. Setting `slurm.launcher_env_passthrough: true` forwards `slurm.env` keys into the launcher via `--env VAR=$VAR`.


### 3. Megatron Training with SLURM

```bash
# Plan and submit using the bundled hydra groups (JUWELS example)
oellm-autoexp plan autoexp --config-dir config -o project=juwels -o slurm=juwels
oellm-autoexp submit autoexp --config-dir config -o project=juwels -o slurm=juwels --fake

# Remove --fake once a real SLURM client is wired up
oellm-autoexp submit autoexp --config-dir config -o project=juwels -o slurm=juwels
```

Scripts land in `slurm.script_dir`, logs in `slurm.log_dir`, and `outputs/sweep.json` documents the rendered sweep.

### 4. Megatron Sweep as a SLURM Job Array

```bash
oellm-autoexp submit autoexp --config-dir config \
  -o project=juwels -o slurm=juwels -o slurm.array=true --fake
```

When `slurm.array=true`, `render_scripts` produces:

- `outputs/sweep.json` (one entry per sweep point).
- An aggregated array script (stored in `slurm.script_dir`) that dispatches via `scripts/run_sweep_entry.py` and `$SLURM_ARRAY_TASK_ID`.
- Metadata so `submit_jobs` routes through `submit_array` when the chosen SLURM client supports it (the bundled `FakeSlurmClient` does).

### 5. Adapting Behaviour for Errors & Timeouts

Restart policies are pluggable through compoconf:

```yaml
restart_policies:
  - mode: stall
    implementation:
      class_name: AlwaysRestartPolicy
      max_retries: 2
  - mode: timeout
    implementation:
      class_name: AdjustTimeLimitPolicy
      minutes: 30
  - mode: crash
    implementation:
      class_name: NoRestartPolicy
      message: "manual intervention required"
```

Combine policy decisions with monitoring metadata (inactivity thresholds, SLURM states, log signals) to enforce the right recovery behaviour.

### 6. Trigger Post-processing from Log Signals

```yaml
monitoring:
  class_name: SlurmLogMonitor
  log_signals:
    - name: checkpoint_ready
      pattern: "Checkpoint saved: (?P<path>.+\.pt)"
      pattern_type: regex
      action: convert_checkpoint
      metadata:
        kind: checkpoint
```

```bash
oellm-autoexp monitor autoexp -C config --fake \
  --log train=outputs/logs/train.out \
  --action convert_checkpoint='python tools/convert.py {checkpoint_path}'
```

Actions are queued whenever the pattern appears, enabling seamless hand-offs to evaluation or conversion scripts.

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
- Ensure `CONTAINER_CACHE_DIR` points to the folder holding `MegatronTraining_x86_64_202510011427.sif` and load the JUWELS environment exporting `SLURM_ACCOUNT` and `SLURM_PARTITION_DEBUG`.
- Update the `templates/juwels.sbatch` module list if your site configuration differs.
- Dry run with the fake SLURM client before touching the real scheduler:

```bash
oellm-autoexp submit autoexp --config-dir config \
  -o project=juwels_speed_test \
  -o slurm=juwels_singularity \
  -o backend=megatron_speed_test_juwels \
  -o sweep=speed_test_juwels \
  --fake
```

- Drop `--fake` and set the real client once the SLURM integration in `oellm-autoexp` is ready, keeping overrides like `slurm.sbatch.partition=$SLURM_PARTITION_DEBUG` if you need a different queue.

## Migration Guides

### autoexperiment ➜ oellm-autoexp
- Swap `autoexperiment build|run|build_and_run` for `oellm-autoexp plan|submit|monitor`; add `--dry-run` to `submit` when you only want a preview.
- Legacy single-file configs map into explicit `project`, `slurm`, `monitoring`, and `sweep` blocks. The bundled `config/autoexp.yaml` composes the same defaults automatically.
- Job recovery data now lives under `.oellm-autoexp/`; keep sweep-driven names deterministic via `sweep.name_template` instead of hand-crafted `{name}` strings.

| autoexperiment concept | oellm-autoexp equivalent | Notes |
| --- | --- | --- |
| `template` | `slurm.template_path` | SBATCH template rendered per job. |
| `sbatch_script` | `slurm.script_dir` | Output directory for rendered scripts. |
| `cmd` | `slurm.submit_cmd` | Still defaults to `sbatch`; supports arrays and launch wrappers. |
| `output_file` | `monitoring.log_path_template` | Defaults to `"{output_dir}/slurm.out"` and feeds the monitor. |
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
output_file: "{logs}/{name}/slurm.out"
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
  log_path_template: "{output_dir}/slurm.out"
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
- Keep Megatron flags under the `megatron` block; sweeps reference them via `sweep.axes`, letting the orchestrator validate every flag against Megatron-LM's parser.

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
| `launcher.cmd` | `slurm.launcher_cmd` | Prefix command (e.g., container runtime). |
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

## State Persistence

- The orchestrator writes `monitor/state.json` inside `project.state_dir` (defaults to `<base_output>/.oellm-autoexp`).
- Rerunning `oellm-autoexp submit` against the same plan rehydrates in-flight jobs without resubmitting them.
- When all jobs finish, the state file is removed automatically.

## Diagnostics & Testing

- `pytest` runs unit + integration suites (fake SLURM, Megatron parser, sweep expansion, monitoring).
- `python scripts/run_autoexp.py ... --dry-run` previews scripts, sweep metadata, and array scripts without launching jobs.
- `oellm-autoexp monitor ... --dry-run --json-output` renders planned actions for log signals while keeping side effects disabled.

## Containers & Validation

- `container/build_container.sh --help` lists backend-specific definition files (currently `megatron`).
- `scripts/run_megatron_container.py` wraps Apptainer/Singularity execution with dry-run and fake-submit toggles.
- `scripts/run_sweep_entry.py` doubles as the SLURM array entrypoint and a convenient local debugger for sweep entries.

Refer to `SPEC.md` for ongoing work such as real SLURM submission paths, scheduler throttling, and orchestrator persistence.
