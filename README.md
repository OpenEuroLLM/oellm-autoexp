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

Here `slurm.launcher_cmd` is treated as a prefix (e.g. `apptainer exec ...`), while the backend command is rendered separately. Setting `slurm.launcher_env_passthrough: true` forwards `slurm.environment` keys into the launcher via `--env VAR=$VAR`.


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
