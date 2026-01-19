# OELLM Auto Experimentation Tool

Single CLI surface for planning sweeps, launching jobs (directly or by way of containers), and monitoring SLURM runs by way of declarative configs. See `SPEC.md` for platform-wide goals; this README focuses on the workflows you touch every day.

## Your own experiments

For you own experiments, first create your own branch `exp_YOURNAME`. Add a folder `config/experiments/YOURNAME`. Then, within that folder you can add your own experiment composition files (see the existing ones), with `# @package _global_` as header to ensure it's located at the top-level of the config. You can then run your experiment with:
```bash
PYTHONPATH=. python scripts/run_autoexp_container.py --config-ref experiments/YOURNAME/myexperiment
```


## Using this repository
First, clone this repository including its submodules:
```bash
git clone https://github.com/OpenEuroLLM/oellm-autoexp.git --recurse-submodules
```
Then, install it to have the basic requirements installed:
```bash
pip install -e .
```
Whenever you have `numpy>=2.0` in your system, apply the annoying `numpy.product` error patch:
```bash
bash ./apply_megatron_numpy_product_patch.sh
```

### Environment variables
The environment variables used in the `config/` here are:
- $OUTPUT_DIR           : general output directory
- $DATA_DIR             : general data directory
- $PROJECT_DIR          : project directory (optional, . is an alternative usually)
- $HOME                 : home sweet home
- $SLURM_QOS            : default SLURM qos (or null)
- $HF_HOME              : huggingface home
- $SLURM_ACCOUNT        : default slurm account
- $SLURM_PARTITION      : default slurm partition ($SLURM_PARITION_DEBUG can point to a debug partition)
- $CONTAINER_CACHE_DIR  : directory containing container images


## Cluster setup: LUMI notes
- Install prerequisites outside the container (rccl-tuner, cray-python, etc.) following the LUMI docs. (SEE: https://github.com/sfantao/rccl-tuner.git)
- Build the Megatron container from the provided defs (see `container/megatron/MegatronTrainingLumi.def.in`) so the correct ROCm + network tuning ends up inside the image.
- Export the usual SLURM/paths (at a minimum `SLURM_ACCOUNT`, `SLURM_PARTITION[_DEBUG]`, `CONTAINER_CACHE_DIR`, `OUTPUT_DIR`) in your profile—scripts read them automatically.
- Apply the numpy patch, as the container numpy version is too new for `np.product` to be supported still

## Cluster setup: MARENOSTRUM notes
You need to install oellm-autoexp or its requirements in a conda environment to run it on MARENOSTRUM. To do this:
- Install conda-pack with: `conda install conda-pack` on your local machine
- Create new conda environment with `conda env create -f base_environment.yaml -n oellm_autoexp`
- Pack/Export the conda env: `conda-pack --name oellm_autoexp --output oellm_autoexp.tar.gz`
- Send this to `marenostrum`: `rsync oellm_autoexp.tar.gz marenostrum:~/`
- Unpack it there: `mkdir oellm_autoexp_env ; cd oellm_autoexp_env ; tar -xzvf ../oellm_autoexp.tar.gz`
- Add the environment to your bashrc: `echo "source ~/oellm_autoexp_env/bin/activate" >> ~/.bashrc` or load it when you need it
- Use a container built on LEONARDO or JUWELS (MARENOSTRUM has no internet access to you can't build anything there)
- Copy datasets and tokenizer etc. manually (no web connection on compute and login nodes)

## Cluster setup: LEONARDO notes
For LEONARDO, all should work with a pre-built container image. To build a container image on LEONARDO, please run these commands:
- Download the pytorch base image from nvcr.io: `singularity build --sandbox --fix-perms --force $CONTAINER_CACHE_DIR/pytorch__25.08-py3_sandbox docker://nvcr.io/nvidia/pytorch:25.08-py3`
- Build the user-base container (in `container`), but from a compute node: `python build_container_user.py --backend megatron --definition MegatronTrainingNoRoot --append-date --container-cmd singularity --base-image $CONTAINER_CACHE_DIR/pytorch__25.08-py3_sandbox`
Otherwise, on the login node you run out of resources and get killed.
Make sure also to have datasets and tokenizers downloaded before starting a job, as there is no web connection on the compute nodes.

## Quick Recipes

### Single job / Sweep debugging
```bash
# Plan + submit + monitor in one go (manifest written to outputs/manifests/)
python scripts/run_autoexp.py project=default

# Prefer explicit plan/submit?
python scripts/plan_autoexp.py --manifest outputs/manifests/demo.json project=default
python scripts/submit_autoexp.py --manifest outputs/manifests/demo.json

# Probe a specific sweep entry
python scripts/run_sweep_entry.py --sweep outputs/sweep.json --index 0 --dry-run
```

### Monitoring sessions
- Every submission drops `<monitoring_state_dir>/<plan_id>.json`. Resuming is symmetric:
```bash
python scripts/monitor_autoexp.py --session <plan_id>
# legacy path works too
python scripts/monitor_autoexp.py --manifest outputs/manifests/<plan>.json
```
- The session file stores `resolved_log_path`, last SLURM state, per-event history, and the manifest path so you can crash/restart without guessing log names. `scripts/tests/test_monitor_resume.py` is the on-cluster smoke test that exercises plan → submit → interrupt → resume.
- The action queue lives next to the session file (`<plan_id>.actions/`). Use the CLI to keep it short and readable:
```bash
# Summaries (optionally filter by event)
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue --queue-event checkpoint_saved

# Inspect or retry a specific entry
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue \
    --queue-id <queue_uuid> --queue-show
python scripts/monitor_autoexp.py --session <plan_id> --cmd queue \
    --queue-id <queue_uuid> --queue-retry

# Run the worker that executes queued actions (for example, RunAutoexpAction)
python scripts/monitor_autoexp.py --session <plan_id> --cmd actions
```
Each entry is a single JSON file (`<event_id>/<queue_id>.json`) so manual inspection is always possible, but the CLI avoids editing files directly.

### Container workflow (recommended for Megatron)
```bash
# Build once (see container/MegatronTraining*.def for examples)
./container/build_container.sh --backend megatron --definition MegatronTraining --output ./artifacts
# On LUMI or other clusters, where --fake-root for container builds is not available
python ./container/build_container_user.py --backend megatron --definition MegatronTrainingLumi \
    --requirements container/megatron/requirements_latest.txt \
    --output ./artifacts

# Plan inside the container, submit/monitor on host
python scripts/run_autoexp_container.py \
  --config-ref experiments/megatron_jupiter_speed_test \
  container=juwels

# Inspect or reuse the manifest later
python scripts/monitor_autoexp.py --manifest outputs/manifests/<plan>.json
```
`run_autoexp_container.py` accepts `--no-submit`, `--no-monitor`, and `--monitor-only` so the same manifest feeds both automation and manual debugging.

## Hyperparameter Sweeps

OELLM Auto-Exp supports powerful sweeping capabilities for hyperparameter exploration, including multi-stage experiments with automatic dependency resolution.

### Basic Sweeping (Grid Format)

Define parameter grids in your config:

```yaml
sweep:
  base_values:
    backend.megatron.num_layers: 20
    backend.megatron.hidden_size: 896
    project.name: "experiment_\\${backend.megatron.lr}_\\${backend.megatron.global_batch_size}"
  grids:
    - backend.megatron.lr: [1e-4, 5e-4, 1e-3]
      backend.megatron.global_batch_size: [64, 128, 256]
```

This creates 9 jobs (3 × 3 grid) with all combinations of learning rates and batch sizes.

### Composable Sweeps (Groups Format)

For complex experiments, use the composable groups format to combine different sweep strategies. Groups can use `type: product` (Cartesian product) or `type: list` (sequential concatenation):

```yaml
sweep:
  type: list  # Top-level: concatenate independent exploration strategies
  defaults:
    project.name: "tuning_\\${backend.megatron.lr}_\\${backend.megatron.global_batch_size}_\\${stage}"
  groups:
    # Strategy 1: Small batch exploration (product of LR × batch sizes)
    - type: product
      groups:
        - params:
            backend.megatron.lr: [1e-4, 5e-4, 1e-3]
        - params:
            backend.megatron.global_batch_size: [64, 128]
      defaults:
        stage: small_batch
        backend.megatron.train_iters: 5000
      # Result: 3 LRs × 2 batch sizes = 6 jobs

    # Strategy 2: Large batch exploration (product of LR × batch sizes)
    - type: product
      groups:
        - params:
            backend.megatron.lr: [5e-4, 1e-3, 2e-3]
        - params:
            backend.megatron.global_batch_size: [256, 512]
      defaults:
        stage: large_batch
        backend.megatron.train_iters: 5000
      # Result: 3 LRs × 2 batch sizes = 6 jobs

    # Strategy 3: Best configurations for production
    - configs:
        - stage: production
          backend.megatron.lr: 5e-4
          backend.megatron.global_batch_size: 256
          backend.megatron.train_iters: 100000
        - stage: production
          backend.megatron.lr: 1e-3
          backend.megatron.global_batch_size: 128
          backend.megatron.train_iters: 100000
      # Result: 2 jobs (hand-picked best configs)
```

**Total result:** 6 + 6 + 2 = **14 jobs** (independent strategies concatenated)

**Composition modes:**
- `type: product` → Creates Cartesian product **across** groups (multiply)
  - Example: 3 LRs × 2 batch sizes = 6 jobs
- `type: list` → Concatenates groups **sequentially** (add)
  - Example: 6 + 6 + 2 = 14 jobs
- `params:` → Creates grid sweeps **within** a group
- `configs:` → Lists individual configurations
- Groups can be **arbitrarily nested** with their own `type`

### Multi-Stage Experiments with Sibling References

For experiments that build on previous stages (for example, different training phases):

```yaml
sweep:
  type: list
  groups:
    - params:
        backend.megatron.lr: [2.5e-4, 5e-4, 1e-3]
        backend.megatron.global_batch_size: [64, 128, 256]
      defaults:
        stage: stable
        backend.megatron.train_iters: 18000

    - params:
        backend.megatron.lr: [2.5e-4, 5e-4, 1e-3]
        backend.megatron.global_batch_size: [64, 128, 256]
      defaults:
        stage: decay6B
        backend.megatron.train_iters: 36000
        # Reference the stable stage sibling
        backend.megatron.load: "\\${sibling.stable.output_dir}/checkpoints"

      # Start conditions - only start when stable stage completes
      job.start_conditions:
        - class_name: FileExistsCondition
          path: "\\${sibling.stable.output_dir}/checkpoints/done.txt"

      # Cancel if stable stage fails
      job.cancel_conditions:
        - class_name: SlurmStateCondition
          job_name: "\\${sibling.stable.name}"
          state: FAILED
```

**Key points:**
- Use `\\${sibling.STAGE.FIELD}` to reference sibling jobs (double-escaped in YAML)
- Available fields: `name`, `output_dir`, `log_path`, `log_path_current`
- Dependencies are automatically resolved by way of the DAG-based resolver
- Jobs start only when their dependencies complete

For SLURM arrays, `project.log_path_current` is resolved per index (default `current_${index}.log`); after submission a symlink is created once the SLURM ID is known, so `log_path_current` is stable for monitoring and tooling.

### Job controls (declarative)
Job gating lives in the `job` section and is fully parsed into the dataclasses (no extra overrides):

```yaml
job:
  start_conditions:
    - class_name: FileExistsCondition
      path: "\\${sibling.stable.output_dir}/checkpoints/done.txt"
  cancel_conditions:
    - class_name: SlurmStateCondition
      job_name: "\\${sibling.stable.name}"
      state: FAILED
  inactivity_threshold_seconds: 1800
```

In sweeps you can also use dotted keys (for example, `job.start_conditions`) to target the same fields.

### Visualizing Your Sweep

Before running, visualize the execution plan:

```bash
# Visualize the multi-stage DAG structure
python scripts/visualize_plan.py --config-ref experiments/my_experiment

# Limit jobs shown per stage
python scripts/visualize_plan.py --config-ref experiments/my_experiment \
    --max-jobs-per-stage 5

# With Hydra overrides
python scripts/visualize_plan.py --config-ref experiments/my_experiment \
    backend.megatron.lr=1e-4
```

**Example output:**
```
======================================================================
 Multi-Stage Experiment Plan: dense_300M_sweep
======================================================================
Total: 75 jobs across 5 stage(s)

┌────────────────────────────────────────────────────────────────────┐
│ Hyperparameter Sweep                                               │
│ • lr: [2.5e-4, 5e-4, 1e-3, 2e-3]                                   │
│ • global_batch_size: [64, 128, 256, 512, 1024]                     │
│ Total combinations: 15                                             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Stage: stable (15 jobs)                                            │
├────────────────────────────────────────────────────────────────────┤
│ Start: Immediate                                                   │
└────────────────────────────────────────────────────────────────────┘

              │             │             │             │

┌────────────────────────────────────────────────────────────────────┐
│ Stage: decay6B (15 jobs)                                           │
├────────────────────────────────────────────────────────────────────┤
│ Start Conditions:                                                  │
│   • FileExists: .../checkpoints/iter_18000/done.txt                │
│ Cancel Conditions:                                                 │
│   • SlurmState: stable_job = FAILED                                │
└────────────────────────────────────────────────────────────────────┘
```

### Logging Control

Control verbosity with the `OELLM_LOG_LEVEL` environment variable or command-line flags:

```bash
# Minimal output (warnings/errors only)
python scripts/visualize_plan.py --config-ref my_experiment

# INFO level logging (shows progress)
OELLM_LOG_LEVEL=INFO python scripts/visualize_plan.py --config-ref my_experiment

# Debug logging (detailed internal operations)
OELLM_LOG_LEVEL=DEBUG python scripts/run_autoexp.py --config-ref my_experiment

# Command-line flags override environment variable
python scripts/visualize_plan.py --debug --config-ref my_experiment
```

**Log levels:** `DEBUG`, `INFO`, `WARNING` (default), `ERROR`, `CRITICAL`

**Priority (highest to lowest):**
1. Command-line flags (`--debug`, `--verbose`)
2. `OELLM_LOG_LEVEL` environment variable
3. Default (WARNING)

### Advanced Sweep Features

#### Filters

Exclude specific combinations using Python expressions:

```yaml
sweep:
  defaults:
    project.name: "experiment_\\${backend.megatron.lr}_\\${backend.megatron.global_batch_size}"
  grids:
    - backend.megatron.lr: [1e-4, 5e-4, 1e-3, 2e-3]
      backend.megatron.global_batch_size: [64, 128, 256, 512, 1024]
  # Exclude configurations where large LR + large batch size
  filter: "${oc.eval:'not (\\${backend.megatron.lr} > 1e-3 and \\${backend.megatron.global_batch_size} > 256)'}"
```

**Result:** Filters out jobs where `lr=2e-3` and `batch_size ∈ {512, 1024}`, keeping only valid combinations.

**Filter capabilities:**
- Access flattened parameters by dotted path (for example, `backend.megatron.lr`)
- Use Python operators: `>`, `<`, `==`, `!=`, `and`, `or`, `not`
- Combine multiple conditions
- Applied after sweep expansion but before job creation

#### Adding Filters to Nested Groups

Apply filters at any level to exclude unstable or redundant combinations. Building on the composable example above:

```yaml
sweep:
  defaults:
    project.name: "tuning_\\${backend.megatron.lr}_\\${backend.megatron.global_batch_size}_\\${stage}"
  type: list
  filter: "\\${oc.eval:'not (\\${backend.megatron.lr} > 2e-3 and \\${backend.megatron.global_batch_size} > 512)'}"
  groups:
    # Strategy 1: Small batch exploration
    - type: product
      groups:
        - params:
            backend.megatron.lr: [1e-4, 5e-4, 1e-3]
        - params:
            backend.megatron.global_batch_size: [64, 128]
      defaults:
        stage: small_batch

    # Strategy 2: Large batch exploration
    - type: product
      groups:
        - params:
            backend.megatron.lr: [5e-4, 1e-3, 2e-3, 5e-3]  # Wider range
        - params:
            backend.megatron.global_batch_size: [256, 512, 1024]  # Added 1024
      defaults:
        stage: large_batch

    # Strategy 3: Production
    - configs:
        - stage: production
          backend.megatron.lr: 5e-4
          backend.megatron.global_batch_size: 256
```

**Result:** 6 + 8 + 1 = **15 jobs** (filters prevent unstable configurations)

#### Other Features

- **OmegaConf interpolations**: Use `${oc.eval:...}` for computed values
  ```yaml
  backend.megatron.train_iters: "${oc.eval:int(50e9 / ${backend.megatron.global_batch_size})}"
  ```
- **Group defaults**: Share common values within a group using `defaults:` block
- **Validation**: The planner validates all sibling references and dependencies before submission (DAG-based resolution).

For complete examples, see:
- `config/experiments/korbi/dense_300M_50BT_pull.yaml` - Multi-stage sweep with 5 training phases
- `config/experiments/korbi/repro_sweep_niccolo_small.yaml` - Large hyperparameter grid
- `docs/multi_stage_design.md` - Design documentation
- `docs/sweep_resolution_ordering.md` - Technical details on DAG resolution

## Monitoring configs (log/state events)
Monitoring behavior lives entirely in YAML. Keep it small, keep it explicit:

```yaml
# config/monitoring/megatron_checkpoint_eval.yaml
log_events:
  - name: checkpoint_saved
    pattern: "saved checkpoint (?P<path>\\S+)"
    pattern_type: regex
    metadata:
      kind: checkpoint
    extract_groups:
      checkpoint_path: path
    actions:
      - class_name: EventAction
        mode: queue
        conditions:
          - class_name: FileExistsCondition
            path: "{checkpoint_path}"
            blocking: true
        action:
          class_name: RunCommandAction
          # does not work yet
          command: "python convert_checkpoint.py {checkpoint_path} {checkpoint_path}/../converted"

state_events:
  - name: stall
    state:
      class_name: StalledState
    actions:
      - class_name: EventAction
        action:
          class_name: RestartAction
          reason: "retry stall"
  - name: success
    state:
      class_name: SuccessState
    actions:
      - class_name: EventAction
        action:
          class_name: LogAction
          message: "run_finished"
```

`log_events` describe detectors (regex/substring/inactivity). `state_events` wire SLURM transitions (`stall`, `timeout`, `success`, etc.) to actions. Because nothing is hidden in code, configs like `config/monitoring/megatron_basic.yaml` act as the canonical reference: fork it, add or remove events, and the controller will follow exactly what the YAML states.

### Tips
- Use `pattern_type: inactivity` to emit events when the log or additional output paths stop changing for `inactivity_threshold_seconds`.
- Queue actions (`mode: queue`) for sidecar jobs or evaluations; inline actions are for immediate restarts/logging.
- Everything templated by way of `str.format` gets the merged metadata (`job_id`, `checkpoint_path`, etc.) so downstream automation stays simple.

For more context (provenance capture, fake SLURM client, restart CLI), refer to `docs/` and the unit/integration tests—they mirror real-world usage.


### Updating the Megatron-LM backend version
For an update of the megatron backend, first check out the new submodule version. Then, create a new container. Within that container,
run the script generation in `scripts/generate_megatron_config.py` and `scripts/generate_megatron_dataclass.py`. You might have to adapt the `transformer_engine` mocks in those scripts.
Also, apparently some containers don't use the correct `C++` path, you might have to `export CXX=$(which clang++)`, for example on LUMI.
