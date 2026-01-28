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


## Supercomputer setup: JUWELS Booster / JUPITER
To be tested.
See also the predecessors https://github.com/SLAMPAI/megatron-autoexp ([Notes](https://iffmd.fz-juelich.de/yAbNVj9eQz647elSwlyHXQ)) and https://github.com/SLAMPAI/autoexperiment for hints.
Testing for JUPITER: see the [JUPITER Notes](https://iffmd.fz-juelich.de/BoygWCOZRciXluqqcDQ1oQ)
Tokenization (tested at JSC, is generic): https://github.com/marianna13/megatron-lm-parallel-data
Containers:
- JUPITER `/p/data1/mmlaion/shared/containers/pytorch_24.09-py3_arm_transformers_latest.sif`
- JUWELS/JURECA `/p/data1/mmlaion/shared/containers/pytorch_24.09-py3.sif ; pytorch_25.03-py3.sif`
- inspect container:
```bash
CONTAINER_IMAGE="image"
apptainer shell ${CONTAINER_IMAGE}
```


## Quick Recipes

### Single job / Sweep debugging
```bash
# Plan + submit + monitor in one go (manifest written to outputs/manifests/)
python scripts/run_autoexp.py --config-name experiments/default

# Prefer explicit plan/submit?
python scripts/render_config.py  --config-name experiments/default
python scripts/submit_autoexp.py  --config-name experiments/default
```

### Monitoring sessions
- Every submission drops `<monitoring_state_dir>/<session_id>/<job_id>.json`. Resuming is symmetric:
```bash
python scripts/monitor_autoexp.py --session <session_id>
python scripts/monitor_autoexp.py --session-dir monitor_state/<session_id>
```
- The session file stores all the config, last SLURM state, per-event history, so you can crash/restart without guessing log names.

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

This creates 9 jobs (3 × 3 grid) with all combinations of learning rates and batch sizes. Within sweep always escape omegaconf interpolations as `\\${...}`, as otherwise the value from outside the sweep will be taken (but this way you can reference those).

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

Exclude specific combinations using omegaconf interpolations:

```yaml
sweep:
  defaults:
    project.name: "experiment_\\${backend.megatron.lr}_\\${backend.megatron.global_batch_size}"
  type: 'product'
  groups:
    - backend.megatron.lr: [1e-4, 5e-4, 1e-3, 2e-3]
    - backend.megatron.global_batch_size: [64, 128, 256, 512, 1024]
  # Exclude configurations where large LR + large batch size
  filter: "\\${oc.eval:'not (\\${backend.megatron.lr} > 1e-3 and \\${backend.megatron.global_batch_size} > 256)'}"
```

**Result:** Filters out jobs where `lr=2e-3` and `batch_size ∈ {512, 1024}`, keeping only valid combinations.

This is applied after the first sweep expansion but before job creation, so siblings could be referenced as well.

#### Nested Sweeps

Apply filters at any level to exclude unstable or redundant combinations. Building on the composable example above:

```yaml
sweep:
  defaults:
    project.name: "tuning_\\${backend.megatron.lr}_\\${backend.megatron.global_batch_size}_\\${stage}"
  type: list
  groups:
    # Strategy 1: Small batch exploration
    - type: product
      groups:
        - params:
            backend.megatron.lr: [1e-4, 5e-4, 1e-3, 2e-3]  # Added 2e-3
        - params:
            backend.megatron.global_batch_size: [64, 128]
      defaults:
        stage: small_batch
      # Filter out aggressive LR (2e-3) to avoid instability
      filter: "\\${oc.eval:'\\${backend.megatron.lr} <= 1e-3'}"
      # Result: 3 LRs × 2 batch sizes = 6 jobs (2e-3 excluded)

    # Strategy 2: Large batch exploration
    - type: product
      groups:
        - params:
            backend.megatron.lr: [5e-4, 1e-3, 2e-3, 5e-3]  # Wider range
        - params:
            backend.megatron.global_batch_size: [256, 512, 1024]  # Added 1024
      defaults:
        stage: large_batch

    # Strategy 3: Production (no filter needed)
    - configs:
        - stage: production
          backend.megatron.lr: 5e-4
          backend.megatron.global_batch_size: 256
```


## Start conditions and monitoring actions (log/state events)
Monitoring behavior lives entirely in YAML. Keep it small, keep it explicit.
`start_condition` describes a condition (or combination of conds.) that needs to be fulfilled before actual job submission
`cancel_condition` describes a condition that causes a cancellation of the job (even before submission)
`log_events` describe detectors (regex/substring/inactivity). `state_events` wire SLURM transitions (`pending`, `success`, etc.) to actions.


### Updating the Megatron-LM backend version
For an update of the megatron backend, first check out the new submodule version. Then, create a new container. Within that container,
run the script generation in `scripts/generate_megatron_config.py` and `scripts/generate_megatron_dataclass.py`. You might have to adapt the `transformer_engine` mocks in those scripts.
Also, apparently some containers don't use the correct `C++` path, you might have to `export CXX=$(which clang++)`, for example on LUMI.
