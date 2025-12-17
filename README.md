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
