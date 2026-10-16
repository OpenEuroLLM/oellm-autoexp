# DataMix chains: training, HF conversion and evaluation of every checkpoint

This complements [lumi_autoexp_installation_guide.md](lumi_autoexp_installation_guide.md): install `oellm-autoexp`
first as described there. The chains run the `datamix-*-ne.yaml` trainings unchanged, convert every saved checkpoint
to HF while training continues, and evaluate each converted checkpoint with oellm-eval.

## Quick start

There are two chains per model size (`0.1B`, `0.4B`, `1.4B`), each run as its own `run_autoexp.py` session:

- `chain-train-and-convert-datamix-<size>-ne.yaml`: the training, plus one convert job per checkpoint, which starts as
  soon as the checkpoint is saved.
- `chain-eval-datamix-<size>-ne.yaml`: one eval job per checkpoint, which starts as soon as its conversion is done.
  It submits no training, so you can start it together with the training or any time later.

Before the first run:

1. Set `backend.megatron.data_path` in `chain-train-and-convert-datamix-<size>-ne.yaml` to your data
   (`weight, prefix, weight, prefix, ...`, prefix = path without `.bin`/`.idx`).
2. Set the evaluation task groups `aux.eval_task_groups` in `chain-eval-datamix-<size>-ne.yaml` (default
   `[open-sci-0.01]`; all groups are listed in `submodules/oellm_evals/oellm/resources/task-groups.yaml`).
3. Check the [prerequisites](#prerequisites) once.

Run both sessions from the repo root, each in its own tmux window, and keep them running: they submit the convert and
eval jobs only when their checkpoints are ready.

```bash
export HF_HOME=/scratch/project_465002530/cache   # tokenizer + eval datasets
export OUTPUT_DIR=/scratch/project_465002530/users/$USER/autoexp_runs/train-0.1B-ne   # optional, see below

uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-train-and-convert-datamix-0.1B-ne
uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-eval-datamix-0.1B-ne   # 2nd tmux window
```

Both sessions must see the same `HF_HOME` and `OUTPUT_DIR`, because the eval session finds the checkpoints through
the training run dir. Start them a few seconds apart: the session id is a timestamp in seconds, so two sessions
started in the same second share their state and both submit the training. Any config value can be overridden after
the config name, e.g. evaluate only two checkpoints:

```bash
uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-eval-datamix-0.1B-ne 'aux.ckpt_iters=[4000,47684]'
```

### Toy test

To test the whole pipeline quickly, run the toy chains the same way:
`chain-train-and-convert-datamix-0.1B-ne-toy` and then `chain-eval-datamix-0.1B-ne-toy`. The toy run has 1 node and
100 iterations, with checkpoints at 50 and 100 and 50 samples per eval task, all on `dev-g`. It writes to
`output/train-0.1B-ne-toy` (ignores `OUTPUT_DIR`). dev-g allows only 2 jobs per user, so a convert or eval job is
submitted only while you have fewer than 2 jobs there (`aux.submit_guard` in `_chain-toy.yaml`), and they run one
after another. Don't run other dev-g jobs at the same time. Delete `output/train-0.1B-ne-toy` before running it again.

## Outputs

Everything is written into the training run dir: `$OUTPUT_DIR` if set, else `output/train-<size>-ne` relative to the
repo root (the datamix default). `OUTPUT_DIR` is the run dir itself, so use a different one per model size.

```
<run dir>/
├── checkpoints/iter_N/                    Megatron checkpoints (every save_interval = 4000 iterations + the final one)
├── checkpoints/latest_checkpointed_iteration.txt   a convert job starts once this holds >= N
├── current.yaml, current.log, slurm-<jobid>.log    training config and logs
├── tensorboard/, wandb/                   training metrics (W&B runs offline)
├── convert/iter_N/                        convert job: sbatch, logs
├── hf/iter_N/  hf/iter_N.done             HF model of checkpoint N; .done = export finished
├── eval/iter_N/                           eval job: sbatch, logs
│   └── <timestamp>/results/*.json         lm-eval results
└── eval/iter_N.done                       eval of checkpoint N finished
```

N is zero-padded to 7 digits (`iter_0004000`). The `.done` markers make re-runs cheap: starting a chain again skips
the checkpoints that are already converted (`hf/iter_N.done`) or evaluated (`eval/iter_N.done`). A re-run eval writes a
new `<timestamp>/` folder, so earlier results are kept. To redo a step, delete its `.done` file.

The session state of each `run_autoexp.py` is stored in `monitor_state/<session>/` in the repo root.

- **A monitor died.** For the eval session, just start it again.
- **The training session.** Don't restart it while its training job is still queued or running, because that
  submits the training a second time. Reattach to the old session instead:
  `uv run python scripts/monitor_autoexp.py --session <session>`.
- **Resuming a finished or killed training.** Starting the train-and-convert chain again resumes from the last
  checkpoint.

## Configuration

The configs are layered; later files override earlier ones:

| File | Role |
|------|------|
| `datamix-<size>-ne.yaml` (+ `datamix.yaml`) | The original training config, unchanged |
| `_chain-ckpt-convert.yaml` | Adds one convert job per checkpoint (Megatron-Bridge, Megatron → HF) |
| `_chain-datamix.yaml` | Adapts the datamix config to the chain: project dir, tokenizer dir, convert container |
| `_chain-ckpt-eval.yaml` | Replaces the sweep with one eval job per checkpoint |
| `chain-train-and-convert-datamix-<size>-ne.yaml` | `datamix-<size>-ne` + convert + adapter; sets `data_path` |
| `chain-eval-datamix-<size>-ne.yaml` | `chain-train-and-convert-datamix-<size>-ne` + eval; sets the eval tasks |
| `_chain-toy.yaml`, `chain-*-datamix-0.1B-ne-toy.yaml` | The [toy test](#toy-test) |

Don't change `job.name` or `job.base_output_dir` in the eval configs: they locate the training run dir.
To check what a config resolves to, render it:
`uv run python scripts/render_config.py --config-name experiments/datamix/chain-eval-datamix-0.1B-ne`.

### Values you may need to change

| Value | Default | Where | Notes |
|-------|---------|-------|-------|
| `backend.megatron.data_path` | `<project_dir>/users/$USER/pca_filtering/...` | `chain-train-and-convert-*` | Your training data |
| `aux.eval_task_groups` | `[open-sci-0.01]` | `chain-eval-*` | Required |
| `aux.eval_limit` | `null` (full tasks) | `chain-eval-*` | Samples per task, e.g. for a quick test |
| `slurm.sbatch.account` | `project_465002530` | `config/slurm/lumi.yaml` | Billed project; also defines `aux.project_dir` |
| `aux.project_dir` | `/scratch/<account>` | `_chain-datamix.yaml` | Project scratch; the eval container (and the default `data_path`) live under it |
| `aux.eval_container` | `<project_dir>/containers/laif-rocm-6.4.4-…sif` | `_chain-ckpt-eval.yaml` | ROCm eval container (vllm + transformers) |
| `aux.convert_container` | `<repo>/artifacts/MegatronTrainingLumi_bridge_x86_64.sif` | `_chain-datamix.yaml` | Training container + `nvidia-modelopt`, `pulp` |
| `aux.tokenizer_dir` | snapshot of `backend.megatron.tokenizer_model` in `$HF_HOME/hub` | `_chain-datamix.yaml` | Conversion needs a local dir; a local path in `tokenizer_model` is used as is |
| `aux.ckpt_iters` | every `save_interval` + the final iteration | `_chain-ckpt-convert.yaml` | Checkpoints to convert/evaluate, e.g. `'[4000,47684]'` |
| `aux.ckpt_skip_done` | `true` | `_chain-ckpt-convert.yaml` | `false` redoes already converted/evaluated checkpoints |
| `aux.convert_time`, `aux.convert_partition` | `01:00:00`, `small-g` | `_chain-ckpt-convert.yaml` | For `dev-g` (max 2 jobs per user) also set `aux.submit_guard`, as `_chain-toy.yaml` does |
| `aux.submit_guard` | `true` | `_chain-ckpt-convert.yaml` | Extra shell check a convert/eval job must pass before it is submitted. A rejected sbatch is not retried; the job is dropped |
| `aux.eval_time`, `aux.eval_partition` | `04:00:00`, `small-g` | `_chain-ckpt-eval.yaml` | Raise the time for large task groups |
| `$OUTPUT_DIR` | `output/train-<size>-ne` | env | Training run dir; same for both sessions |
| `$HF_HOME` | – | env | Must hold the tokenizer and the eval datasets |

### Prerequisites

Check these once per user/project:

- **Tokenizer.** `$HF_HOME/hub/models--openeurollm--tokenizer-256k/` must exist; training loads it offline. The
  config fails to resolve if `refs/main` is missing.
- **Eval environment and datasets.** oellm-eval and lm-eval go into `~/eval_local` (linked to scratch because of the
  home quota), and the eval datasets go into `$HF_HOME`:

  ```bash
  EVAL_LOCAL_DIR=/scratch/project_465002530/users/$USER/eval_local \
  OELLM_EVAL_SIF=/scratch/project_465002530/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif \
      bash scripts/install_eval_env.sh --cluster lumi --prefetch
  ```

- **Convert container.** `artifacts/MegatronTrainingLumi_bridge_x86_64.sif` must exist in the repo. It is the
  training container plus `nvidia-modelopt` and `pulp`, which Megatron-Bridge imports during conversion. Copy it from
  a colleague, or build it from `MegatronTrainingLumi_x86_64.sif`:

  ```bash
  SB=/tmp/$USER-sandbox   # local disk: the unpacked image has too many files for the scratch quota
  singularity build --sandbox --fix-perms $SB artifacts/MegatronTrainingLumi_x86_64.sif
  umask 022; singularity exec --writable --no-home --no-umask $SB \
      python3 -m pip install --no-cache-dir --no-deps nvidia-modelopt pulp
  chmod -R a+rX $SB/opt/venv/lib/python3.12/site-packages/{modelopt,pulp,nvidia_modelopt-*,pulp-*}
  singularity build artifacts/MegatronTrainingLumi_bridge_x86_64.sif $SB && rm -rf $SB
  ```
