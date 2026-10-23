# DataMix chain: training, HF conversion and evaluation of every checkpoint

This complements [lumi_autoexp_installation_guide.md](lumi_autoexp_installation_guide.md): install `oellm-autoexp`
first as described there. The chain runs the DataMix trainings (`scale/<scale>.yaml`), converts every saved checkpoint
to HF while training continues, and evaluates each converted checkpoint with oellm-eval.

## Quick start

One config, `chain-datamix.yaml`, runs everything. You name the experiment and choose the model sizes and the stages
in it:

```yaml
aux:
  name: ???                           # your experiment, e.g. fin-bin00: names run dirs, SLURM jobs, W&B runs
  output_root: ${oc.env:OUTPUT_ROOT,output}   # relative to the repo root
  scales: [0.1B, 0.4B, 1.4B]          # any subset
  stages: [train, convert, eval]      # any subset
  data_path:
    - ${aux.project_dir}/users/${oc.env:USER}/pca_filtering/data/tokenization/fin_Latn/bin_00_text_document
  eval_task_groups: [open-sci-0.01]   # needed for the eval stage
  eval_limit: null                    # samples per task; null = full tasks
```

For each selected scale, it submits:

- `train`: the training of `scale/<scale>.yaml`.
- `convert`: one convert job per checkpoint, which starts as soon as the checkpoint is saved.
- `eval`: one eval job per checkpoint, which starts as soon as its conversion is done.

Before the first run:

1. Set `aux.name`, a short name of your experiment (e.g. of the data mix). Each experiment needs its own: runs with the
   same name share run dirs, so a second one would resume or skip the first one's jobs.
2. Set `aux.data_path` to your training data, see [Training data](#training-data).
3. Set the evaluation task groups `aux.eval_task_groups` (all groups are listed in
   `submodules/oellm_evals/oellm/resources/task-groups.yaml`). They are only needed if `aux.stages` includes `eval`.
4. Check the [prerequisites](#prerequisites) once.

Run it from the repo root in a tmux window and keep it running: it submits the convert and eval jobs only when their
inputs are ready.

```bash
export HF_HOME=/scratch/project_465002530/cache   # tokenizer + eval datasets
export OUTPUT_ROOT=/scratch/project_465002530/users/$USER/autoexp_runs   # optional, default output/ in the repo

uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix
```

Any value can be overridden after the config name instead of editing the file, e.g. only the 0.4B model, or only
two of its checkpoints:

```bash
uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix 'aux.scales=[0.4B]'
uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix 'aux.scales=[0.4B]' 'aux.ckpt_iters=[4000,47684]'
```

To see what would be submitted, add `--dry-run`: it renders the job scripts into the run dirs without submitting.

### Training data

`aux.data_path` is Megatron's `--data-path`, one list item per word. A single dataset is just its prefix (the path
without `.bin`/`.idx`). A mix alternates weight and prefix items, with the weights quoted:

```yaml
  data_path:
    - "0.3"
    - /path/to/bin_00_text_document
    - "0.7"
    - /path/to/bin_01_text_document
```

Megatron normalises the weights, so they don't need to sum to 1. All scales train on the same data.

### Changing training settings

The training settings come from `scale/<scale>.yaml` and `datamix.yaml`. Anything under `backend.megatron` set in
`chain-datamix.yaml` overrides them, for all selected scales:

```yaml
backend:
  megatron:
    min_lr: 1.0e-5
```

For a value that differs per scale, put a per-scale table into `aux` and pick from it with the job's scale:

```yaml
aux:
  lr: {0.1B: 0.003, 0.4B: 0.0015, 1.4B: 0.0008}
backend:
  megatron:
    lr: ${aux.lr[${aux.scale}]}
```

Keep in mind that the datamix comparison assumes the same training setup for every mix.

### Changing evaluation settings

The eval jobs are set through the `aux.eval_*` values, in `chain-datamix.yaml` or on the command line:

```yaml
aux:
  eval_task_groups: [open-sci-0.01]
  eval_limit: 100          # samples per task; null = full tasks
  eval_time: "08:00:00"    # SLURM time limit per eval job, raise it for large task groups
  eval_partition: small-g
```

Other oellm-eval options aren't exposed. Don't set them under `backend` in `chain-datamix.yaml`: that would also change
the training config, and the eval jobs override most `backend` values anyway. Add an `aux.eval_*` value for such an
option in `_chain-datamix.yaml` instead, next to `backend.limit: "\\${aux.eval_limit}"`.

### Selecting stages

Only the jobs of the stages in `aux.stages` are created, so the settings of the other stages aren't needed. E.g. a
training-only run doesn't need `aux.eval_task_groups`:

```yaml
aux:
  name: fin-bin00
  stages: [train]
```

If `aux.stages` includes `eval` but `aux.eval_task_groups` is missing, the config fails to resolve with
`Missing mandatory value: aux.eval_task_groups`.

### Several sessions

The stages can be split over sessions that cooperate through the files in the run dir, e.g. training and conversion
in one tmux window and evaluation in another:

```bash
uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix 'aux.stages=[train,convert]'
uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix 'aux.stages=[eval]'   # 2nd window
```

The eval session waits for `hf/iter_N.done`, which the convert jobs write, so you can start it together with the
training or any time later. Rules:

- Both sessions must use the same `aux.name`, `HF_HOME` and `OUTPUT_ROOT`, because they find each other's outputs through the
  run dir.
- Don't select the same stage of the same scale in two sessions that run at once, or its jobs are submitted twice.
- Start sessions a few seconds apart: the session id is a timestamp in seconds, so two sessions started in the same
  second share their state.

### Test run

To test the whole pipeline on `dev-g`, run `run_test.sh`. Outside tmux it starts itself in the tmux session
`datamix-test` and attaches to it (detach: `Ctrl-b d`, reattach: `tmux attach -t datamix-test`); the shell stays open
after the monitor exits, so its output stays readable. Close it with `exit` before the next test run.

```bash
bash config/experiments/datamix/run_test.sh                       # all scales
bash config/experiments/datamix/run_test.sh 'aux.scales=[0.1B]'   # quicker: one scale
bash config/experiments/datamix/run_test.sh --dry-run             # render the job scripts only
```

It checks the [prerequisites](#prerequisites) and runs `chain-datamix-test.yaml`: every scale on 1 node for 100
iterations, with checkpoints at 50 and 100, and 50 samples per eval task. It writes to `output/test/datamix/<scale>`
(ignores `OUTPUT_ROOT`). dev-g allows only 2 jobs per user and a rejected job is dropped, so every job, trainings
included, is submitted only while you have fewer than 2 jobs there (`aux.submit_guard` in `_chain-test.yaml`). The 15
jobs therefore run two at a time. Don't run other dev-g jobs at the same time. Finished jobs are skipped, so delete
`output/test/datamix` to test again from scratch.

## Outputs

Each scale writes into its own run dir, `<output_root>/<name>/<scale>` (`aux.output_root` defaults to `$OUTPUT_ROOT`,
else `output`, relative to the repo root). The SLURM jobs are named `<name>-<scale>` (+ `_convert_N`, `_eval_N`), and
so is the W&B run.

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

N is zero-padded to 7 digits (`iter_0004000`).

A job whose output already exists is not submitted (`aux.skip_done`):

- A finished training, i.e. its final checkpoint is saved.
- A converted checkpoint (`hf/iter_N.done`).
- An evaluated checkpoint (`eval/iter_N.done`).

So starting the chain again only submits what is missing. A re-run eval writes a new `<timestamp>/` folder, so
earlier results are kept. To redo a step, delete its `.done` file.

The session state of each `run_autoexp.py` is stored in `monitor_state/<session>/` in the repo root.

- **A monitor died.** If its session has no training running or queued, just start it again.
- **A session with a running training.** Don't start it again while its training job is still queued or running,
  because that submits the training a second time. Reattach to the old session instead:
  `uv run python scripts/monitor_autoexp.py --session <session>`. Alternatively, start a new session without the
  training: `'aux.stages=[convert,eval]'`.
- **Resuming a killed training.** Starting the chain again resumes the training from its last checkpoint.

## Configuration

The configs are layered; later files override earlier ones:

| File | Role |
|------|------|
| `datamix.yaml` | Settings shared by all scales |
| `scale/<scale>.yaml` | The training of one model size; the sweep picks one per job. Also runs on its own |
| `_chain-datamix.yaml` | The plumbing: run dirs, the train/convert/eval jobs, skipping of finished jobs |
| `chain-datamix.yaml` | What you run: scales, stages, data, eval tasks |
| `_chain-test.yaml`, `chain-datamix-test.yaml`, `run_test.sh` | The [test run](#test-run) on dev-g |

To check what a config resolves to, render it:
`uv run python scripts/render_config.py --config-name experiments/datamix/chain-datamix`.

### Values you may need to change

| Value | Default | Where | Notes |
|-------|---------|-------|-------|
| `aux.name` | – | `chain-datamix.yaml` | Required; one per experiment |
| `aux.output_root` | `$OUTPUT_ROOT`, else `output` | `chain-datamix.yaml` | Holds the run dirs `<name>/<scale>`; same for all sessions |
| `aux.scales` | `[0.1B, 0.4B, 1.4B]` | `chain-datamix.yaml` | Model sizes to run |
| `aux.stages` | `[train, convert, eval]` | `chain-datamix.yaml` | Stages to run, see [Several sessions](#several-sessions) |
| `aux.data_path` | `<project_dir>/users/$USER/pca_filtering/...` | `chain-datamix.yaml` | Your training data, see [Training data](#training-data) |
| `aux.eval_task_groups` | `[open-sci-0.01]` | `chain-datamix.yaml` | Required for the eval stage |
| `aux.eval_limit` | `null` (full tasks) | `chain-datamix.yaml` | Samples per task, e.g. for a quick test |
| `$HF_HOME` | – | env | Must hold the tokenizer and the eval datasets |
| `slurm.sbatch.account` | `project_465002530` | `config/slurm/lumi.yaml` | Billed project; also defines `aux.project_dir` |
| `aux.project_dir` | `/scratch/<account>` | `_chain-datamix.yaml` | Project scratch; the eval container (and the default `data_path`) live under it |
| `aux.eval_container` | `<project_dir>/containers/laif-rocm-6.4.4-…sif` | `_chain-datamix.yaml` | ROCm eval container (vllm + transformers) |
| `aux.convert_container` | `<repo>/artifacts/MegatronTrainingLumi_bridge_x86_64.sif` | `_chain-datamix.yaml` | Training container + `nvidia-modelopt`, `pulp` |
| `aux.tokenizer_dir` | snapshot of `backend.megatron.tokenizer_model` in `$HF_HOME/hub` | `_chain-datamix.yaml` | Conversion needs a local dir; a local path in `tokenizer_model` is used as is |
| `aux.ckpt_iters` | every `save_interval` + the final iteration | `_chain-datamix.yaml` | Checkpoints to convert/evaluate, e.g. `'[4000,47684]'` |
| `aux.skip_done` | `true` | `_chain-datamix.yaml` | `false` redoes finished trainings, conversions and evals |
| `aux.convert_time`, `aux.convert_partition` | `01:00:00`, `small-g` | `_chain-datamix.yaml` | For `dev-g` (max 2 jobs per user) also set `aux.submit_guard`, as `_chain-test.yaml` does |
| `aux.submit_guard` | `true` | `_chain-datamix.yaml` | Extra shell check every job must pass before it is submitted. A rejected sbatch is not retried; the job is dropped |
| `aux.eval_time`, `aux.eval_partition` | `04:00:00`, `small-g` | `_chain-datamix.yaml` | Raise the time for large task groups |

`aux.ckpt_iters` is computed once, from the 0.1B config. This works because all datamix scales share `train_iters`
(100B tokens, global batch 512, sequence length 4096) and `save_interval`. If a scale ever differs, set
`aux.ckpt_iters` explicitly.

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
