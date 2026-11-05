# DataMix on LUMI: quick start

From nothing to a finished test run of the DataMix chain (training → HF conversion →
evaluation), then your own run. For the chain itself (stages, outputs, re-runs, all settings),
see [chain_guide.md](chain_guide.md).

## Quick start

On a LUMI login node:

```bash
cd /scratch/<your project>/users/$USER
git clone -b datamix --recurse-submodules https://github.com/OpenEuroLLM/oellm-autoexp.git
cd oellm-autoexp
bash config/experiments/datamix/setup_lumi.sh
```

- **Branch:** the DataMix configs exist only on the `datamix` branch.
- **No SSH key for GitHub on LUMI?** Some submodules use SSH URLs. If the clone fails on them, run
  `git config --global url."https://github.com/".insteadOf git@github.com:` first, which makes git use
  HTTPS for them. Undo it later with `git config --global --unset url."https://github.com/".insteadOf`.

`setup_lumi.sh` takes you through the setup in 8 steps.
- **Check first:** each step checks whether its part is already done, and skips it if so.
- **Ask:** otherwise it explains what it will do and asks, with a default you can accept with Enter or replace.
- **Show:** it shows every command before running it.

It deletes or overwrites nothing without asking. You can run it again at any time, for example after fixing a failed
step. At the end it starts a test run, if you give it a training dataset.

After the setup, start every new shell and every tmux window with:

```bash
source ~/datamix_env.sh
```

## What the setup does

| Step | Checks / does | Notes |
|------|---------------|-------|
| 1 Checks | Login node, `git`, `singularity`, `tmux`, slurm commands, branch, submodules | Offers to check out missing submodules |
| 2 Project | Asks which of your LUMI projects to bill (default `project_465002530` if you are a member) | All jobs run under it |
| 3 Paths | Asks for your scratch dir, the HF cache and the containers dir, and writes `~/datamix_env.sh` | See [The env file](#the-env-file) |
| 4 uv | Installs `uv` 0.4.15 into `$WORK/.local/bin` if missing, then `uv sync` | autoexp runs on the login node in this env |
| 5 Containers | Checks the train, convert and eval containers; offers to build the convert container | See [Containers](#containers) |
| 6 Eval env | Installs oellm-eval + lm-eval into `$WORK/eval_local` (`~/eval_local` links there), ~10 min | See [Eval environment](#eval-environment) |
| 7 Tokenizer | Checks `openeurollm/tokenizer-256k` in the HF cache, offers to download it | Training loads it offline |
| 8 Test run | Asks for training data and starts `run_test.sh` in tmux | See [Test run](#test-run) |

Options: `--no-test` stops before the test. `--data PATH` and `--scales LIST` answer the test's questions in advance.
`--yes` accepts all defaults. `--help` lists them all.

### The env file

`~/datamix_env.sh` holds everything the later steps and your runs need:

| Variable | Default | Used for |
|----------|---------|----------|
| `WORK` | `/scratch/<project>/users/$USER` | `uv`, its caches, the eval env (`$HOME` has only 20 GB / 100k files) |
| `HF_HOME` | `/scratch/<project>/cache` (shared) | Tokenizer and eval datasets; the jobs read them offline |
| `DATAMIX_PROJECT` | `project_465002530` | Billed project of all jobs (`aux.account` in `datamix.yaml`) |
| `DATAMIX_CONTAINERS` | `/scratch/project_465002530/containers` | Where the three containers are (`aux.containers_dir`) |

`HF_HOME` must be set before anything is run: the tokenizer is looked up there as soon as a config
is loaded. Without it, you get `No such file or directory: '/hub/models--openeurollm--tokenizer-256k/refs/main'`.

### Containers

| Stage | Container | Origin |
|-------|-----------|--------|
| train | `AutoExp_2026-07-02.sif` | LUMI training container, built by Ville Komulainen from `container/megatron/MegatronTrainingLumi.def.in` with `container/build_container_user.py` (see `container/README.md`) |
| convert | `MegatronTrainingLumi_bridge_x86_64.sif` | The training container + `nvidia-modelopt` and `pulp`, which Megatron-Bridge needs for the conversion |
| eval | `laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif` | Prebuilt LUMI AI Factory image (vllm, transformers) |

In `project_465002530` all three are in the shared dir `/scratch/project_465002530/containers/`.

The convert container is built by `container/megatron/build_lumi_bridge_container.sh`, which step 5 offers to run:

```bash
bash container/megatron/build_lumi_bridge_container.sh OUTPUT.sif [BASE.sif]   # BASE default: the shared training container
```

- **Why a sandbox:** LUMI allows neither `singularity build --fakeroot` nor `proot`, so a `.def` recipe can't be built.
  Instead the script unpacks the training container into a sandbox on the login node's `/tmp`, installs the two
  packages and packs it again. That takes ~15 min and needs no root.
- **Pinned:** the two packages are pinned and installed with `--no-deps`, so the training stack is exactly the
  training container's.
- **Overwriting:** the script refuses to overwrite an existing image.

You don't need to rebuild the training container for DataMix.

### Eval environment

The eval jobs run `oellm-eval` and `lm-eval` inside the eval container. `scripts/install_eval_env.sh` installs them
on top of it, into `$WORK/eval_local`. That env has ~44k files, too many for the `$HOME` quota, so
`~/eval_local` is a link to it. The installer also creates `~/eval_venv_stub`, a stub "venv" that the eval jobs
activate.

The installer also **patches `submodules/Megatron-Bridge` in place** (`container/megatron/patch_bridge_lazy_imports.py`).
It wraps the model imports in `megatron/bridge/models/**/__init__.py` in `try/except`, so that model bridges that
don't import with the OpenEuroLLM Megatron-LM fork are skipped instead of breaking the conversion. After this,
`git status` shows ~35 modified files in that submodule. This is intended; don't revert it.

### Test run

The test runs the whole chain for real, but small, all on `dev-g`:
- **Training:** 100 iterations per scale, on 1 node.
- **Conversion:** checkpoints 50 and 100.
- **Evaluation:** both checkpoints, on task group `generic-multilingual` with 50 samples per task.

It needs training data: the prefix (the path without `.bin`) of a Megatron `.bin`/`.idx` pair tokenized with
`openeurollm/tokenizer-256k`. Step 8 asks for it, or you start the test yourself:

```bash
bash config/experiments/datamix/run_test.sh 'aux.data_path=[/path/to/bin_00_text_document]' 'aux.scales=[0.1B]'
```

- **What it runs:** `run_test.sh` checks the prerequisites and downloads the eval datasets (`prefetch_eval.sh`,
  ~1 min). Then it starts the chain in the tmux session `datamix-test` (detach: `Ctrl-b d`, reattach:
  `tmux attach -t datamix-test`).
- **Keep it running:** it submits each job only when its input is ready, and prints `All jobs finished.` at the end.
- **`dev-g` allows only one job per project**, shared with all your colleagues in the project. The test therefore
  submits its jobs one at a time, and only while nobody in the project has a job on `dev-g`; otherwise it waits.
  `small-g` and `standard-g` have no such limit but queue for days.
- **Duration:** one scale is 5 jobs, ~1 h: training ~6 min, conversions ~3 min, evals 5-15 min. All scales are
  15 jobs. `--dry-run` only renders the job scripts.
- **Results:** in `output/test/datamix/<scale>/`: `checkpoints/`, `hf/iter_000N/`, `eval/iter_000N/<timestamp>/results/*.json`.
- **When it passed:** all four `.done` files exist (`hf/iter_0000050.done`, `hf/iter_0000100.done`,
  `eval/iter_0000050.done`, `eval/iter_0000100.done`).
- **Starting again:** finished jobs are skipped. Delete `output/test/datamix` to test again from scratch.

## Your own run

Three values in `config/experiments/datamix/chain-datamix.yaml` have no default. Set them in the file or on the
command line:

```yaml
aux:
  name: fin-bin00                     # your experiment: names run dirs, SLURM jobs, W&B runs
  data_path:                          # your training data (prefix without .bin/.idx)
    - /scratch/<project>/users/<you>/data/bin_00_text_document
  eval_task_groups: [open-sci-0.01]   # all groups: submodules/oellm_evals/oellm/resources/task-groups.yaml
```

Use the same `eval_task_groups` for all mixes you want to compare. Then, from the repo root:

```bash
source ~/datamix_env.sh

# 1. download the eval datasets of your config (once, and after every change of eval_task_groups)
bash config/experiments/datamix/prefetch_eval.sh --config-name experiments/datamix/chain-datamix

# 2. start the chain in tmux and keep it running (days for the full trainings)
tmux new -s datamix
source ~/datamix_env.sh
export OUTPUT_ROOT=$WORK/autoexp_runs   # optional; default: output/ in the repo
uv run python scripts/run_autoexp.py --config-name experiments/datamix/chain-datamix
```

`prefetch_eval.sh` takes the same `--config-name` and overrides as `run_autoexp.py`. It exists because autoexp
never downloads the eval datasets itself:
- **The eval jobs run offline** on the compute nodes.
- **oellm-eval on its own downloads first:** when you use oellm-eval directly, `oellm-eval schedule` runs on the
  login node and downloads the datasets before it submits its eval jobs.
- **autoexp runs it inside the eval job**, on a compute node without internet, where that download step has to be
  skipped.

The script's header and [chain_guide.md](chain_guide.md#eval-datasets) explain this in detail.

The real runs use `standard-g` (training, 16 nodes) and `small-g` (conversion, eval), so expect queueing.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `FATAL: Unable to build from artifacts/MegatronTrainingLumi_x86_64.sif: ... no such file or directory` | The old convert-container recipe; `artifacts/` is not in git. Use `setup_lumi.sh` (step 5) |
| `could not use fakeroot: no mapping entry found in /etc/subuid` | `singularity build --fakeroot` doesn't work on LUMI; use `build_lumi_bridge_container.sh` |
| `No such file or directory: '/hub/models--openeurollm--tokenizer-256k/refs/main'` | `HF_HOME` not set: `source ~/datamix_env.sh` |
| `Missing mandatory value: aux.data_path` / `aux.eval_task_groups` / `aux.name` | Set it in `chain-datamix.yaml` or on the command line ([Your own run](#your-own-run)) |
| `sbatch: error: MaxSubmitJobsPerAccount` / `Job violates accounting/QOS policy` | Someone in the project had a `dev-g` job (limit: 1 per project). autoexp drops a rejected job; start the test again, it skips finished jobs |
| An eval job can't find a dataset (`Couldn't reach ...`) | Its datasets weren't downloaded: run `prefetch_eval.sh` with the same config and overrides |
| `[ERROR] 'cache_position' is part of Qwen3ASR...forward's signature, but not documented` in a convert log | Harmless: a docstring check of an unrelated model in Megatron-Bridge |
| One eval takes much longer than another, often minutes before the first task | At start, lm-eval reads thousands of task definitions from `~/eval_local` on Lustre: 0.5-8 min, depending on Lustre load. Within the time limits |
| `git status` shows modified files in `submodules/Megatron-Bridge` | Intended: the eval installer's patch ([Eval environment](#eval-environment)) |
| A new tmux window can't find `uv`, or the config fails to load | The env isn't loaded in it: `source ~/datamix_env.sh` |

## Another LUMI project

Choose your project in step 2, and in step 3 a containers dir you can read. The setup writes them to
`~/datamix_env.sh` as `DATAMIX_PROJECT` and `DATAMIX_CONTAINERS`, which all jobs use.

- **Containers:** the train and eval containers can't be built on LUMI. Copy them from
  `/scratch/project_465002530/containers/` (needs membership in that project) into your containers dir.
- **Tokenizer and HF cache:** step 7 downloads the tokenizer into a new HF cache.

This path has not been tested with a real second project yet.
